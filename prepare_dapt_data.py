#!/usr/bin/env python3
"""
Data Preparation for Domain Adaptive Pretraining (DAPT)

Reads raw corpus text files, tokenizes with the extended SikuBERT tokenizer,
packs multiple sentences into fixed-length sequences, and saves as sharded
Parquet files for efficient streaming during training.

Supports two input formats:
  1. Directory of .txt files  (--input_format txt)
  2. Single JSONL file        (--input_format jsonl)

Usage:
    python prepare_dapt_data.py \
        --input_path /path/to/corpus \
        --input_format txt \
        --output_dir /path/to/dapt_data \
        --tokenizer_path sikubert_extended_vocab \
        --max_length 512 \
        --shard_size 100000 \
        --num_workers 8
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEXT READING
# ============================================================================

def iter_sentences_from_txt(input_path: str):
    """Yield sentences from a directory of .txt files.

    Each file is read and split into sentences by newlines.
    Blank lines are skipped.
    """
    txt_files = sorted(glob.glob(os.path.join(input_path, "**", "*.txt"), recursive=True))
    logger.info(f"Found {len(txt_files):,} .txt files")
    for fpath in txt_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
        except Exception as e:
            logger.warning(f"Error reading {fpath}: {e}")


def iter_sentences_from_jsonl(input_path: str, text_key: str = "text"):
    """Yield sentences from a JSONL file (one JSON object per line)."""
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_key, "").strip()
                if text:
                    yield text
            except json.JSONDecodeError:
                if line_no <= 5:
                    logger.warning(f"Skipping invalid JSON at line {line_no}")


# ============================================================================
# SEQUENCE PACKING
# ============================================================================

def pack_sentences(sentences, tokenizer, max_length: int):
    """Pack multiple sentences into sequences of exactly `max_length` tokens.

    Uses [CLS] at the start and [SEP] between sentences and at the end.
    This preserves sentence boundary information which benefits downstream
    sentence segmentation tasks.

    Yields:
        dict with 'input_ids' (list[int]) and 'attention_mask' (list[int])
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    current_ids = [cls_id]

    for sentence in sentences:
        # Tokenize character-by-character (Classical Chinese)
        sent_ids = tokenizer.encode(
            list(sentence),
            is_split_into_words=True,
            add_special_tokens=False,
        )
        if not sent_ids:
            continue

        # Will this sentence fit?  Need room for sent_ids + [SEP]
        needed = len(sent_ids) + 1  # +1 for [SEP]
        space_left = max_length - len(current_ids)

        if needed > space_left:
            # Flush current buffer
            if len(current_ids) > 1:  # more than just [CLS]
                # Close with [SEP] if not already there
                if current_ids[-1] != sep_id:
                    current_ids.append(sep_id)
                yield _pad_and_mask(current_ids, pad_id, max_length)
            current_ids = [cls_id]
            space_left = max_length - 1

        # Handle sentences longer than max_length: truncate
        if len(sent_ids) + 1 > space_left:
            sent_ids = sent_ids[: space_left - 1]

        current_ids.extend(sent_ids)
        current_ids.append(sep_id)

    # Flush remaining
    if len(current_ids) > 1:
        if current_ids[-1] != sep_id:
            current_ids.append(sep_id)
        yield _pad_and_mask(current_ids, pad_id, max_length)


def _pad_and_mask(token_ids, pad_id, max_length):
    """Pad/truncate token_ids and build attention_mask."""
    token_ids = token_ids[:max_length]
    length = len(token_ids)
    attention_mask = [1] * length + [0] * (max_length - length)
    token_ids = token_ids + [pad_id] * (max_length - length)
    return {
        "input_ids": token_ids,
        "attention_mask": attention_mask,
    }


# ============================================================================
# PARQUET WRITING
# ============================================================================

def write_shard(records, output_path):
    """Write a list of records to a Parquet file."""
    table = pa.table({
        "input_ids": pa.array([r["input_ids"] for r in records], type=pa.list_(pa.int32())),
        "attention_mask": pa.array([r["attention_mask"] for r in records], type=pa.list_(pa.int32())),
    })
    pq.write_table(table, output_path, compression="snappy")
    return len(records)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare packed sequences for DAPT from raw corpus."
    )
    parser.add_argument("--input_path", required=True,
                        help="Path to corpus directory (txt) or JSONL file")
    parser.add_argument("--input_format", choices=["txt", "jsonl"], default="txt",
                        help="Input format: 'txt' (dir of .txt) or 'jsonl'")
    parser.add_argument("--text_key", default="text",
                        help="JSON key for text field (jsonl mode)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for parquet shards")
    parser.add_argument("--tokenizer_path", required=True,
                        help="Path to tokenizer (directory or HF model name)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length for packing")
    parser.add_argument("--shard_size", type=int, default=100_000,
                        help="Number of sequences per parquet shard")
    args = parser.parse_args()

    # ---- Output directory ----
    train_dir = os.path.join(args.output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    # ---- Tokenizer ----
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    logger.info(f"Tokenizer vocab size: {len(tokenizer):,}")

    # ---- Sentence iterator ----
    if args.input_format == "txt":
        sentence_iter = iter_sentences_from_txt(args.input_path)
    else:
        sentence_iter = iter_sentences_from_jsonl(args.input_path, args.text_key)

    # ---- Pack and shard ----
    shard_idx = 0
    buffer = []
    total_sequences = 0
    total_sentences_read = 0

    logger.info(f"Packing sentences into sequences of {args.max_length} tokens...")

    # We need to batch sentences for packing
    SENTENCE_BATCH = 50_000  # Process sentences in batches
    sentence_batch = []

    for sentence in sentence_iter:
        total_sentences_read += 1
        sentence_batch.append(sentence)

        if total_sentences_read % 1_000_000 == 0:
            logger.info(f"  Read {total_sentences_read:,} sentences...")

        if len(sentence_batch) >= SENTENCE_BATCH:
            for record in pack_sentences(sentence_batch, tokenizer, args.max_length):
                buffer.append(record)

                if len(buffer) >= args.shard_size:
                    shard_path = os.path.join(train_dir, f"shard_{shard_idx:05d}.parquet")
                    n = write_shard(buffer, shard_path)
                    total_sequences += n
                    logger.info(
                        f"  Wrote shard {shard_idx} ({n:,} sequences) → {shard_path}"
                    )
                    shard_idx += 1
                    buffer = []

            sentence_batch = []

    # Flush remaining sentences
    if sentence_batch:
        for record in pack_sentences(sentence_batch, tokenizer, args.max_length):
            buffer.append(record)

    # Flush remaining buffer
    if buffer:
        shard_path = os.path.join(train_dir, f"shard_{shard_idx:05d}.parquet")
        n = write_shard(buffer, shard_path)
        total_sequences += n
        logger.info(f"  Wrote final shard {shard_idx} ({n:,} sequences) → {shard_path}")

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total sentences read : {total_sentences_read:,}")
    logger.info(f"  Total sequences      : {total_sequences:,}")
    logger.info(f"  Sequence length      : {args.max_length}")
    logger.info(f"  Number of shards     : {shard_idx + 1}")
    logger.info(f"  Output directory     : {train_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
