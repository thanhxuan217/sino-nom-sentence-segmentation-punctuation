#!/usr/bin/env python3
"""
Parallel Sliding Window Inference Pipeline
===========================================

Streams a pre-chunked Parquet dataset (max_length=256, overlap=128),
runs batch GPU inference with SikuBERT, then stitches predicted
fragments back into complete documents by dynamically detecting
document boundaries through 128-char raw overlap matching.

Algorithm summary
-----------------
For every adjacent pair of predicted fragments (prev, current):

  Case A – OVERLAP MATCH  (same document, contiguous chunks)
      • Drop the last predicted sentence of `prev` (semantically
        incomplete at the boundary).
      • Yield `prev`'s remaining valid text.
      • When emitting `current` later, skip the portion that already
        appeared in `prev`'s valid output to avoid duplication.

  Case B – OVERLAP MISMATCH  (document boundary detected)
      • `prev` is the FINAL fragment of its document → yield it in
        full (do NOT drop the last sentence).
      • Treat `current` as fragment #1 of a brand-new document (no
        overlap removal at the start).

  Case C – EOF  (end of stream)
      • The final buffered fragment is always the last fragment of the
        last document → yield it in full.

Index tracking
--------------
The model inserts punctuation/segmentation markers that change string
lengths.  Overlap is compared on the raw (original) characters.  A
mapping   raw_char_index → predicted_char_index   is built so that we
can correctly slice the predicted string after identifying the overlap
boundary in raw-character space.
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple


import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Project imports
from src.checkpoint import load_model_from_trainer_checkpoint
from src.config import TaskConfig, TrainingConfig
from src.data import load_streaming_dataset, preprocess_function, streaming_collate_fn
from src.model import SikuBERTForTokenClassification
from src.utils import apply_punctuation_labels, apply_segmentation_inline


# ============================================================================
# CONSTANTS
# ============================================================================

# Sliding-window parameters that were used during data preparation.
WINDOW_SIZE = 256   # max_length used when the parquet was created
OVERLAP     = 128   # overlap between consecutive windows


# ============================================================================
# RAW ↔ PREDICTED INDEX MAPPING
# ============================================================================

def build_raw_to_pred_map(raw_text: str, pred_text: str) -> List[int]:
    """Build a mapping from each raw character position to its position in
    the predicted (label-inserted) string.

    The predicted string is identical to the raw string except that extra
    characters (punctuation marks or segmentation separators) may have
    been inserted *after* certain raw characters.

    Returns
    -------
    raw_to_pred : list[int]
        ``raw_to_pred[i]`` is the index in ``pred_text`` where the
        i-th raw character appears.  Length == len(raw_text).
    """
    raw_to_pred: List[int] = []
    pred_idx = 0

    for raw_char in raw_text:
        # Walk the predicted string until we find the matching raw char.
        while pred_idx < len(pred_text) and pred_text[pred_idx] != raw_char:
            pred_idx += 1
        raw_to_pred.append(pred_idx)
        pred_idx += 1  # advance past the matched character

    return raw_to_pred


# ============================================================================
# SENTENCE-LEVEL UTILITIES
# ============================================================================

# Punctuation characters that mark the END of a sentence in Classical Chinese.
_SENTENCE_ENDING_PUNCTS = set("。？！")

# Segmentation separator used by apply_segmentation_inline (default " | ").
_SEG_SEP = " | "


def strip_last_sentence(
    pred_text: str,
    task_name: str,
) -> str:
    """Remove the last 'sentence' from the predicted text.

    For **punctuation** tasks a sentence is delimited by one of the
    sentence-ending punctuation marks (。？！).  We discard everything
    after (and including) the last complete sentence-ending mark.

    For **segmentation** tasks a sentence is delimited by the segment
    separator " | ".  We discard the last segment.

    If no sentence boundary is found the *entire* text is considered one
    incomplete sentence and an empty string is returned.
    """
    if task_name == "punctuation":
        # Find the last sentence-ending punctuation mark.
        last_idx = -1
        for i in range(len(pred_text) - 1, -1, -1):
            if pred_text[i] in _SENTENCE_ENDING_PUNCTS:
                last_idx = i
                break
        if last_idx == -1:
            return ""  # no sentence boundary found
        # Keep everything up to and including the sentence-ending mark.
        return pred_text[: last_idx + 1]
    else:
        # segmentation: sentences are separated by " | "
        last_sep = pred_text.rfind(_SEG_SEP)
        if last_sep == -1:
            return ""
        return pred_text[: last_sep + len(_SEG_SEP)]


# ============================================================================
# PER-SAMPLE DECODING  (token predictions → char-level labels)
# ============================================================================

def decode_sample(
    raw_text: str,
    token_preds: torch.Tensor,
    tokenizer,
    task_config: TaskConfig,
    max_length: int,
) -> Tuple[str, List[str]]:
    """Decode a single sample from token-level predictions back to
    character-level labels, then apply labels to produce the predicted text.

    Returns
    -------
    pred_text : str
        The text after label insertion (e.g. with punctuation marks).
    pred_labels : list[str]
        One label per raw character.
    """
    # Re-tokenize to get word_ids alignment.
    tokenized = tokenizer(
        list(raw_text),
        is_split_into_words=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    word_ids = tokenized.word_ids()

    pred_labels: List[str] = []
    preds_cpu = token_preds.cpu()

    for token_idx, word_id in enumerate(word_ids):
        if word_id is not None and token_idx < len(preds_cpu):
            label_id = preds_cpu[token_idx].item()
            pred_labels.append(task_config.id2label[label_id])

    # Apply labels to raw text to produce the predicted text.
    if task_config.task_name == "punctuation":
        pred_text = apply_punctuation_labels(raw_text, pred_labels)
    elif task_config.task_name == "segmentation":
        pred_text = apply_segmentation_inline(raw_text, pred_labels)
    else:
        pred_text = raw_text

    return pred_text, pred_labels


# ============================================================================
# BATCH INFERENCE GENERATOR
# ============================================================================

def infer_batches(
    dataloader: DataLoader,
    model: torch.nn.Module,
    tokenizer,
    task_config: TaskConfig,
    device: str,
    max_length: int,
    use_amp: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Generator[Tuple[str, str, List[str], List[str]], None, None]:
    """Yield (raw_text, pred_text, pred_labels, gold_labels) for every
    sample in the streaming dataloader, running inference in batches.

    This is a generator so that the downstream stitching logic can
    process results one-by-one without holding the entire dataset in RAM.

    ``gold_labels`` are the ground-truth per-character labels read
    straight from the parquet dataset (via ``keep_raw=True``).
    """
    model.eval()
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = input_ids.size(0)
            raw_texts: List[str] = batch["raw_text"]
            # Ground-truth labels per character (list of list[str]).
            raw_labels_list: List[List[str]] = batch.get("raw_labels", [None] * batch_size)

            # --- Forward pass ---
            with torch.amp.autocast("cuda", enabled=use_amp and torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if "predictions" in outputs:
                predictions = outputs["predictions"]
            else:
                predictions = torch.argmax(outputs["logits"], dim=-1)

            # --- Per-sample decoding ---
            for i in range(batch_size):
                raw_text = raw_texts[i]
                gold_labels = raw_labels_list[i] if raw_labels_list[i] is not None else []
                pred_text, pred_labels = decode_sample(
                    raw_text=raw_text,
                    token_preds=predictions[i],
                    tokenizer=tokenizer,
                    task_config=task_config,
                    max_length=max_length,
                )
                yield raw_text, pred_text, pred_labels, gold_labels
                total_samples += 1

            # --- Free GPU memory ---
            del outputs, input_ids, attention_mask, predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if logger:
        logger.info(f"  Inference complete — {total_samples} fragments processed.")


# ============================================================================
# STATEFUL DOCUMENT STITCHER
# ============================================================================

def stitch_documents(
    fragment_stream: Generator[Tuple[str, str, List[str], List[str]], None, None],
    task_name: str,
    overlap: int = OVERLAP,
    logger: Optional[logging.Logger] = None,
) -> Generator[Tuple[str, List[str], List[str]], None, None]:
    """Consume a stream of (raw_text, pred_text, pred_labels, gold_labels)
    fragments and yield fully-stitched documents together with their
    stitched gold and predicted per-character labels.

    This is the core "Parallel Sliding Window" reconstruction algorithm.
    It maintains a two-fragment buffer (prev / current) and uses the
    128-character raw overlap to detect document boundaries.

    Labels are stitched using the **same raw-character indices** as the
    text — since labels map 1:1 to raw characters, the same overlap /
    boundary logic applies without modification.

    Yields
    ------
    (pred_text, gold_labels, pred_labels) for each completed document.
    """

    # ------------------------------------------------------------------
    #  State variables
    # ------------------------------------------------------------------
    # ``doc_parts`` accumulates the valid (non-overlapping) predicted
    # text slices that belong to the *current* document being built.
    doc_parts: List[str] = []
    # Parallel accumulators for stitched labels.
    doc_gold_labels: List[str] = []
    doc_pred_labels: List[str] = []

    # The previous fragment's data.  ``None`` means we are at the very
    # start of the stream (no previous fragment yet).
    prev_raw: Optional[str] = None
    prev_pred: Optional[str] = None
    prev_pred_labels: Optional[List[str]] = None
    prev_gold_labels: Optional[List[str]] = None

    # How many raw characters at the START of ``prev_pred`` have already
    # been emitted (i.e. belong to the overlap with the fragment before
    # it).  For the first fragment in a document this is 0.
    prev_overlap_start_raw: int = 0

    doc_count = 0

    for raw_text, pred_text, pred_labels, gold_labels in fragment_stream:

        if prev_raw is None:
            # ============================================================
            # FIRST FRAGMENT EVER — just buffer it, nothing to compare.
            # ============================================================
            prev_raw = raw_text
            prev_pred = pred_text
            prev_pred_labels = pred_labels
            prev_gold_labels = gold_labels
            prev_overlap_start_raw = 0
            continue

        # ================================================================
        # COMPARE OVERLAPS between prev and current fragment.
        # ================================================================
        # The last ``overlap`` raw characters of prev should equal the
        # first ``overlap`` raw characters of current — IF they come
        # from the same document.
        prev_tail_raw = prev_raw[-overlap:]  # last 128 raw chars of prev
        curr_head_raw = raw_text[:overlap]   # first 128 raw chars of curr

        if prev_tail_raw == curr_head_raw:
            # ============================================================
            # CASE A — OVERLAP MATCH (contiguous chunks, same document)
            # ============================================================
            # 1. Remove the last sentence from prev's prediction because
            #    it was formed with incomplete context at the boundary.
            valid_prev_pred = strip_last_sentence(prev_pred, task_name)

            if valid_prev_pred:
                # 2. If prev had an overlap start (from the fragment
                #    *before* it), we need to remove that leading portion
                #    from the predicted text to avoid duplication.
                if prev_overlap_start_raw > 0:
                    # Build the raw→pred index map so we know where to
                    # slice in the predicted string.
                    r2p = build_raw_to_pred_map(prev_raw, prev_pred)
                    # The valid predicted text starts at the pred index
                    # corresponding to the first non-overlapped raw char.
                    pred_start = r2p[prev_overlap_start_raw]
                    # Slice the valid portion only.
                    valid_prev_pred = valid_prev_pred[pred_start:]

                doc_parts.append(valid_prev_pred)

            # 3. Determine how many raw chars of `current` overlap with
            #    the *valid* part of prev.  The valid part of prev
            #    covers raw chars from prev_overlap_start_raw to the end
            #    of the last complete sentence.  The raw chars that
            #    appear in both prev and current are the overlap region,
            #    but only the portion NOT already consumed by the
            #    "strip last sentence" step is relevant.
            #
            #    Since we stripped the last sentence of prev, the raw
            #    chars belonging to that stripped tail now remain ONLY in
            #    current's prediction.  The overlap region of current
            #    that duplicates prev's *valid output* starts at raw
            #    index 0 of current and extends up to the point where
            #    prev's valid raw content ends within the overlap.
            #
            #    We compute this by finding how many raw chars of prev
            #    were kept after stripping.
            if valid_prev_pred and prev_pred:
                # Compute where valid_prev_pred sits within the FULL
                # predicted string, then walk the raw→pred map to find
                # the last raw character that falls inside that region.
                r2p_full = build_raw_to_pred_map(prev_raw, prev_pred)

                # pred_start: index in full prev_pred where valid output
                #             begins (skipping the overlap-with-earlier).
                pred_start = (
                    r2p_full[prev_overlap_start_raw]
                    if prev_overlap_start_raw > 0
                    else 0
                )
                # pred_end: one-past-the-last index of valid output in
                #           the full prev_pred string.
                pred_end = pred_start + len(valid_prev_pred)

                # Walk raw indices forward from prev_overlap_start_raw
                # to find the exclusive raw-char end covered by the
                # valid predicted text.
                valid_raw_end = prev_overlap_start_raw
                for ri in range(prev_overlap_start_raw, len(prev_raw)):
                    if r2p_full[ri] >= pred_end:
                        break
                    valid_raw_end = ri + 1  # exclusive end
                else:
                    # Reached the end of prev_raw without breaking —
                    # all remaining raw chars are within valid output.
                    valid_raw_end = len(prev_raw)

                # --- Stitch labels for the valid portion of prev ---
                doc_gold_labels.extend(prev_gold_labels[prev_overlap_start_raw:valid_raw_end])
                doc_pred_labels.extend(prev_pred_labels[prev_overlap_start_raw:valid_raw_end])

                # How many raw chars of prev's tail are within current's
                # overlap region AND were kept (not stripped)?
                overlap_zone_start = len(prev_raw) - overlap
                kept_in_overlap = max(0, valid_raw_end - overlap_zone_start)
                curr_overlap_start_raw = kept_in_overlap
            else:
                # prev yielded nothing useful; current starts fresh
                # within the overlap — just skip the whole overlap.
                curr_overlap_start_raw = overlap

            # Buffer current as the new prev for the next iteration.
            prev_raw = raw_text
            prev_pred = pred_text
            prev_pred_labels = pred_labels
            prev_gold_labels = gold_labels
            prev_overlap_start_raw = curr_overlap_start_raw

        else:
            # ============================================================
            # CASE B — OVERLAP MISMATCH (document boundary detected)
            # ============================================================
            # prev is the LAST fragment of the current document.
            # Do NOT strip its last sentence — it is the true ending.

            if prev_overlap_start_raw > 0 and prev_pred:
                r2p = build_raw_to_pred_map(prev_raw, prev_pred)
                pred_start = r2p[prev_overlap_start_raw]
                doc_parts.append(prev_pred[pred_start:])
            else:
                doc_parts.append(prev_pred)

            # --- Stitch labels for the final fragment of this doc ---
            doc_gold_labels.extend(prev_gold_labels[prev_overlap_start_raw:])
            doc_pred_labels.extend(prev_pred_labels[prev_overlap_start_raw:])

            # --- Yield the completed document ---
            completed_doc = "".join(doc_parts)
            doc_count += 1
            if logger and doc_count % 500 == 0:
                logger.info(f"  Stitched document #{doc_count} ({len(completed_doc)} chars)")
            yield completed_doc, list(doc_gold_labels), list(doc_pred_labels)

            # Reset state for a new document.
            doc_parts = []
            doc_gold_labels = []
            doc_pred_labels = []

            # current becomes the first fragment of the next document.
            prev_raw = raw_text
            prev_pred = pred_text
            prev_pred_labels = pred_labels
            prev_gold_labels = gold_labels
            prev_overlap_start_raw = 0  # no overlap to skip

    # ====================================================================
    # CASE C — END OF STREAM (EOF)
    # ====================================================================
    # The last buffered fragment is the final fragment of the last
    # document.  Yield it in full (do NOT strip last sentence).
    if prev_pred is not None:
        if prev_overlap_start_raw > 0:
            r2p = build_raw_to_pred_map(prev_raw, prev_pred)
            pred_start = r2p[prev_overlap_start_raw]
            doc_parts.append(prev_pred[pred_start:])
        else:
            doc_parts.append(prev_pred)

        # --- Stitch labels for the last fragment ---
        doc_gold_labels.extend(prev_gold_labels[prev_overlap_start_raw:])
        doc_pred_labels.extend(prev_pred_labels[prev_overlap_start_raw:])

    if doc_parts:
        completed_doc = "".join(doc_parts)
        doc_count += 1
        yield completed_doc, list(doc_gold_labels), list(doc_pred_labels)

    if logger:
        logger.info(f"  ✓ Stitching complete — {doc_count} documents reconstructed.")


# ============================================================================
# OUTPUT WRITER
# ============================================================================

def write_documents_jsonl(
    doc_stream: Generator[Tuple[str, List[str], List[str]], None, None],
    output_path: str,
    task_config: TaskConfig,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, np.ndarray, int]:
    """Stream completed documents to a JSONL file and accumulate a
    confusion matrix for post-hoc metric computation.

    Instead of keeping all labels in RAM, we update a small
    ``num_labels × num_labels`` confusion matrix per document.

    Each output line is a JSON object with keys:
        doc_id, text, gold_labels, pred_labels

    Returns
    -------
    (total_docs, conf_matrix, total_labels)
        conf_matrix : np.ndarray of shape (num_labels, num_labels)
        total_labels : total number of label pairs evaluated
    """
    num_labels = task_config.num_labels
    conf_matrix = np.zeros((num_labels, num_labels), dtype=np.int64)
    total = 0
    total_labels = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for doc_text, gold_labels, pred_labels in doc_stream:
            record = {
                "doc_id": total,
                "text": doc_text,
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # --- Update confusion matrix (streaming, O(1) RAM) ---
            min_len = min(len(gold_labels), len(pred_labels))
            for k in range(min_len):
                g_id = task_config.label2id.get(gold_labels[k])
                p_id = task_config.label2id.get(pred_labels[k])
                if g_id is not None and p_id is not None:
                    conf_matrix[g_id, p_id] += 1
                    total_labels += 1

            total += 1

    if logger:
        logger.info(f"  ✓ Wrote {total} documents to {output_path}")
    return total, conf_matrix, total_labels


# ============================================================================
# POST-STITCH METRICS
# ============================================================================

def compute_stitched_metrics(
    conf_matrix: np.ndarray,
    task_config: TaskConfig,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """Compute Precision, Recall, F1 from an accumulated confusion matrix.

    Uses the same methodology as ``evaluate_model`` in
    ``src/evaluation.py`` — per-label P/R/F1 derived from the confusion
    matrix, then macro-averaged.

    Labels listed in ``task_config.ignore_labels`` (e.g. "O" for
    punctuation) are excluded from the macro average.

    Returns a dict with ``precision``, ``recall``, ``f1``,
    ``per_label`` metrics, and a ``classification_report`` string.
    """
    num_labels = task_config.num_labels

    if conf_matrix.sum() == 0:
        if logger:
            logger.warning("  ⚠ Confusion matrix is empty — skipping metrics.")
        return {}

    ignore_ids = set()
    if task_config.ignore_labels:
        ignore_ids = {
            task_config.label2id[lbl]
            for lbl in task_config.ignore_labels
            if lbl in task_config.label2id
        }

    per_label: Dict[str, Dict] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for i in range(num_labels):
        tp = int(conf_matrix[i, i])
        fp = int(conf_matrix[:, i].sum() - tp)
        fn = int(conf_matrix[i, :].sum() - tp)
        support = int(conf_matrix[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        label_name = task_config.id2label.get(i, str(i))
        per_label[label_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

        # Only include non-ignored labels with support for macro average.
        if support > 0 and i not in ignore_ids:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # --- Build a text report similar to classification_report ---
    report_lines = [
        f"{'':>14s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}",
        "",
    ]
    for i in range(num_labels):
        label_name = task_config.id2label.get(i, str(i))
        m = per_label[label_name]
        tag = "  *" if i in ignore_ids else "   "
        report_lines.append(
            f"{label_name:>14s} {m['precision']:10.4f} {m['recall']:10.4f} "
            f"{m['f1']:10.4f} {m['support']:10d}{tag}"
        )
    report_lines.append("")
    report_lines.append(
        f"{'macro avg':>14s} {macro_precision:10.4f} {macro_recall:10.4f} "
        f"{macro_f1:10.4f} {int(conf_matrix.sum()):10d}"
    )
    if ignore_ids:
        ignored_names = [task_config.id2label[i] for i in sorted(ignore_ids)]
        report_lines.append(f"\n  (* = excluded from macro avg: {ignored_names})")
    report_str = "\n".join(report_lines)

    return {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "per_label": per_label,
        "classification_report": report_str,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Set multiprocessing start method for CUDA compatibility.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Parallel Sliding Window Inference — stream, predict, stitch."
    )

    # --- Required ---
    parser.add_argument("--task", type=str, required=True,
                        choices=["punctuation", "segmentation"],
                        help="Task type")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing split folders with parquet files")
    parser.add_argument("--split", type=str, default="test",
                        help="Split folder name (default: test)")

    # --- Model ---
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikubert")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer path/name (default: same as model_name). "
                             "Use this to load an extended vocab tokenizer.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_type", type=str, default="cnn",
                        choices=["softmax", "crf", "cnn"])
    parser.add_argument("--cnn_kernel_sizes", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--cnn_num_filters", type=int, default=256)

    # --- QLoRA ---
    parser.add_argument("--use_qlora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["query", "key", "value"])

    # --- Inference ---
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--overlap", type=int, default=OVERLAP,
                        help="Overlap size used during data chunking (default: 128)")

    # --- Output ---
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--log_dir", type=str, default="logs")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_file = os.path.join(args.log_dir, f"infer_{args.task}_{args.split}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("PARALLEL SLIDING WINDOW INFERENCE")
    logger.info("=" * 70)
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 70)
    logger.info(f"  Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Task configuration
    # ------------------------------------------------------------------
    if args.task == "punctuation":
        task_config = TaskConfig.create(
            task_name="punctuation",
            labels=["O", "，", "。", "：", "、", "；", "？", "！"],
            ignore_labels=["O"],
        )
    else:
        task_config = TaskConfig.create(
            task_name="segmentation",
            labels=["B", "M", "E", "S"],
            ignore_labels=[],
        )

    logger.info(f"\n✓ Task: {task_config.task_name}")
    logger.info(f"  Labels: {task_config.labels}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer_path = args.tokenizer_name or args.model_name
    logger.info(f"\n✓ Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info(f"\n✓ Creating model (head_type={args.head_type}, "
                f"use_qlora={args.use_qlora})...")

    qlora_config = None
    if args.use_qlora:
        qlora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

    model = SikuBERTForTokenClassification(
        model_name=args.model_name,
        num_labels=task_config.num_labels,
        dropout=args.dropout,
        head_type=args.head_type,
        cnn_kernel_sizes=args.cnn_kernel_sizes,
        cnn_num_filters=args.cnn_num_filters,
        use_qlora=args.use_qlora,
        qlora_config=qlora_config,
    )
    
    # Resize embedding layer if tokenizer vocab is larger than model's default
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"  Embedding size aligned to tokenizer vocab: {len(tokenizer)}")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    logger.info(f"\n✓ Loading checkpoint from: {args.model_path}")
    model = load_model_from_trainer_checkpoint(
        checkpoint_path=args.model_path,
        model=model,
        logger=logger,
    )

    # Move to device (same logic as evaluate.py).
    if not args.use_qlora:
        model = model.to(device)
    else:
        if model.cnn_layer is not None:
            model.cnn_layer = model.cnn_layer.to(device)
        model.classifier = model.classifier.to(device)
        model.dropout = model.dropout.to(device)
        if model.crf is not None:
            model.crf = model.crf.to(device)

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✓ Model loaded — {num_params:,} parameters")

    # ------------------------------------------------------------------
    # Streaming data
    # ------------------------------------------------------------------
    logger.info(f"\n✓ Loading data from: {args.data_dir}/{args.split}")

    # Count total samples for progress info (metadata only, no RAM load).
    data_path = Path(args.data_dir) / args.split
    parquet_files = list(data_path.glob("*.parquet"))
    total_samples = sum(pq.read_metadata(str(f)).num_rows for f in parquet_files)
    logger.info(f"  Total fragments: {total_samples}")

    raw_dataset = load_streaming_dataset(args.data_dir, args.split)

    # Preprocess (tokenize + align labels), keep raw_text for stitching.
    tokenized_dataset = raw_dataset.map(
        partial(
            preprocess_function,
            tokenizer=tokenizer,
            task_config=task_config,
            max_length=args.max_length,
            keep_raw=True,
        ),
        batched=True,
        remove_columns=["text", "labels"],
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=streaming_collate_fn,
    )

    # ------------------------------------------------------------------
    # PIPELINE:  Stream → Infer → Stitch → Write
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STARTING INFERENCE + STITCHING PIPELINE")
    logger.info("=" * 70)

    # Step 1: Batch inference generator (yields per-fragment results).
    fragment_gen = infer_batches(
        dataloader=dataloader,
        model=model,
        tokenizer=tokenizer,
        task_config=task_config,
        device=device,
        max_length=args.max_length,
        use_amp=args.fp16,
        logger=logger,
    )

    # Step 2: Stateful stitcher (consumes fragments, yields documents).
    doc_gen = stitch_documents(
        fragment_stream=fragment_gen,
        task_name=task_config.task_name,
        overlap=args.overlap,
        logger=logger,
    )

    # Step 3: Stream documents to JSONL output + accumulate confusion matrix.
    output_path = os.path.join(
        args.output_dir,
        f"{args.split}_{args.task}_stitched.jsonl",
    )
    total_docs, conf_matrix, total_labels = write_documents_jsonl(
        doc_gen, output_path, task_config=task_config, logger=logger,
    )

    # ------------------------------------------------------------------
    # Post-stitch metrics  (Precision / Recall / F1)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("POST-STITCH EVALUATION METRICS")
    logger.info("=" * 70)

    metrics = compute_stitched_metrics(
        conf_matrix=conf_matrix,
        task_config=task_config,
        logger=logger,
    )

    if metrics:
        logger.info(f"\n  Macro Precision : {metrics['precision']:.4f}")
        logger.info(f"  Macro Recall    : {metrics['recall']:.4f}")
        logger.info(f"  Macro F1        : {metrics['f1']:.4f}")
        logger.info("\n" + "-" * 60)
        logger.info("DETAILED CLASSIFICATION REPORT (Per-Label)")
        logger.info("-" * 60)
        logger.info("\n" + metrics["classification_report"])

        # Save metrics to JSON file alongside the JSONL output.
        metrics_path = os.path.join(
            args.output_dir,
            f"{args.split}_{args.task}_stitched_metrics.json",
        )
        metrics_to_save = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "per_label": metrics["per_label"],
            "total_documents": total_docs,
            "total_evaluated_labels": total_labels,
        }
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(metrics_to_save, mf, ensure_ascii=False, indent=2)
        logger.info(f"\n  ✓ Metrics saved to: {metrics_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("🎉 INFERENCE COMPLETE")
    logger.info(f"  Fragments processed : {total_samples}")
    logger.info(f"  Documents stitched  : {total_docs}")
    logger.info(f"  Output              : {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
