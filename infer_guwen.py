#!/usr/bin/env python3
"""
Guwen-Seg / Guwen-Punc Baseline Inference Pipeline
====================================================

Runs the user's pre-chunked Parquet dataset through the pre-trained
``ethanyt/guwen-seg`` (sentence segmentation) or ``ethanyt/guwen-punc``
(punctuation restoration) models from HuggingFace, then evaluates
against the user's ground-truth labels.

Label Mapping — Segmentation
-----------------------------
Guwen-seg labels:  O (no boundary), B (sentence boundary)
User labels:       B, M, E, S  (word segmentation — BMES scheme)

Conversion (gold BMES → O/B for fair evaluation):
  - First character → always ``B``
  - After ``E`` or ``S`` (end of a word) → next char is ``B``
  - Otherwise → ``O``

Label Mapping — Punctuation
-----------------------------
Guwen-punc labels:  O, B-,  B-.  B-?  B-!  B-\\  B-:  B-;
User labels:        O, ，    。   ？    ！    、    ：   ；

These map 1-to-1, so we convert guwen-punc predictions to the user's
label space and evaluate directly.

Guwen-punc ID → User label:
  0  O     → O
  1  B-,   → ，
  2  B-.   → 。
  3  B-?   → ？
  4  B-!   → ！
  5  B-\\  → 、
  6  B-:   → ：
  7  B-;   → ；
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
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Project imports — reuse data loading + streaming collate utilities.
from src.config import TaskConfig
from src.data import load_streaming_dataset, streaming_collate_fn
from src.utils import apply_punctuation_labels

# Reuse stitching infrastructure from the sliding-window pipeline.
from infer_sliding_window import (
    build_raw_to_pred_map,
    strip_last_sentence,
    stitch_documents,
    write_documents_jsonl,
    compute_stitched_metrics,
    OVERLAP,
)


# ============================================================================
# CONSTANTS
# ============================================================================

# HuggingFace model identifiers.
GUWEN_SEG_MODEL = "ethanyt/guwen-seg"
GUWEN_PUNC_MODEL = "ethanyt/guwen-punc"

# ---------- Segmentation label scheme (guwen-seg) ----------
GUWEN_SEG_LABELS = ["O", "B"]

# ---------- Punctuation label scheme (guwen-punc) ----------
GUWEN_PUNC_LABELS = ["O", "B-,", "B-.", "B-?", "B-!", "B-\\", "B-:", "B-;"]

# Mapping from guwen-punc model label ID → user's Chinese punctuation label.
GUWEN_PUNC_ID_TO_USER = {
    0: "O",
    1: "，",
    2: "。",
    3: "？",
    4: "！",
    5: "、",
    6: "：",
    7: "；",
}

# Reverse: user Chinese punctuation → guwen-punc label (for gold conversion).
USER_TO_GUWEN_PUNC = {v: k for k, v in zip(GUWEN_PUNC_LABELS, GUWEN_PUNC_ID_TO_USER.values())}
# {'O': 'O', '，': 'B-,', '。': 'B-.', '？': 'B-?', '！': 'B-!', '、': 'B-\\', '：': 'B-:', '；': 'B-;'}


# ============================================================================
# LABEL CONVERSION:  User BMES  →  Guwen-Seg O/B  (segmentation only)
# ============================================================================

def bmes_to_ob(bmes_labels: List[str]) -> List[str]:
    """Convert a sequence of BMES sentence-segmentation labels to guwen-seg
    style O/B labels.

    Logic
    -----
    Guwen-seg marks ``B`` at positions where punctuation (sentence boundary)
    exists, and ``O`` elsewhere.

    - ``E`` (end of sentence) → ``B`` (punctuation here)
    - ``S`` (single-char sentence) → ``B`` (punctuation here)
    - ``B`` (beginning of sentence) → ``O``
    - ``M`` (middle of sentence) → ``O``
    """
    if not bmes_labels:
        return []

    ob_labels: List[str] = []
    for lbl in bmes_labels:
        if lbl in ("E", "S"):
            ob_labels.append("B")
        else:
            ob_labels.append("O")
    return ob_labels


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_guwen(
    examples,
    tokenizer,
    task_config: TaskConfig,
    task_type: str,
    max_length: int = 510,
    keep_raw: bool = False,
):
    """Tokenize examples for a guwen model (seg or punc).

    For **segmentation**: converts user BMES → O/B for the labels tensor
    and stores O/B as ``raw_labels``.

    For **punctuation**: maps user Chinese punctuation labels to
    ``task_config.label2id`` (which uses the user's own label set) and
    stores the original labels as ``raw_labels``.
    """
    tokenized_inputs = tokenizer(
        [list(text) for text in examples["text"]],
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    labels = []
    raw_labels_out = []

    for i, gold_seq in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        if task_type == "segmentation":
            # Convert BMES → O/B for evaluation in guwen-seg's label space.
            eval_seq = bmes_to_ob(gold_seq)
        else:
            # Punctuation: gold labels are already in user's label space,
            # which matches task_config.labels.
            eval_seq = gold_seq

        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label = eval_seq[word_id]
                lid = task_config.label2id.get(label)
                label_ids.append(lid if lid is not None else -100)
        labels.append(label_ids)

        if keep_raw:
            raw_labels_out.append(eval_seq)

    tokenized_inputs["labels"] = labels

    if keep_raw:
        tokenized_inputs["raw_text"] = examples["text"]
        tokenized_inputs["raw_labels"] = raw_labels_out

    return tokenized_inputs


# ============================================================================
# PER-SAMPLE DECODING  (token predictions → char-level labels)
# ============================================================================

def decode_sample_guwen(
    raw_text: str,
    token_preds: torch.Tensor,
    tokenizer,
    task_type: str,
    max_length: int,
) -> Tuple[str, List[str]]:
    """Decode a single sample from token-level predictions back to
    character-level labels for a guwen model.

    For **segmentation** (guwen-seg):
      - Predictions are O/B.  Insert " | " *before* each ``B`` (except
        the first character).

    For **punctuation** (guwen-punc):
      - Map guwen-punc label IDs to user's Chinese punctuation labels
        via ``GUWEN_PUNC_ID_TO_USER``, then insert punctuation marks
        after the character.

    Returns
    -------
    pred_text : str
        The text with segmentation/punctuation applied.
    pred_labels : list[str]
        One label per raw character (in the evaluation label space).
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
    preds_cpu = token_preds.cpu()

    if task_type == "segmentation":
        # Guwen-seg: model IDs map directly to O/B.
        pred_labels: List[str] = []
        for token_idx, word_id in enumerate(word_ids):
            if word_id is not None and token_idx < len(preds_cpu):
                label_id = preds_cpu[token_idx].item()
                pred_labels.append(GUWEN_SEG_LABELS[label_id] if label_id < len(GUWEN_SEG_LABELS) else "O")

        # Build predicted text: insert " | " before every B (except first).
        sep = " | "
        output = []
        for i, (ch, lbl) in enumerate(zip(raw_text, pred_labels)):
            if i > 0 and lbl == "B":
                output.append(sep)
            output.append(ch)
        pred_text = "".join(output)

    else:
        # Guwen-punc: map model IDs → user Chinese punctuation labels.
        pred_labels: List[str] = []
        for token_idx, word_id in enumerate(word_ids):
            if word_id is not None and token_idx < len(preds_cpu):
                label_id = preds_cpu[token_idx].item()
                user_label = GUWEN_PUNC_ID_TO_USER.get(label_id, "O")
                pred_labels.append(user_label)

        # Build predicted text using the standard punctuation insertion.
        pred_text = apply_punctuation_labels(raw_text, pred_labels)

    return pred_text, pred_labels


# ============================================================================
# BATCH INFERENCE GENERATOR
# ============================================================================

def infer_batches_guwen(
    dataloader: DataLoader,
    model: torch.nn.Module,
    tokenizer,
    task_type: str,
    device: str,
    max_length: int,
    use_amp: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Generator[Tuple[str, str, List[str], List[str]], None, None]:
    """Yield (raw_text, pred_text, pred_labels, gold_labels) for every
    sample in the streaming dataloader, running inference in batches.
    """
    model.eval()
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = input_ids.size(0)
            raw_texts: List[str] = batch["raw_text"]
            raw_labels_list: List[List[str]] = batch.get(
                "raw_labels", [None] * batch_size
            )

            # --- Forward pass ---
            with torch.amp.autocast(
                "cuda", enabled=use_amp and torch.cuda.is_available()
            ):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            # Standard HuggingFace token classification model.
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # --- Per-sample decoding ---
            for i in range(batch_size):
                raw_text = raw_texts[i]
                gold_labels = (
                    raw_labels_list[i]
                    if raw_labels_list[i] is not None
                    else []
                )
                pred_text, pred_labels = decode_sample_guwen(
                    raw_text=raw_text,
                    token_preds=predictions[i],
                    tokenizer=tokenizer,
                    task_type=task_type,
                    max_length=max_length,
                )
                yield raw_text, pred_text, pred_labels, gold_labels
                total_samples += 1

            # --- Free GPU memory ---
            del outputs, input_ids, attention_mask, predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if logger:
        logger.info(
            f"  Inference complete — {total_samples} fragments processed."
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Guwen-Seg / Guwen-Punc Baseline Inference — run user dataset "
            "through ethanyt/guwen-seg or ethanyt/guwen-punc and evaluate."
        )
    )

    # --- Task ---
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["segmentation", "punctuation"],
        help="Task type: segmentation (guwen-seg) or punctuation (guwen-punc)",
    )

    # --- Data ---
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing split folders with parquet files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split folder name (default: test)",
    )

    # --- Model ---
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "HuggingFace model name (default: auto-select based on task — "
            f"{GUWEN_SEG_MODEL} for segmentation, {GUWEN_PUNC_MODEL} for punctuation)"
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=510,
        help="Max sequence length (guwen models support up to 510+2=512)",
    )

    # --- Inference ---
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help="Overlap size used during data chunking (default: 128)",
    )

    # --- Output ---
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--log_dir", type=str, default="logs")

    args = parser.parse_args()

    # Auto-select model based on task if not specified.
    if args.model_name is None:
        args.model_name = (
            GUWEN_SEG_MODEL if args.task == "segmentation" else GUWEN_PUNC_MODEL
        )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_file = os.path.join(
        args.log_dir, f"infer_guwen_{args.task}_{args.split}.log"
    )
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
    logger.info("GUWEN BASELINE INFERENCE")
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
    if args.task == "segmentation":
        # Evaluate in guwen-seg's O/B label space.
        task_config = TaskConfig.create(
            task_name="segmentation",
            labels=GUWEN_SEG_LABELS,  # ["O", "B"]
            ignore_labels=[],
        )
    else:
        # Evaluate in the user's Chinese punctuation label space.
        # This is a 1-to-1 mapping from guwen-punc labels.
        task_config = TaskConfig.create(
            task_name="punctuation",
            labels=["O", "，", "。", "：", "、", "；", "？", "！"],
            ignore_labels=["O"],
        )

    logger.info(f"\n✓ Task: {args.task} (guwen baseline)")
    logger.info(f"  Evaluation labels: {task_config.labels}")
    logger.info(f"  Ignore labels for macro avg: {task_config.ignore_labels}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    logger.info(f"\n✓ Loading tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ------------------------------------------------------------------
    # Model  (standard HuggingFace RobertaForTokenClassification)
    # ------------------------------------------------------------------
    logger.info(f"\n✓ Loading model from: {args.model_name}")
    model = AutoModelForTokenClassification.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✓ Model loaded — {num_params:,} parameters")
    logger.info(f"  Model config labels: {model.config.id2label}")

    # ------------------------------------------------------------------
    # Streaming data
    # ------------------------------------------------------------------
    logger.info(f"\n✓ Loading data from: {args.data_dir}/{args.split}")

    data_path = Path(args.data_dir) / args.split
    parquet_files = list(data_path.glob("*.parquet"))
    total_samples = sum(
        pq.read_metadata(str(f)).num_rows for f in parquet_files
    )
    logger.info(f"  Total fragments: {total_samples}")

    raw_dataset = load_streaming_dataset(args.data_dir, args.split)

    # Preprocess: tokenize with guwen tokenizer + convert labels.
    tokenized_dataset = raw_dataset.map(
        partial(
            preprocess_guwen,
            tokenizer=tokenizer,
            task_config=task_config,
            task_type=args.task,
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

    # Step 1: Batch inference generator.
    fragment_gen = infer_batches_guwen(
        dataloader=dataloader,
        model=model,
        tokenizer=tokenizer,
        task_type=args.task,
        device=device,
        max_length=args.max_length,
        use_amp=args.fp16,
        logger=logger,
    )

    # Step 2: Stateful stitcher (reuse from infer_sliding_window.py).
    doc_gen = stitch_documents(
        fragment_stream=fragment_gen,
        task_name=task_config.task_name,
        overlap=args.overlap,
        logger=logger,
    )

    # Step 3: Stream documents to JSONL + accumulate confusion matrix.
    model_short = args.model_name.split("/")[-1]  # e.g. "guwen-seg"
    output_path = os.path.join(
        args.output_dir,
        f"{args.split}_{model_short}_stitched.jsonl",
    )
    total_docs, conf_matrix, total_labels = write_documents_jsonl(
        doc_gen,
        output_path,
        task_config=task_config,
        logger=logger,
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

        # Save metrics to JSON.
        metrics_path = os.path.join(
            args.output_dir,
            f"{args.split}_{model_short}_stitched_metrics.json",
        )
        label_note = (
            "Gold BMES labels were converted to O/B for segmentation eval."
            if args.task == "segmentation"
            else "Guwen-punc B-X predictions were mapped to user Chinese "
            "punctuation labels for evaluation."
        )
        metrics_to_save = {
            "model": args.model_name,
            "task": args.task,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "per_label": metrics["per_label"],
            "total_documents": total_docs,
            "total_evaluated_labels": total_labels,
            "label_mapping_note": label_note,
        }
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(metrics_to_save, mf, ensure_ascii=False, indent=2)
        logger.info(f"\n  ✓ Metrics saved to: {metrics_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info(f"🎉 GUWEN BASELINE INFERENCE COMPLETE ({args.task})")
    logger.info(f"  Model               : {args.model_name}")
    logger.info(f"  Fragments processed : {total_samples}")
    logger.info(f"  Documents stitched  : {total_docs}")
    logger.info(f"  Output              : {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
