#!/usr/bin/env python3
"""
Domain-Level Evaluation Script
===============================

Loads a trained SikuBERT model, runs sliding-window inference on parquet
test data, stitches fragments into complete documents, and computes
per-domain Precision / Recall / F1 metrics for debugging.

Outputs:
  1. {split}_{task}_domain_eval.jsonl  — per-document gold vs predicted text
  2. {split}_{task}_domain_metrics.json — per-domain metric breakdown
  3. Console / log summary table sorted by F1 score
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Project imports
from src.checkpoint import load_model_from_trainer_checkpoint
from src.config import TaskConfig
from src.data import load_streaming_dataset, preprocess_function, streaming_collate_fn
from src.model import SikuBERTForTokenClassification
from src.utils import apply_punctuation_labels, apply_segmentation_inline

# Reuse sliding-window pipeline from infer_sliding_window
from infer_sliding_window import (
    OVERLAP,
    WINDOW_SIZE,
    decode_sample,
    infer_batches,
    stitch_documents,
)


# ============================================================================
# PER-DOMAIN METRIC COMPUTATION
# ============================================================================

def compute_domain_metrics(
    domain_conf_matrices: Dict[str, np.ndarray],
    task_config: TaskConfig,
) -> Dict[str, Dict]:
    """Compute Precision/Recall/F1 for each domain from per-domain
    confusion matrices.

    Returns
    -------
    dict mapping domain name → {precision, recall, f1, support, per_label: {...}}
    """
    ignore_ids = set()
    if task_config.ignore_labels:
        ignore_ids = {
            task_config.label2id[lbl]
            for lbl in task_config.ignore_labels
            if lbl in task_config.label2id
        }

    results = {}
    num_labels = task_config.num_labels

    for domain, conf_matrix in domain_conf_matrices.items():
        per_label = {}
        precisions, recalls, f1s = [], [], []

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
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }

            if support > 0 and i not in ignore_ids:
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

        macro_precision = float(np.mean(precisions)) if precisions else 0.0
        macro_recall = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        results[domain] = {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
            "support": int(conf_matrix.sum()),
            "num_docs": 0,  # will be filled by caller
            "per_label": per_label,
        }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Domain-Level Evaluation — per-domain metrics + full text output"
    )

    # --- Required ---
    parser.add_argument("--task", type=str, required=True,
                        choices=["punctuation", "segmentation"])
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing split folders with parquet files")
    parser.add_argument("--split", type=str, default="test")

    # --- Model ---
    parser.add_argument("--model_name", type=str, default="SIKU-BERT/sikubert")
    parser.add_argument("--tokenizer_name", type=str, default=None)
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
    parser.add_argument("--overlap", type=int, default=OVERLAP)

    # --- Output ---
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--log_dir", type=str, default="logs")

    # --- Debug ---
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Limit number of documents for quick debugging")
    parser.add_argument("--sample_per_domain", type=int, default=3,
                        help="Number of sample outputs to print per domain")

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
    log_file = os.path.join(args.log_dir, f"eval_domain_{args.task}_{args.split}.log")
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
    logger.info("DOMAIN-LEVEL EVALUATION")
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

    data_path = Path(args.data_dir) / args.split
    parquet_files = list(data_path.glob("*.parquet"))
    total_samples = sum(pq.read_metadata(str(f)).num_rows for f in parquet_files)
    logger.info(f"  Total fragments: {total_samples}")

    raw_dataset = load_streaming_dataset(args.data_dir, args.split)

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
    # PIPELINE: Stream → Infer → Stitch → Per-Domain Eval
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STARTING DOMAIN-LEVEL EVALUATION PIPELINE")
    logger.info("=" * 70)

    # Step 1: Batch inference generator
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

    # Step 2: Stitch fragments into documents
    doc_gen = stitch_documents(
        fragment_stream=fragment_gen,
        task_name=task_config.task_name,
        overlap=args.overlap,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Step 3: Process stitched documents — per-domain metrics + output
    # ------------------------------------------------------------------
    num_labels = task_config.num_labels

    # Per-domain confusion matrices
    domain_conf_matrices: Dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((num_labels, num_labels), dtype=np.int64)
    )
    # Overall confusion matrix
    overall_conf_matrix = np.zeros((num_labels, num_labels), dtype=np.int64)

    # Per-domain document counts
    domain_doc_counts: Dict[str, int] = defaultdict(int)

    # Per-domain sample outputs (for logging)
    domain_samples: Dict[str, List[dict]] = defaultdict(list)

    output_path = os.path.join(
        args.output_dir,
        f"{args.split}_{args.task}_domain_eval.jsonl",
    )

    total_docs = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for doc_text, gold_labels, pred_labels, domain, filename in doc_gen:
            # Build gold text from raw chars + gold labels
            # doc_text is already the predicted text (with punctuation inserted)
            # We need the raw text (without punctuation) to apply gold labels
            # Since stitching gives us pred_text, we need raw chars.
            # The gold_labels and pred_labels are per-raw-character, so we can
            # extract raw text by removing inserted punctuation from doc_text.
            # Actually, we have gold_labels per raw character. Let's reconstruct
            # the raw text from pred_labels + doc_text by stripping inserted chars.
            
            # Simpler approach: compute raw_text from pred_labels
            # Since pred_labels has one label per raw character, len(pred_labels)
            # = number of raw characters. We can extract raw chars from doc_text.
            raw_chars = []
            pi = 0
            for pred_label in pred_labels:
                if pi < len(doc_text):
                    raw_chars.append(doc_text[pi])
                    pi += 1
                    # Skip the inserted punctuation character
                    if pred_label != "O" and task_config.task_name == "punctuation":
                        pi += 1  # skip the inserted punct
                    elif task_config.task_name == "segmentation" and pred_label in ("E", "S"):
                        # skip " | " separator
                        pi += 3
            
            raw_text = "".join(raw_chars)

            # Now apply gold labels to raw text
            if task_config.task_name == "punctuation":
                gold_text = apply_punctuation_labels(raw_text, gold_labels)
            elif task_config.task_name == "segmentation":
                gold_text = apply_segmentation_inline(raw_text, gold_labels)
            else:
                gold_text = raw_text

            domain = domain or "unknown"

            record = {
                "doc_id": total_docs,
                "domain": domain,
                "filename": filename,
                "raw_text": raw_text,
                "gold_text": gold_text,
                "pred_text": doc_text,
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Update confusion matrices
            min_len = min(len(gold_labels), len(pred_labels))
            for k in range(min_len):
                g_id = task_config.label2id.get(gold_labels[k])
                p_id = task_config.label2id.get(pred_labels[k])
                if g_id is not None and p_id is not None:
                    domain_conf_matrices[domain][g_id, p_id] += 1
                    overall_conf_matrix[g_id, p_id] += 1

            domain_doc_counts[domain] += 1

            # Keep samples for logging
            if len(domain_samples[domain]) < args.sample_per_domain:
                domain_samples[domain].append({
                    "filename": filename,
                    "gold_text": gold_text[:300],
                    "pred_text": doc_text[:300],
                })

            total_docs += 1

            if total_docs % 500 == 0:
                logger.info(f"  Processed {total_docs} documents...")

            if args.max_docs is not None and total_docs >= args.max_docs:
                logger.info(f"  Reached --max_docs={args.max_docs}, stopping.")
                break

    logger.info(f"\n✓ Wrote {total_docs} documents to {output_path}")

    # ------------------------------------------------------------------
    # Compute per-domain metrics
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PER-DOMAIN EVALUATION RESULTS")
    logger.info("=" * 70)

    domain_metrics = compute_domain_metrics(domain_conf_matrices, task_config)

    # Fill in doc counts
    for domain in domain_metrics:
        domain_metrics[domain]["num_docs"] = domain_doc_counts[domain]

    # Compute overall metrics
    overall_metrics = compute_domain_metrics({"overall": overall_conf_matrix}, task_config)
    overall = overall_metrics["overall"]
    overall["num_docs"] = total_docs

    # Sort domains by F1 (ascending — worst first for debugging)
    sorted_domains = sorted(
        domain_metrics.items(),
        key=lambda x: x[1]["f1"],
        reverse=False,  # worst domains first
    )

    # Print summary table
    header = f"{'Domain':<20s} {'Docs':>6s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}"
    logger.info(f"\n{header}")
    logger.info("-" * len(header))

    for domain, metrics in sorted_domains:
        logger.info(
            f"{domain:<20s} {metrics['num_docs']:>6d} "
            f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} {metrics['support']:>10d}"
        )

    logger.info("-" * len(header))
    logger.info(
        f"{'OVERALL':<20s} {overall['num_docs']:>6d} "
        f"{overall['precision']:>10.4f} {overall['recall']:>10.4f} "
        f"{overall['f1']:>10.4f} {overall['support']:>10d}"
    )

    # ------------------------------------------------------------------
    # Print per-domain per-label details
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("DETAILED PER-DOMAIN PER-LABEL METRICS")
    logger.info("=" * 70)

    for domain, metrics in sorted_domains:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Domain: {domain}  (F1={metrics['f1']:.4f}, {metrics['num_docs']} docs)")
        logger.info(f"{'─' * 50}")

        label_header = f"  {'Label':<12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}"
        logger.info(label_header)
        logger.info("  " + "-" * (len(label_header) - 2))

        for label_name in task_config.labels:
            if label_name in metrics["per_label"]:
                lm = metrics["per_label"][label_name]
                marker = " *" if label_name in (task_config.ignore_labels or []) else ""
                logger.info(
                    f"  {label_name:<12s} {lm['precision']:>10.4f} {lm['recall']:>10.4f} "
                    f"{lm['f1']:>10.4f} {lm['support']:>10d}{marker}"
                )

    # ------------------------------------------------------------------
    # Print sample predictions per domain
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE PREDICTIONS PER DOMAIN")
    logger.info("=" * 70)

    for domain, _ in sorted_domains:
        samples = domain_samples.get(domain, [])
        if not samples:
            continue
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Domain: {domain}")
        logger.info(f"{'─' * 50}")
        for idx, sample in enumerate(samples):
            logger.info(f"\n  --- Sample {idx + 1} ({sample['filename']}) ---")
            logger.info(f"  Gold: {sample['gold_text']}")
            logger.info(f"  Pred: {sample['pred_text']}")

    # ------------------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------------------
    metrics_path = os.path.join(
        args.output_dir,
        f"{args.split}_{args.task}_domain_metrics.json",
    )

    metrics_output = {
        "overall": overall,
        "per_domain": {
            domain: metrics
            for domain, metrics in sorted(domain_metrics.items(), key=lambda x: x[1]["f1"])
        },
        "config": vars(args),
    }

    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics_output, mf, ensure_ascii=False, indent=2)

    logger.info(f"\n✓ Domain metrics saved to: {metrics_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("🎉 DOMAIN EVALUATION COMPLETE!")
    logger.info(f"  Documents evaluated : {total_docs}")
    logger.info(f"  Domains found       : {len(domain_metrics)}")
    logger.info(f"  Overall F1          : {overall['f1']:.4f}")
    logger.info(f"  Best domain         : {sorted_domains[-1][0]} (F1={sorted_domains[-1][1]['f1']:.4f})")
    logger.info(f"  Worst domain        : {sorted_domains[0][0]} (F1={sorted_domains[0][1]['f1']:.4f})")
    logger.info(f"  JSONL output        : {output_path}")
    logger.info(f"  Metrics output      : {metrics_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
