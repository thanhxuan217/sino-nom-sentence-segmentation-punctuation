#!/usr/bin/env python3
"""
Domain Adaptive Pretraining (DAPT) for SikuBERT

Continued pretraining of SikuBERT on a historical Chinese corpus using
Masked Language Modeling (MLM) with punctuation-aware masking.

Supports:
    - Multi-GPU training via torchrun (DDP)
    - Streaming from sharded Parquet files
    - Two-phase training (short → long sequences)
    - Checkpoint resuming
    - Mixed precision (fp16/bf16)

Usage:
    torchrun --nproc_per_node=2 dapt_train.py \
        --model_name SIKU-BERT/sikubert \
        --tokenizer_path sikubert_extended_vocab \
        --data_dir /path/to/dapt_data \
        --output_dir /path/to/dapt_outputs \
        --max_length 512 \
        --per_device_batch_size 32 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --fp16
"""

import argparse
import logging
import os
import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.dapt_collator import PunctuationAwareMLMCollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Domain Adaptive Pretraining (DAPT) for SikuBERT"
    )

    # ---- Model & tokenizer ----
    parser.add_argument("--model_name", default="SIKU-BERT/sikubert",
                        help="Pretrained model name or path")
    parser.add_argument("--tokenizer_path", default="sikubert_extended_vocab",
                        help="Path to tokenizer directory")

    # ---- Data ----
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing train/ with parquet shards")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Sequence length (should match data preparation)")
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Use streaming mode for large datasets")

    # ---- Training hyperparameters ----
    parser.add_argument("--per_device_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use epochs)")
    parser.add_argument("--seed", type=int, default=42)

    # ---- MLM masking ----
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Overall MLM masking probability")
    parser.add_argument("--punct_mask_prob", type=float, default=0.30,
                        help="Masking probability for punctuation tokens")

    # ---- Mixed precision ----
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 mixed precision")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 mixed precision (requires Ampere+)")

    # ---- Saving & logging ----
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_total_limit", type=int, default=5,
                        help="Max checkpoints to keep")
    parser.add_argument("--logging_steps", type=int, default=100)

    # ---- Resuming ----
    parser.add_argument("--resume_from", default=None,
                        help="Path to checkpoint directory to resume from")

    # ---- Workers ----
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # ================================================================
    # 1. TOKENIZER
    # ================================================================
    logger.info(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    logger.info(f"  Vocab size: {len(tokenizer):,}")

    # ================================================================
    # 2. MODEL (full precision, no QLoRA for DAPT)
    # ================================================================
    logger.info(f"Loading model from: {args.model_name}")
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name,
        use_safetensors=True,
    )

    # Resize embeddings if vocab was extended
    if len(tokenizer) != model.config.vocab_size:
        logger.info(
            f"  Resizing embeddings: {model.config.vocab_size} → {len(tokenizer)}"
        )
        model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters:     {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # ================================================================
    # 3. DATASET
    # ================================================================
    train_data_path = os.path.join(args.data_dir, "train", "*.parquet")
    logger.info(f"Loading dataset from: {train_data_path}")

    import glob
    parquet_files = sorted(glob.glob(train_data_path))
    if not parquet_files:
        logger.error(f"No parquet files found at {train_data_path}")
        sys.exit(1)
    logger.info(f"  Found {len(parquet_files)} parquet shards")

    if args.streaming:
        dataset = load_dataset(
            "parquet",
            data_files={"train": parquet_files},
            streaming=True,
        )["train"]
        dataset = dataset.shuffle(seed=args.seed, buffer_size=10_000)
    else:
        dataset = load_dataset(
            "parquet",
            data_files={"train": parquet_files},
        )["train"]
        dataset = dataset.shuffle(seed=args.seed)

    # ================================================================
    # 4. DATA COLLATOR (Punctuation-Aware MLM)
    # ================================================================
    data_collator = PunctuationAwareMLMCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        punct_mask_prob=args.punct_mask_prob,
    )
    logger.info(
        f"  MLM probability: {args.mlm_probability}, "
        f"Punct mask boost: {args.punct_mask_prob}"
    )

    # ================================================================
    # 5. TRAINING ARGUMENTS
    # ================================================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Batch & gradient
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Optimizer
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,

        # Duration
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        # Precision
        fp16=args.fp16,
        bf16=args.bf16,

        # Saving
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,

        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="tensorboard",

        # Other
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # Log effective batch size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch = (
        args.per_device_batch_size
        * world_size
        * args.gradient_accumulation_steps
    )
    logger.info(f"  Effective batch size: {effective_batch}")
    logger.info(f"  World size: {world_size}")

    # ================================================================
    # 6. TRAINER
    # ================================================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ================================================================
    # 7. TRAIN
    # ================================================================
    logger.info("=" * 60)
    logger.info("STARTING DOMAIN ADAPTIVE PRETRAINING")
    logger.info("=" * 60)

    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # ================================================================
    # 8. SAVE FINAL MODEL
    # ================================================================
    final_dir = os.path.join(args.output_dir, "final_dapt_model")
    logger.info(f"Saving final DAPT model to: {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info("=" * 60)
    logger.info("DAPT COMPLETE")
    logger.info(f"  Final model saved to: {final_dir}")
    logger.info(f"  To finetune downstream, use: --model_name {final_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
