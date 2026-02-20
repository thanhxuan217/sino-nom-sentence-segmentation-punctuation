#!/usr/bin/env python3
"""
SikuBERT Fine-tuning with QLoRA for Token Classification
Enhanced version with streaming data and HuggingFace Trainer
"""

import argparse
import json
import logging
import multiprocessing
import os
import math
import sys
import pyarrow.parquet as pq
from pathlib import Path
from functools import partial

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType

# Import from src modules
from src.config import TaskConfig, TrainingConfig
from src.model import SikuBERTForTokenClassification
from src.data import load_streaming_dataset, preprocess_function
from src.utils import set_seed
from src.metrics import compute_metrics
from src.callbacks import LimitedEvalCallback


# ============================================================================
# MAIN TRAINING FUNCTION (using Trainer API)
# ============================================================================

def main():
    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description='Train SikuBERT with QLoRA')
    
    # Task configuration
    parser.add_argument('--task', type=str, required=True, 
                       choices=['punctuation', 'segmentation'])
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing train/val/test folders with parquet files')
    parser.add_argument('--train_path', type=str, default='',
                       help='Legacy: Path to training data (for backward compatibility)')
    parser.add_argument('--val_path', type=str, default='',
                       help='Legacy: Path to validation data')
    parser.add_argument('--test_path', type=str, default='',
                       help='Legacy: Path to test data')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='SIKU-BERT/sikubert')
    parser.add_argument('--max_length', type=int, default=256)
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=-1,
                       help='Max training steps (overrides num_epochs if set)')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--max_eval_samples', type=int, default=1000,
                       help='Max samples for intermediate evaluation')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    # QLoRA configuration
    parser.add_argument('--use_qlora', action='store_true', default=True,
                       help='Use QLoRA (default: True)')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, nargs='+',
                       default=['query', 'key', 'value'],
                       help='LoRA target modules')
    
    # DataLoader configuration
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--bf16', action='store_true', default=False)
    
    # Streaming configuration
    parser.add_argument('--use_streaming', action='store_true', default=True,
                       help='Use streaming dataset (default: True)')
    
    # DataLoader configuration (Extra)
    parser.add_argument('--pin_memory', action='store_true', default=False,
                       help='Pin memory for faster GPU transfer')
    parser.add_argument('--persistent_workers', action='store_true', default=False,
                       help='Keep workers alive between epochs')
    
    # CNN configuration
    parser.add_argument('--cnn_kernel_sizes', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('--cnn_num_filters', type=int, default=256)
    parser.add_argument('--head_type', type=str, default='cnn',
                       choices=['softmax', 'crf', 'cnn'])
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--resume_from_checkpoint', type=str, default='')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    log_file = os.path.join(args.log_dir, f"train_{args.task}_qlora.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*70)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("="*70)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n✓ Device: {device}")
    
    # Task configuration
    if args.task == "punctuation":
        task_config = TaskConfig.create(
            task_name="punctuation",
            labels=['O', '，', '。', '：', '、', '；', '？', '！'],
            ignore_labels=['O']
        )
    else:
        task_config = TaskConfig.create(
            task_name="segmentation",
            labels=['B', 'M', 'E', 'S'],
            ignore_labels=[]
        )
    
    logger.info(f"\n✓ Task: {task_config.task_name}")
    logger.info(f"  Labels: {task_config.labels}")
    logger.info(f"  Num labels: {task_config.num_labels}")
    
    # Load tokenizer
    logger.info("\n✓ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Calculate max_steps if explicitly set to -1 (use num_epochs)
    if args.max_steps == -1 and args.use_streaming:
        logger.info("\n✓ Calculating max_steps from dataset size...")
        try:
            train_dir = Path(args.data_dir) / "train"
            parquet_files = list(train_dir.glob("*.parquet"))
            total_samples = 0
            for pf in parquet_files:
                meta = pq.read_metadata(pf)
                total_samples += meta.num_rows
            
            # Adjust for distributed training
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
            steps_per_epoch = math.ceil(total_samples / effective_batch_size)
            
            args.max_steps = steps_per_epoch * args.num_epochs
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  Steps per epoch: {steps_per_epoch}")
            logger.info(f"  Max steps: {args.max_steps} (for {args.num_epochs} epochs)")
            
        except Exception as e:
            logger.warning(f"  Could not calculate dataset size: {e}")
            logger.warning("  Please provide --max_steps manually if training fails.")

    # Load datasets
    logger.info("\n✓ Loading datasets...")
    
    # Streaming mode with parquet files
    logger.info(f"  Using streaming mode from: {args.data_dir}")
    
    train_dataset = load_streaming_dataset(args.data_dir, "train")
    val_dataset = load_streaming_dataset(args.data_dir, "val")
    
    # Preprocess
    train_dataset = train_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer, task_config=task_config, max_length=args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer, task_config=task_config, max_length=args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # For evaluation during training, take limited samples
    eval_dataset = val_dataset.take(args.max_eval_samples)
        
    logger.info(f"✓ Datasets loaded")
    
    # Setup QLoRA config
    qlora_config = None
    if args.use_qlora:
        logger.info("\n✓ Setting up QLoRA configuration...")
        qlora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            # Vì bạn dùng BERT + CNN, BERT chỉ đóng vai trò feature extractor
            task_type=TaskType.FEATURE_EXTRACTION
        )
        logger.info(f"  LoRA rank: {args.lora_r}")
        logger.info(f"  LoRA alpha: {args.lora_alpha}")
        logger.info(f"  Target modules: {args.lora_target_modules}")
    
    # Create model
    logger.info(f"\n✓ Creating model with head_type={args.head_type}...")
    model = SikuBERTForTokenClassification(
        model_name=args.model_name,
        num_labels=task_config.num_labels,
        dropout=args.dropout,
        head_type=args.head_type,
        cnn_kernel_sizes=args.cnn_kernel_sizes,
        cnn_num_filters=args.cnn_num_filters,
        use_qlora=args.use_qlora,
        qlora_config=qlora_config
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=args.bf16 and torch.cuda.is_available(),
        logging_dir=args.log_dir,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.pin_memory,
        dataloader_persistent_workers=args.persistent_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["tensorboard"],
        disable_tqdm=False,  # Hiển thị progress bar trên console
        log_level="info",  # In training metrics (loss, lr) ra console real-time
        logging_first_step=True,  # In metrics ngay từ step đầu tiên
        seed=args.seed,
        # [FIX QUAN TRỌNG CHO DDP + QLoRA]
        # Tắt tính năng tự động tìm tham số không sử dụng của DDP
        # để tránh xung đột với Gradient Checkpointing.
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=10
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_length
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, task_config=task_config),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            LimitedEvalCallback(max_eval_samples=args.max_eval_samples)
        ]
    )
    
    # Train
    logger.info("\n" + "="*70)
    logger.info("TRAINING START")
    logger.info("="*70)
    
    if args.resume_from_checkpoint:
        logger.info(f"\n✓ Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    
    # Save final model
    final_model_path = os.path.join(args.model_save_dir, f"final_{args.task}_model_{args.head_type}")
    trainer.save_model(final_model_path)
    logger.info(f"\n✓ Final model saved to: {final_model_path}")
    
    # Full validation evaluation
    logger.info("\n" + "="*70)
    logger.info("FULL VALIDATION EVALUATION")
    logger.info("="*70)
    
    full_val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    logger.info(f"\nFull Validation Metrics:")
    for key, value in full_val_metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"final_metrics_{args.task}_{args.head_type}.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(full_val_metrics, f, indent=2)
    
    logger.info(f"\n✓ Metrics saved to: {metrics_path}")
    logger.info("="*70)
    logger.info("🎉 ALL DONE!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
