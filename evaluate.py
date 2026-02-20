#!/usr/bin/env python3
"""
SikuBERT Evaluation Script with Multi-GPU (DDP) Support
Evaluates trained model (QLoRA checkpoint from HuggingFace Trainer) on test set
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path
import pyarrow.parquet as pq

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType

# Import from src modules
from src.config import TaskConfig
from src.model import SikuBERTForTokenClassification
from src.data import load_streaming_dataset, preprocess_function, streaming_collate_fn
from src.ddp import setup_ddp, cleanup_ddp, is_main_process
from src.checkpoint import load_model_from_trainer_checkpoint
from src.evaluation import evaluate_model, run_test_set_ddp


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Set multiprocessing start method for CUDA compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description='Evaluate SikuBERT model on test set')
    
    # Required arguments
    parser.add_argument('--task', type=str, required=True, 
                       choices=['punctuation', 'segmentation'],
                       help='Task type')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (directory or .pt file)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing split folders (test/) with parquet files')
    parser.add_argument('--test_split', type=str, default='test',
                       help='Name of the test split folder (default: test)')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='SIKU-BERT/sikubert',
                       help='Pretrained model name (must match training)')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (must match training)')
    parser.add_argument('--cnn_kernel_sizes', type=int, nargs='+', default=[3, 5, 7],
                       help='CNN kernel sizes (only used when head_type=cnn)')
    parser.add_argument('--cnn_num_filters', type=int, default=256,
                       help='Number of CNN filters (only used when head_type=cnn)')
    
    # Head type configuration
    parser.add_argument('--head_type', type=str, default='cnn',
                       choices=['softmax', 'crf', 'cnn'],
                       help='Classification head type (must match training)')
    
    # QLoRA configuration (must match training!)
    parser.add_argument('--use_qlora', action='store_true', default=False,
                       help='Use QLoRA (must match training)')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank (must match training)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha (must match training)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout (must match training)')
    parser.add_argument('--lora_target_modules', type=str, nargs='+',
                       default=['query', 'key', 'value'],
                       help='LoRA target modules (must match training)')
    
    # Inference configuration
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of DataLoader workers')
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='Use mixed precision for inference')
    parser.add_argument('--pin_memory', action='store_true', default=False,
                       help='Pin memory for DataLoader')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    
    args = parser.parse_args()
    
    # Setup DDP if available
    use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if use_ddp:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        setup_ddp(rank, world_size)
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_rank = 0
    
    # Create output directories (only main process)
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
    
    if use_ddp:
        dist.barrier()
    
    # Setup logging
    log_file = os.path.join(args.log_dir, f"{args.test_split}_evaluate_{args.task}.log")
    if is_main_process():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
            force=True
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s | [Rank %(process)d] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
    logger = logging.getLogger(__name__)
    
    # Log arguments
    if is_main_process():
        logger.info("="*70)
        logger.info("EVALUATION CONFIGURATION")
        logger.info("="*70)
        for arg, value in vars(args).items():
            logger.info(f"{arg}: {value}")
        logger.info("="*70)
        
        logger.info(f"\n✓ Device: {device}")
        if use_ddp:
            logger.info(f"  Using DDP with {int(os.environ.get('WORLD_SIZE', 1))} GPUs")
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    
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
    
    if is_main_process():
        logger.info(f"\n✓ Task: {task_config.task_name}")
        logger.info(f"  Labels: {task_config.labels}")
    
    # Load tokenizer
    if is_main_process():
        logger.info("\n✓ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # ================================================================
    # CREATE MODEL (matching train.py architecture)
    # ================================================================
    if is_main_process():
        logger.info(f"\n✓ Creating model with head_type={args.head_type}, use_qlora={args.use_qlora}...")
    
    # Setup QLoRA config if needed
    qlora_config = None
    if args.use_qlora:
        qlora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        if is_main_process():
            logger.info(f"  LoRA rank: {args.lora_r}")
            logger.info(f"  LoRA alpha: {args.lora_alpha}")
            logger.info(f"  Target modules: {args.lora_target_modules}")
    
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
    
    # ================================================================
    # LOAD CHECKPOINT WEIGHTS
    # ================================================================
    if is_main_process():
        logger.info(f"\n✓ Loading checkpoint from: {args.model_path}")
    
    model = load_model_from_trainer_checkpoint(
        checkpoint_path=args.model_path,
        model=model,
        logger=logger if is_main_process() else None,
    )
    
    # Move non-quantized parts to device (QLoRA BERT is already on device via device_map)
    if not args.use_qlora:
        model = model.to(device)
    else:
        # For QLoRA, move custom heads to device (BERT is already placed by device_map)
        if model.cnn_layer is not None:
            model.cnn_layer = model.cnn_layer.to(device)
        model.classifier = model.classifier.to(device)
        model.dropout = model.dropout.to(device)
        if model.crf is not None:
            model.crf = model.crf.to(device)
    
    model.eval()
    
    # Wrap with DDP for inference (only for non-QLoRA, since QLoRA uses device_map)
    if use_ddp and not args.use_qlora:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process():
            logger.info(f"  Model wrapped with DistributedDataParallel")
    
    if is_main_process():
        logger.info(f"  ✓ Model loaded successfully")
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Total parameters: {num_params:,}")
    
    # Load test data from parquet files (streaming mode)
    if is_main_process():
        logger.info("\n✓ Loading test data from parquet files (streaming)...")
    
    # Count total samples via pyarrow metadata (does not load data into RAM)
    test_data_path = Path(args.data_dir) / args.test_split
    parquet_files = list(test_data_path.glob("*.parquet"))
    total_test_samples = sum(pq.read_metadata(str(f)).num_rows for f in parquet_files)
    if is_main_process():
        logger.info(f"  Test samples: {total_test_samples}")
    
    raw_test_dataset = load_streaming_dataset(args.data_dir, args.test_split)
    
    # Preprocess (tokenize + align labels), keep raw_text/raw_labels for predictions
    tokenized_test_dataset = raw_test_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer, task_config=task_config, max_length=args.max_length, keep_raw=True),
        batched=True,
        remove_columns=["text", "labels"]
    )
    
    # Streaming IterableDataset: no DistributedSampler, no set_format
    # Use custom collate_fn to handle mixed tensor+string fields
    test_loader = DataLoader(
        tokenized_test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=streaming_collate_fn
    )
    
    # Evaluate
    if is_main_process():
        logger.info("\n" + "="*70)
        logger.info("TEST SET EVALUATION")
        logger.info("="*70)
    
    test_metrics = evaluate_model(
        model,
        test_loader,
        task_config,
        device,
        use_amp=args.fp16
    )
    
    if is_main_process():
        logger.info(f"\nTest Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Precision (Overall): {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall (Overall): {test_metrics['recall']:.4f}")
        logger.info(f"Test F1 (Overall): {test_metrics['f1']:.4f}")
        
        # Log per-label metrics
        logger.info("\n  Per-Label Metrics:")
        logger.info("  {:<12} {:>10} {:>10} {:>10} {:>10}".format(
            "Label", "Precision", "Recall", "F1", "Support"))
        logger.info("  " + "-" * 52)
        for label_name, metrics in test_metrics['per_label'].items():
            logger.info("  {:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10}".format(
                label_name,
                metrics['precision'],
                metrics['recall'],
                metrics['f1'],
                metrics['support']
            ))
    
    if args.test_split != "val":
        # Run predictions on test set
        if is_main_process():
            logger.info("\n" + "="*70)
            logger.info("RUNNING PREDICTIONS ON TEST SET")
            logger.info("="*70)
        
        predictions_path = os.path.join(args.output_dir, f"{args.test_split}_{args.task}_predictions.json")
        
        # Re-create streaming dataset + DataLoader (IterableDataset can only be iterated once)
        pred_raw_dataset = load_streaming_dataset(args.data_dir, args.test_split)
        pred_tokenized = pred_raw_dataset.map(
            partial(preprocess_function, tokenizer=tokenizer, task_config=task_config, max_length=args.max_length, keep_raw=True),
            batched=True,
            remove_columns=["text", "labels"]
        )
        pred_loader = DataLoader(
            pred_tokenized,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=streaming_collate_fn
        )
        
        run_test_set_ddp(
            model=model,
            tokenizer=tokenizer,
            config=task_config,
            device=device,
            dataloader=pred_loader,
            output_path=predictions_path,
            max_length=args.max_length,
            logger=logger if is_main_process() else None
        )
    
    # Save results (only main process)
    if is_main_process():
        results = {
            'task': args.task,
            'model_path': args.model_path,
            'data_dir': args.data_dir,
            'use_qlora': args.use_qlora,
            'head_type': args.head_type,
            'test_metrics': test_metrics,
            'config': vars(args)
        }
        
        results_path = os.path.join(args.output_dir, f"{args.test_split}_{args.task}_eval_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Results saved to: {results_path}")
        logger.info("="*70)
        logger.info("🎉 EVALUATION COMPLETE!")
        logger.info("="*70)
    
    # Cleanup DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()
