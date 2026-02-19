#!/usr/bin/env python3
"""
Evaluation and prediction functions for SikuBERT
"""

import json
import os
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.config import TaskConfig
from src.ddp import is_main_process
from src.utils import apply_punctuation_labels, apply_segmentation_inline


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_labels(
    model,
    text: str,
    tokenizer,
    config: TaskConfig,
    device: str,
    max_length: int = 256
) -> List[str]:
    """Predict labels for a single text"""
    model.eval()
    chars = list(text)
    
    tokenized = tokenizer(
        chars,
        is_split_into_words=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    )
    
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs["logits"], dim=-1)[0]
    
    word_ids = tokenized.word_ids()
    pred_labels = []
    
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            label_id = preds[idx].item()
            pred_labels.append(config.id2label[label_id])
    
    return pred_labels


def run_test_set(
    model,
    tokenizer,
    config: TaskConfig,
    device: str,
    test_texts: List[str],
    test_labels: List[List[str]],
    output_path: str,
    max_length: int = 256,
    logger=None
):
    """Run predictions on test set and save results (single-GPU, no DDP)"""
    results = []
    
    for i, (text, gold_labels) in enumerate(tqdm(zip(test_texts, test_labels), 
                                                   total=len(test_texts),
                                                   desc="Running predictions")):
        pred_labels = predict_labels(
            model=model,
            text=text,
            tokenizer=tokenizer,
            config=config,
            device=device,
            max_length=max_length
        )
        
        if config.task_name == "punctuation":
            gold_text = apply_punctuation_labels(text, gold_labels)
            pred_text = apply_punctuation_labels(text, pred_labels)
        elif config.task_name == "segmentation":
            gold_text = apply_segmentation_inline(text, gold_labels)
            pred_text = apply_segmentation_inline(text, pred_labels)
        else:
            gold_text = text
            pred_text = text
        
        results.append({
            "text": text,
            "gold_labels": gold_labels,
            "pred_labels": pred_labels,
            "gold_text_labeled": gold_text,
            "pred_text_labeled": pred_text,
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if logger:
        logger.info(f"✓ Predictions saved to: {output_path}")
        logger.info(f"  Total samples: {len(results)}")
        
        logger.info("\n" + "="*70)
        logger.info("SAMPLE PREDICTIONS")
        logger.info("="*70)
        for idx in range(min(3, len(results))):
            sample = results[idx]
            logger.info(f"\n--- Sample {idx + 1} ---")
            logger.info(f"Original:  {sample['text'][:100]}...")
            logger.info(f"Gold:      {sample['gold_text_labeled'][:100]}...")
            logger.info(f"Predicted: {sample['pred_text_labeled'][:100]}...")


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, dataloader, task_config, device, use_amp: bool = False):
    """Evaluate model on a dataset with minimal memory footprint"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Initialize confusion matrix
    num_labels = task_config.num_labels
    conf_matrix = np.zeros((num_labels, num_labels), dtype=np.int64)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main_process()):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp and torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs['loss']
            if loss is None:
                raise ValueError("Model did not return a loss. Ensure labels are passed correctly.")
            if loss.dim() > 0:
                loss = loss.mean()

            total_loss += loss.item()
            num_batches += 1
            
            if 'predictions' in outputs:
                predictions = outputs['predictions']
            else:
                predictions = torch.argmax(outputs['logits'], dim=-1)
            
            # Move to CPU and process immediately
            predictions = predictions.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            # Delete GPU tensors
            del outputs, input_ids, attention_mask, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Update confusion matrix (only valid labels)
            mask = labels_cpu != -100
            valid_preds = predictions[mask]
            valid_labels = labels_cpu[mask]
            
            # Filter out ignore labels
            if task_config.ignore_labels:
                ignore_ids = [task_config.label2id[label] for label in task_config.ignore_labels]
                keep_mask = ~np.isin(valid_labels, ignore_ids)
                valid_preds = valid_preds[keep_mask]
                valid_labels = valid_labels[keep_mask]
            
            # Update confusion matrix
            conf_matrix += confusion_matrix(
                valid_labels, valid_preds,
                labels=list(range(num_labels))
            )
            
            # Delete CPU arrays
            del predictions, labels_cpu, mask, valid_preds, valid_labels
    
    # Calculate metrics from confusion matrix
    avg_loss = total_loss / max(num_batches, 1)
    
    # Calculate per-class precision, recall, F1
    per_label_metrics = {}
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(num_labels):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        support = conf_matrix[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        label_name = task_config.id2label.get(i, str(i))
        per_label_metrics[label_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(support)
        }
        
        if support > 0:  # Only include labels with support for macro average
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
    
    # Macro average
    macro_precision = np.mean(precisions) if precisions else 0.0
    macro_recall = np.mean(recalls) if recalls else 0.0
    macro_f1 = np.mean(f1s) if f1s else 0.0
    
    return {
        'loss': avg_loss,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'per_label': per_label_metrics
    }


def run_test_set_ddp(
    model,
    tokenizer,
    config: TaskConfig,
    device: str,
    dataloader,
    output_path: str,
    max_length: int = 256,
    max_samples: int = 100,
    logger=None
):
    """Run predictions on test set with DDP support.
    
    Streams results to per-rank JSONL temp files to avoid OOM on large datasets.
    Rank 0 merges all temp files into the final output after inference.
    Raw text and labels are obtained from the batch (raw_text, raw_labels keys)
    instead of random-accessing a dataset, to support streaming.
    
    Args:
        max_samples: Maximum number of samples to process. Set to None to run all.
                     Default is 100 for quick output inspection.
    """
    import tempfile
    
    model.eval()
    
    # Each rank writes to its own temp file to avoid RAM buildup
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    output_dir = os.path.dirname(output_path) or "."
    tmp_path = os.path.join(output_dir, f".predictions_rank{rank}.jsonl")
    
    sample_results = []  # Keep only first 3 samples for logging
    total_written = 0
    global_sample_idx = 0  # Global counter for sample indexing
    
    if max_samples is not None and is_main_process() and logger:
        logger.info(f"  ⚠ Limiting predictions to {max_samples} samples")
    
    with open(tmp_path, "w", encoding="utf-8") as tmp_f:
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Running predictions", disable=not is_main_process())
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = input_ids.size(0)
                raw_texts = batch['raw_text']
                raw_labels_list = batch['raw_labels']
                    
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                if 'predictions' in outputs:
                    predictions = outputs['predictions']
                else:
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                
                # Process each sample in batch and write immediately
                for i in range(batch_size):
                    text = raw_texts[i]
                    gold_labels = raw_labels_list[i]
                    idx = global_sample_idx + i
                    
                    # Get word_ids for this sample
                    tokenized = tokenizer(
                        list(text),
                        is_split_into_words=True,
                        return_tensors="pt",
                        max_length=max_length,
                        truncation=True
                    )
                    word_ids = tokenized.word_ids()
                    
                    # Extract predictions for valid positions
                    pred_labels = []
                    pred = predictions[i].cpu()
                    for token_idx, word_id in enumerate(word_ids):
                        if word_id is not None and token_idx < len(pred):
                            label_id = pred[token_idx].item()
                            pred_labels.append(config.id2label[label_id])
                    
                    # Apply labels to get formatted text
                    if config.task_name == "punctuation":
                        gold_text = apply_punctuation_labels(text, gold_labels)
                        pred_text = apply_punctuation_labels(text, pred_labels)
                    elif config.task_name == "segmentation":
                        gold_text = apply_segmentation_inline(text, gold_labels)
                        pred_text = apply_segmentation_inline(text, pred_labels)
                    else:
                        gold_text = text
                        pred_text = text
                    
                    result = {
                        "idx": idx,
                        "text": text,
                        "gold_labels": gold_labels,
                        "pred_labels": pred_labels,
                        "gold_text_labeled": gold_text,
                        "pred_text_labeled": pred_text,
                    }
                    
                    # Stream to disk immediately
                    tmp_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    total_written += 1
                    
                    # Keep first 3 for sample logging (tiny overhead)
                    if len(sample_results) < 3:
                        sample_results.append(result)
                    
                    # Early exit if max_samples reached
                    if max_samples is not None and total_written >= max_samples:
                        break
                
                global_sample_idx += batch_size
                
                # Free GPU memory
                del outputs, input_ids, attention_mask, predictions
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Break outer batch loop too
                if max_samples is not None and total_written >= max_samples:
                    break
    
    if logger:
        logger.info(f"  Rank {rank}: wrote {total_written} predictions to temp file")
    
    # Synchronize all ranks before merging
    if dist.is_initialized():
        dist.barrier()
    
    # Only rank 0 merges all temp files into final output
    if is_main_process():
        if logger:
            logger.info(f"  Merging predictions from {world_size} rank(s)...")
        
        # Read all temp files, sort by idx, write final JSONL output
        with open(output_path, "w", encoding="utf-8") as out_f:
            all_lines = []
            for r in range(world_size):
                rpath = os.path.join(output_dir, f".predictions_rank{r}.jsonl")
                if os.path.exists(rpath):
                    with open(rpath, "r", encoding="utf-8") as rf:
                        for line in rf:
                            if line.strip():
                                all_lines.append(json.loads(line))
            
            # Sort by original index
            all_lines.sort(key=lambda x: x['idx'])
            
            # Write sorted results (without idx field)
            total_count = len(all_lines)
            for item in all_lines:
                del item['idx']
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            # Free the list
            del all_lines
        
        if logger:
            logger.info(f"✓ Predictions saved to: {output_path}")
            logger.info(f"  Total samples: {total_count}")
            
            logger.info("\n" + "="*70)
            logger.info("SAMPLE PREDICTIONS")
            logger.info("="*70)
            for idx, sample in enumerate(sample_results):
                logger.info(f"\n--- Sample {idx + 1} ---")
                logger.info(f"Original:  {sample['text'][:100]}...")
                logger.info(f"Gold:      {sample['gold_text_labeled'][:100]}...")
                logger.info(f"Predicted: {sample['pred_text_labeled'][:100]}...")
        
        # Cleanup temp files
        for r in range(world_size):
            rpath = os.path.join(output_dir, f".predictions_rank{r}.jsonl")
            if os.path.exists(rpath):
                os.remove(rpath)
    
    # Final barrier so non-main ranks wait for cleanup
    if dist.is_initialized():
        dist.barrier()
