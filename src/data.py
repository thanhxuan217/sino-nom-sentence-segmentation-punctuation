#!/usr/bin/env python3
"""
Data loading and preprocessing utilities for SikuBERT
"""

from pathlib import Path

import torch
from datasets import load_dataset


# ============================================================================
# STREAMING DATA UTILITIES
# ============================================================================

def load_streaming_dataset(data_dir: str, split: str = "train"):
    """Load dataset from multiple parquet files in streaming mode
    
    Args:
        data_dir: Directory containing split folders with parquet files
        split: Split name (train/val/test)
    
    Returns:
        IterableDataset or Dataset
    """
    data_path = Path(data_dir) / split
    
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_path}")
    
    parquet_files = list(data_path.glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in: {data_path}")
    
    dataset = load_dataset(
        "parquet",
        data_files={split: [str(f) for f in parquet_files]},
        streaming=True
    )[split]
    
    return dataset


def preprocess_function(examples, tokenizer, task_config, max_length=256, keep_raw=False):
    """Preprocess function for token classification
    
    Compatible with both streaming and regular datasets.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: HuggingFace tokenizer
        task_config: TaskConfig instance
        max_length: Maximum sequence length
        keep_raw: If True, keep raw_text and raw_labels in output
                  (needed for run_test_set_ddp in evaluate.py)
    """
    tokenized_inputs = tokenizer(
        [list(text) for text in examples["text"]],
        is_split_into_words=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
    )
    
    labels = []
    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label = label_seq[word_id]
                label_ids.append(task_config.label2id[label])
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    
    # Keep raw text and labels for run_test_set_ddp (streaming has no random access)
    if keep_raw:
        tokenized_inputs["raw_text"] = examples["text"]
        tokenized_inputs["raw_labels"] = examples["labels"]
    
    return tokenized_inputs


def streaming_collate_fn(batch):
    """Custom collate for streaming dataset.
    
    Converts numeric fields to tensors, keeps string/list fields as Python lists.
    """
    result = {}
    for key in ['input_ids', 'attention_mask', 'labels']:
        result[key] = torch.stack([
            item[key] if isinstance(item[key], torch.Tensor) else torch.tensor(item[key])
            for item in batch
        ])
    # Keep raw text and labels as Python lists (not convertible to tensors)
    if 'raw_text' in batch[0]:
        result['raw_text'] = [item['raw_text'] for item in batch]
    if 'raw_labels' in batch[0]:
        result['raw_labels'] = [item['raw_labels'] for item in batch]
    return result
