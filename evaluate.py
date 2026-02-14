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
from dataclasses import dataclass
from typing import List, Dict, Optional
from functools import partial
from pathlib import Path
import pyarrow.parquet as pq

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from datasets import load_dataset as hf_load_dataset
from transformers import AutoModel, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from torchcrf import CRF
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ============================================================================
# CONSTANTS
# ============================================================================

TASK = "test"


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class TaskConfig:
    """Configuration for each task"""
    task_name: str
    labels: List[str]
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    num_labels: int
    ignore_labels: List[str] = None
    
    @classmethod
    def create(cls, task_name: str, labels: List[str], ignore_labels: List[str] = None):
        """Factory method to create config"""
        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        return cls(
            task_name=task_name,
            labels=labels,
            label2id=label2id,
            id2label=id2label,
            num_labels=len(labels),
            ignore_labels=ignore_labels or []
        )


# ============================================================================
# DDP UTILITIES
# ============================================================================

def setup_ddp(rank: int, world_size: int):
    """Initialize distributed process group"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29502')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process (rank 0)"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# ============================================================================
# STREAMING DATA UTILITIES (Matches train.py)
# ============================================================================

def load_streaming_dataset(data_dir: str, split: str = TASK):
    """Load dataset from multiple parquet files
    
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
    
    dataset = hf_load_dataset(
        "parquet",
        data_files={TASK: [str(f) for f in parquet_files]},
        streaming=True
    )[TASK]
    
    return dataset


def preprocess_function(examples, tokenizer, task_config, max_length=256):
    """Preprocess function for token classification
    
    Compatible with both streaming and regular datasets.
    Keeps raw_text and raw_labels for prediction output in run_test_set_ddp.
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


# ============================================================================
# MODEL ARCHITECTURE (Matches train.py exactly)
# ============================================================================

class MultiKernelCNN(nn.Module):
    """Multi-kernel CNN layer"""
    
    def __init__(self, hidden_size: int, kernel_sizes: list, num_filters: int):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.output_size = num_filters * len(kernel_sizes)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]
        conv_outputs = [out.transpose(1, 2) for out in conv_outputs]
        return torch.cat(conv_outputs, dim=-1)


class SikuBERTForTokenClassification(nn.Module):
    """SikuBERT with configurable classification head
    
    Supports 3 head types:
    - softmax: BERT → Dropout → Linear → CrossEntropyLoss (Softmax)
    - crf: BERT → Dropout → Linear → CRF
    - cnn: BERT → Dropout → MultiKernelCNN → Dropout → Linear
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        head_type: str = 'softmax',
        cnn_kernel_sizes: list = None,
        cnn_num_filters: int = 128,
        use_qlora: bool = False,
        qlora_config: Optional[LoraConfig] = None
    ):
        super().__init__()
        
        self.head_type = head_type
        self.num_labels = num_labels
        self.use_qlora = use_qlora
        
        # Backbone: SikuBERT
        if use_qlora and qlora_config is not None:
            # Load model in 4-bit for QLoRA
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.bert = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={'': int(os.environ.get('LOCAL_RANK', 0))} if int(os.environ.get("WORLD_SIZE", 1)) > 1 else "auto",
                use_safetensors=True,
                add_pooling_layer=False
            )
            
            # Prepare for k-bit training
            self.bert = prepare_model_for_kbit_training(self.bert)
            
            # Apply LoRA
            self.bert = get_peft_model(self.bert, qlora_config)
            
            if hasattr(self.bert, 'print_trainable_parameters'):
                self.bert.print_trainable_parameters()
        else:
            self.bert = AutoModel.from_pretrained(
                model_name,
                use_safetensors=True,
                add_pooling_layer=False
            )

        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # CNN layer (only for 'cnn' head type)
        self.cnn_layer = None
        if head_type == 'cnn':
            if cnn_kernel_sizes is None:
                cnn_kernel_sizes = [3, 5, 7]
            
            self.cnn_layer = MultiKernelCNN(
                hidden_size=self.hidden_size,
                kernel_sizes=cnn_kernel_sizes,
                num_filters=cnn_num_filters
            )
            classifier_input_size = self.cnn_layer.output_size
        else:
            classifier_input_size = self.hidden_size
        
        # Classification head
        self.classifier = nn.Linear(classifier_input_size, num_labels)
        
        # CRF layer (only for 'crf' head type)
        self.crf = None
        if head_type == 'crf':
            self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None, **kwargs):
        # Chỉ lấy những gì BertModel thực sự cần
        bert_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Một số model không dùng token_type_ids, chỉ thêm nếu có
        if token_type_ids is not None:
            bert_inputs["token_type_ids"] = token_type_ids

        bert_outputs = self.bert(**bert_inputs)
        sequence_output = bert_outputs.last_hidden_state
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # CNN layer (if using CNN head)
        if self.cnn_layer is not None:
            sequence_output = self.cnn_layer(sequence_output)
            sequence_output = self.dropout(sequence_output)
        
        # Get emission scores
        emissions = self.classifier(sequence_output)
        
        # Calculate loss and get predictions based on head type
        result = {'logits': emissions}
        
        if self.head_type == 'crf':
            # CRF head
            if labels is not None:
                crf_mask = attention_mask.bool()
                crf_labels = labels.clone()
                crf_labels[labels == -100] = 0
                
                loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction='mean')
                result['loss'] = loss
            
            # Viterbi decoding
            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=crf_mask)
            
            max_len = emissions.size(1)
            batch_size = emissions.size(0)
            pred_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long, device=emissions.device)
            
            for i, pred_seq in enumerate(predictions):
                pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=emissions.device)
            
            result['predictions'] = pred_tensor

        else:
            # Softmax or CNN head
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
                result['loss'] = loss
            
            result['predictions'] = torch.argmax(emissions, dim=-1)
            
        return result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def apply_punctuation_labels(text: str, labels: List[str]) -> str:
    """Apply punctuation labels to text"""
    output = []
    for ch, label in zip(text, labels):
        output.append(ch)
        if label != "O":
            output.append(label)
    return "".join(output)


def apply_segmentation_inline(text: str, labels: List[str], sep: str = " | ") -> str:
    """Apply segmentation labels to text"""
    output = []
    for ch, label in zip(text, labels):
        output.append(ch)
        if label in ("E", "S"):
            output.append(sep)
    return "".join(output).rstrip(sep)


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_model_from_trainer_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    logger=None,
):
    """Load model weights from a HuggingFace Trainer checkpoint directory.
    
    Trainer saves:
      - pytorch_model.bin or model.safetensors (full state dict)
      - OR adapter weights + custom head weights
    
    This function handles both cases.
    """
    ckpt_dir = Path(checkpoint_path)
    
    if not ckpt_dir.is_dir():
        # Fallback: single .pt file (legacy format)
        if logger:
            logger.info(f"  Loading from single file: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return model
    
    # --- Trainer checkpoint directory ---
    if logger:
        logger.info(f"  Loading from Trainer checkpoint directory: {checkpoint_path}")
        logger.info(f"  Contents: {[f.name for f in ckpt_dir.iterdir()]}")
    
    # Strategy 1: pytorch_model.bin (full state dict)
    full_state_path = ckpt_dir / "pytorch_model.bin"
    if full_state_path.exists():
        if logger:
            logger.info(f"  Found pytorch_model.bin, loading full state dict...")
        state_dict = torch.load(str(full_state_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return model
    
    # Strategy 2: model.safetensors (full state dict in safetensors format)
    safetensors_path = ckpt_dir / "model.safetensors"
    if safetensors_path.exists():
        if logger:
            logger.info(f"  Found model.safetensors, loading full state dict...")
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
        model.load_state_dict(state_dict, strict=False)
        return model
    
    # Strategy 3: QLoRA checkpoint (adapter weights + custom head weights)
    # Trainer with custom nn.Module saves the entire model's state via
    # trainer._save() which calls torch.save(self.model.state_dict(), ...)
    # But with PEFT/QLoRA, we may get separate adapter and head files.
    
    # 3a: Load LoRA adapter weights into BERT
    adapter_config_path = ckpt_dir / "adapter_config.json"
    adapter_model_path = ckpt_dir / "adapter_model.safetensors"
    adapter_model_bin_path = ckpt_dir / "adapter_model.bin"
    
    if adapter_config_path.exists():
        if logger:
            logger.info(f"  Found adapter_config.json, loading LoRA adapters...")
        
        # Load adapter weights
        if adapter_model_path.exists():
            from safetensors.torch import load_file
            adapter_state = load_file(str(adapter_model_path))
        elif adapter_model_bin_path.exists():
            adapter_state = torch.load(str(adapter_model_bin_path), map_location='cpu', weights_only=True)
        else:
            raise FileNotFoundError(f"No adapter model found in {checkpoint_path}")
        
        # Load adapter into model.bert (which is a PeftModel)
        bert_state = {}
        for k, v in adapter_state.items():
            bert_state[k] = v
        
        model.bert.load_state_dict(bert_state, strict=False)
        if logger:
            logger.info(f"  ✓ Loaded {len(adapter_state)} adapter weight tensors")
    
    # 3b: Load custom head weights (CNN, classifier, CRF, dropout)
    # Trainer with PEFT may save non-adapter weights separately
    custom_head_files = [
        "custom_head.bin",        # Custom saving convention
        "non_lora_trainables.bin", # Another common convention  
    ]
    
    head_loaded = False
    for fname in custom_head_files:
        fpath = ckpt_dir / fname
        if fpath.exists():
            if logger:
                logger.info(f"  Found {fname}, loading custom head weights...")
            head_state = torch.load(str(fpath), map_location='cpu', weights_only=True)
            model.load_state_dict(head_state, strict=False)
            head_loaded = True
            break
    
    if not head_loaded:
        # Try loading any remaining .bin files that might contain head weights
        bin_files = list(ckpt_dir.glob("*.bin"))
        for bf in bin_files:
            if bf.name in ("adapter_model.bin", "optimizer.bin", "scheduler.bin", 
                           "trainer_state.bin", "training_args.bin", "rng_state.pth"):
                continue
            if logger:
                logger.info(f"  Trying to load head weights from: {bf.name}")
            try:
                head_state = torch.load(str(bf), map_location='cpu', weights_only=True)
                model.load_state_dict(head_state, strict=False)
                head_loaded = True
                if logger:
                    logger.info(f"  ✓ Loaded head weights from {bf.name}")
                break
            except Exception as e:
                if logger:
                    logger.warning(f"  Could not load {bf.name}: {e}")
    
    if not head_loaded and not adapter_config_path.exists():
        raise FileNotFoundError(
            f"Could not find any loadable weights in {checkpoint_path}. "
            f"Expected pytorch_model.bin, model.safetensors, or adapter files."
        )
    
    return model


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
    log_file = os.path.join(args.log_dir, f"{TASK}_evaluate_{args.task}.log")
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
        partial(preprocess_function, tokenizer=tokenizer, task_config=task_config, max_length=args.max_length),
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
    
    # Run predictions on test set
    if is_main_process():
        logger.info("\n" + "="*70)
        logger.info("RUNNING PREDICTIONS ON TEST SET")
        logger.info("="*70)
    
    predictions_path = os.path.join(args.output_dir, f"{TASK}_{args.task}_predictions.json")
    
    # Re-create streaming dataset + DataLoader (IterableDataset can only be iterated once)
    pred_raw_dataset = load_streaming_dataset(args.data_dir, args.test_split)
    pred_tokenized = pred_raw_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer, task_config=task_config, max_length=args.max_length),
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
        
        results_path = os.path.join(args.output_dir, f"{TASK}_{args.task}_eval_results.json")
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
