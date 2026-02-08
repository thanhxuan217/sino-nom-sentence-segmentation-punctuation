#!/usr/bin/env python3
"""
SikuBERT Fine-tuning with CNN for Token Classification
Converted from Jupyter notebook for SLURM execution
"""

import argparse
import json
import logging
import multiprocessing
import os
import random
import math
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


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


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    model_name: str = "SIKU-BERT/sikubert"
    max_length: int = 256

    # Hyperparameters to be tuned
    batch_size: int = 64
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


# ============================================================================
# DATASET CLASS
# ============================================================================

class ClassicalChineseDataset(Dataset):
    """Dataset for token classification tasks"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[List[str]],
        tokenizer,
        config: TaskConfig,
        max_length: int = 256
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        
        # Validate data
        assert len(texts) == len(labels), "texts and labels must have same length"
        for text, label_seq in zip(texts, labels):
            assert len(text) == len(label_seq), \
                f"Text and labels mismatch: {len(text)} vs {len(label_seq)}"
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_seq = self.labels[idx]
        
        # Tokenize with word tracking
        tokenized = self.tokenizer(
            list(text),
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels with subword tokens
        word_ids = tokenized.word_ids(batch_index=0)
        label_ids = []
        
        for word_id in word_ids:
            if word_id is None:
                # [CLS], [SEP], [PAD] -> -100 (ignored by loss)
                label_ids.append(-100)
            else:
                label = label_seq[word_id]
                label_ids.append(self.config.label2id[label])
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


# ============================================================================
# MODEL ARCHITECTURE
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
        # x: [batch, seq_len, hidden_size]
        x = x.transpose(1, 2)  # [batch, hidden_size, seq_len]
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
        cnn_num_filters: int = 128
    ):
        super().__init__()
        
        self.head_type = head_type
        self.num_labels = num_labels
        
        # Backbone: SikuBERT
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
    
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # CNN layer (if using CNN head)
        if self.cnn_layer is not None:
            sequence_output = self.cnn_layer(sequence_output)
            sequence_output = self.dropout(sequence_output)
        
        # Get emission scores
        emissions = self.classifier(sequence_output)
        
        # Calculate loss and get predictions based on head type
        loss = None
        
        if self.head_type == 'crf':
            # CRF head
            # Create mask: True for valid tokens, False for padding and special tokens
            # labels == -100 are ignored positions (padding, [CLS], [SEP])
            if labels is not None:
                # Create CRF mask (True for valid, False for ignored)
                crf_mask = (labels != -100)
                # Replace -100 with 0 for CRF (it will be masked anyway)
                crf_labels = labels.clone()
                crf_labels[~crf_mask] = 0
                # CRF forward returns negative log-likelihood
                loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction='mean')
            
            return {'loss': loss, 'logits': emissions}
        else:
            # Softmax or CNN head (both use CrossEntropyLoss)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
            
            return {'loss': loss, 'logits': emissions}
    
    def decode(self, input_ids, attention_mask):
        """Decode predictions (especially useful for CRF)
        
        Returns:
            List of predicted label sequences
        """
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Dropout (in eval mode, dropout is identity)
        sequence_output = self.dropout(sequence_output)
        
        # CNN layer
        if self.cnn_layer is not None:
            sequence_output = self.cnn_layer(sequence_output)
            sequence_output = self.dropout(sequence_output)
        
        # Get emission scores
        emissions = self.classifier(sequence_output)
        
        if self.head_type == 'crf':
            # Use Viterbi decoding for CRF
            # Use attention_mask as the CRF mask
            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=crf_mask)
            return predictions
        else:
            # Argmax for softmax/cnn heads
            predictions = torch.argmax(emissions, dim=-1)
            return predictions.tolist()


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_ddp(rank: int, world_size: int):
    """Initialize distributed process group"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
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


def load_data(json_path: str) -> Tuple[List[str], List[List[str]]]:
    texts = []
    labels = []
    """Load data from JSON file"""
    with open(json_path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        # Case 1: JSON array
        if first == "[":
            data = json.load(f)
            for item in data:
                texts.append(item["text"])
                labels.append(item["labels"])

        # Case 2: JSONL
        else:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(item["labels"])

    return texts, labels


def create_dataloaders(
    train_texts: List[str],
    train_labels: List[List[str]],
    val_texts: List[str],
    val_labels: List[List[str]],
    tokenizer,
    config: TaskConfig,
    training_config: TrainingConfig,
    use_ddp: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    train_dataset = ClassicalChineseDataset(
        train_texts, train_labels, tokenizer, config, training_config.max_length
    )
    val_dataset = ClassicalChineseDataset(
        val_texts, val_labels, tokenizer, config, training_config.max_length
    )
    
    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(train_dataset) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
    
    # persistent_workers only works with num_workers > 0
    use_persistent = training_config.persistent_workers and training_config.num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=use_persistent
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=use_persistent
    )
    
    return train_loader, val_loader, train_sampler


def apply_punctuation_labels(text: str, labels: List[str]) -> str:
    """Apply punctuation labels to text
    
    Args:
        text: Original text string
        labels: List of labels ('O' or punctuation marks)
    
    Returns:
        Text with punctuation inserted after characters
    """
    output = []
    for ch, label in zip(text, labels):
        output.append(ch)
        if label != "O":
            output.append(label)
    return "".join(output)


def apply_segmentation_inline(text: str, labels: List[str], sep: str = " | ") -> str:
    """Apply segmentation labels to text
    
    Args:
        text: Original text string
        labels: List of BMES labels
        sep: Separator to use between segments
    
    Returns:
        Text with separators between segments
    """
    output = []
    for ch, label in zip(text, labels):
        output.append(ch)
        if label in ("E", "S"):
            output.append(sep)
    return "".join(output).rstrip(sep)


def predict_labels(
    model,
    text: str,
    tokenizer,
    config: TaskConfig,
    device: str,
    max_length: int = 256
) -> List[str]:
    """Predict labels for a single text
    
    Args:
        model: The trained model
        text: Input text string
        tokenizer: The tokenizer
        config: Task configuration
        device: Device to run on
        max_length: Maximum sequence length
    
    Returns:
        List of predicted labels for each character
    """
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
    """Run predictions on test set and save results with actual text
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        config: Task configuration
        device: Device to run on
        test_texts: List of test texts
        test_labels: List of ground truth labels
        output_path: Path to save results JSON
        max_length: Maximum sequence length
        logger: Logger instance
    """
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
        
        results.append({
            "text": text,
            "gold_labels": gold_labels,
            "pred_labels": pred_labels,
            "gold_text_labeled": gold_text,
            "pred_text_labeled": pred_text,
        })
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if logger:
        logger.info(f"✓ Predictions saved to: {output_path}")
        logger.info(f"  Total samples: {len(results)}")
        
        # Show a few examples
        logger.info("\n" + "="*70)
        logger.info("SAMPLE PREDICTIONS")
        logger.info("="*70)
        for idx in range(min(3, len(results))):
            sample = results[idx]
            logger.info(f"\n--- Sample {idx + 1} ---")
            logger.info(f"Original:  {sample['text'][:100]}...")
            logger.info(f"Gold:      {sample['gold_text_labeled'][:100]}...")
            logger.info(f"Predicted: {sample['pred_text_labeled'][:100]}...")


def evaluate_model(model, dataloader, task_config, device, use_amp: bool = False):
    """Evaluate model on a dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp and torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs['loss']
            if loss is None:
                raise ValueError("Model did not return a loss. Ensure labels are passed correctly.")
            # DataParallel gathers per-device scalar losses into a vector; reduce to scalar.
            if loss.dim() > 0:
                loss = loss.mean()

            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=-1)
            
            # Collect predictions and labels (excluding -100)
            mask = labels != -100
            
            for pred, label, m in zip(predictions, labels, mask):
                valid_preds = pred[m].cpu().numpy()
                valid_labels = label[m].cpu().numpy()
                all_preds.extend(valid_preds)
                all_labels.extend(valid_labels)
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    
    # Filter out ignore labels for metrics
    if task_config.ignore_labels:
        ignore_ids = [task_config.label2id[label] for label in task_config.ignore_labels]
        filtered_preds = []
        filtered_labels = []
        
        for pred, label in zip(all_preds, all_labels):
            if label not in ignore_ids:
                filtered_preds.append(pred)
                filtered_labels.append(label)
        
        all_preds = filtered_preds
        all_labels = filtered_labels
    
    # Calculate overall metrics (macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Calculate per-label metrics
    label_ids = sorted(set(all_labels) | set(all_preds))
    per_label_precision, per_label_recall, per_label_f1, per_label_support = precision_recall_fscore_support(
        all_labels, all_preds, labels=label_ids, average=None, zero_division=0
    )
    
    # Build per-label metrics dictionary
    per_label_metrics = {}
    for i, label_id in enumerate(label_ids):
        label_name = task_config.id2label.get(label_id, str(label_id))
        per_label_metrics[label_name] = {
            'precision': float(per_label_precision[i]),
            'recall': float(per_label_recall[i]),
            'f1': float(per_label_f1[i]),
            'support': int(per_label_support[i])
        }
    
    return {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_label': per_label_metrics
    }


def train_epoch(model, train_loader, optimizer, scheduler, device, config, logger, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None and scaler.is_enabled()
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

        if loss is None:
            raise ValueError("Model did not return a loss. Ensure labels are passed correctly.")
        # DataParallel gathers per-device scalar losses into a vector; reduce to scalar.
        if loss.dim() > 0:
            loss = loss.mean()
        
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        step_loss = loss.item() * (config.gradient_accumulation_steps if config.gradient_accumulation_steps > 1 else 1)
        total_loss += step_loss
        progress_bar.set_postfix({'loss': f'{step_loss:.4f}'})
        
        # Log loss for each step
        logger.info(
            "Epoch %d | Step %d/%d | Loss %.4f",
            epoch, step + 1, len(train_loader), step_loss
        )
    
    return total_loss / len(train_loader)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    task_config,
    training_config,
    logger,
    save_path,
    train_sampler=None
):
    """Train model with early stopping"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    scaler = torch.amp.GradScaler('cuda', enabled=training_config.fp16 and torch.cuda.is_available())
    
    steps_per_epoch = math.ceil(len(train_loader) / training_config.gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * training_config.num_epochs
    num_warmup_steps = int(num_training_steps * training_config.warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=num_training_steps
    )
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    if is_main_process():
        logger.info("\n" + "="*70)
        logger.info("TRAINING START")
        logger.info("="*70)
    
    for epoch in range(training_config.num_epochs):
        # Set epoch for DistributedSampler (important for proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if is_main_process():
            logger.info(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            training_config.device,
            training_config,
            logger,
            epoch + 1,
            scaler
        )
        
        # Validate
        val_metrics = evaluate_model(
            model,
            val_loader,
            task_config,
            training_config.device,
            use_amp=training_config.fp16
        )
        
        if is_main_process():
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Precision (Overall): {val_metrics['precision']:.4f}")
            logger.info(f"Val Recall (Overall): {val_metrics['recall']:.4f}")
            logger.info(f"Val F1 (Overall): {val_metrics['f1']:.4f}")
            
            # Log per-label metrics
            logger.info("\n  Per-Label Metrics:")
            logger.info("  {:<12} {:>10} {:>10} {:>10} {:>10}".format(
                "Label", "Precision", "Recall", "F1", "Support"))
            logger.info("  " + "-" * 52)
            for label_name, metrics in val_metrics['per_label'].items():
                logger.info("  {:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10}".format(
                    label_name,
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1'],
                    metrics['support']
                ))
        
        # Early stopping (only main process makes decisions, but all processes follow)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            # Save model state - for DDP, save the underlying module
            if is_main_process():
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, save_path)
                logger.info(f"✓ New best F1: {best_val_f1:.4f} - Model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if is_main_process():
                logger.info(f"Patience: {patience_counter}/{training_config.early_stopping_patience}")
            
            if patience_counter >= training_config.early_stopping_patience:
                if is_main_process():
                    logger.info(f"\n⚠ Early stopping triggered!")
                break
    
    if is_main_process():
        logger.info("\n" + "="*70)
        logger.info(f"Training complete! Best Val F1: {best_val_f1:.4f}")
        logger.info("="*70)
    
    return best_val_f1


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    # Set multiprocessing start method for CUDA compatibility on Slurm
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(description='Train SikuBERT for Token Classification')
    
    # Task configuration
    parser.add_argument('--task', type=str, required=True, 
                       choices=['punctuation', 'segmentation'],
                       help='Task type')
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_path', type=str, required=True,
                       help='Path to validation data')
    parser.add_argument('--test_path', type=str, required=True,
                       help='Path to test data')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='SIKU-BERT/sikubert',
                       help='Pretrained model name')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm (for clipping)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of steps for gradient accumulation')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # DataLoader configuration (important for Slurm)
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of DataLoader workers (set 0 for Slurm compatibility)')
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='Use mixed precision training (FP16)')
    parser.add_argument('--pin_memory', action='store_true', default=False,
                       help='Pin memory for DataLoader')
    parser.add_argument('--persistent_workers', action='store_true', default=False,
                       help='Use persistent workers for DataLoader')
    
    # CNN configuration
    parser.add_argument('--cnn_kernel_sizes', type=int, nargs='+', default=[3, 5, 7],
                       help='CNN kernel sizes (only used when head_type=cnn)')
    parser.add_argument('--cnn_num_filters', type=int, default=256,
                       help='Number of CNN filters (only used when head_type=cnn)')
    
    # Head type configuration
    parser.add_argument('--head_type', type=str, default='cnn',
                       choices=['softmax', 'crf', 'cnn'],
                       help='Classification head type: softmax (FC only), crf (BERT+CRF), cnn (BERT+CNN)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--model_save_dir', type=str, default='models',
                       help='Model save directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    
    # Resume from checkpoint (for multi-phase training)
    parser.add_argument('--resume_from_checkpoint', type=str, default='',
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Setup DDP if available FIRST (torchrun sets these environment variables)
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
    
    # Create output directories (only main process to avoid race conditions)
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.model_save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
    
    # Wait for main process to create directories
    if use_ddp:
        dist.barrier()
    
    # Setup logging - only main process writes to file
    log_file = os.path.join(args.log_dir, f"train_{args.task}.log")
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
        # Non-main processes: minimal logging (only warnings/errors)
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s | [Rank %(process)d] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
    logger = logging.getLogger(__name__)
    
    # Log arguments (only from main process)
    if is_main_process():
        logger.info("="*70)
        logger.info("TRAINING CONFIGURATION")
        logger.info("="*70)
        for arg, value in vars(args).items():
            logger.info(f"{arg}: {value}")
        logger.info("="*70)
    
    # Set seed
    set_seed(args.seed)
    
    if is_main_process():
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
        logger.info(f"  Num labels: {task_config.num_labels}")
    
    # Training configuration
    training_config = TrainingConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        seed=args.seed,
        num_workers=args.num_workers,
        fp16=args.fp16,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )
    
    # Load tokenizer
    if is_main_process():
        logger.info("\n✓ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    if is_main_process():
        logger.info("✓ Loading data...")
    train_texts, train_labels = load_data(args.train_path)
    val_texts, val_labels = load_data(args.val_path)
    
    if is_main_process():
        logger.info(f"  Train samples: {len(train_texts)}")
        logger.info(f"  Val samples: {len(val_texts)}")
    
    # Create dataloaders
    if is_main_process():
        logger.info("✓ Creating dataloaders...")
    train_loader, val_loader, train_sampler = create_dataloaders(
        train_texts, train_labels,
        val_texts, val_labels,
        tokenizer, task_config, training_config,
        use_ddp=use_ddp
    )
    
    # Create model
    if is_main_process():
        logger.info(f"✓ Creating model with head_type={args.head_type}...")
    model = SikuBERTForTokenClassification(
        model_name=args.model_name,
        num_labels=task_config.num_labels,
        dropout=args.dropout,
        head_type=args.head_type,
        cnn_kernel_sizes=args.cnn_kernel_sizes,
        cnn_num_filters=args.cnn_num_filters
    )
    
    # Load checkpoint if resuming
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        if is_main_process():
            logger.info(f"\n✓ Resuming from checkpoint: {args.resume_from_checkpoint}")
        model.load_state_dict(torch.load(args.resume_from_checkpoint, weights_only=True))

    model = model.to(device)
    
    # Wrap model with DDP
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process():
            logger.info(f"  Model wrapped with DistributedDataParallel")
    
    num_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        logger.info(f"  Total parameters: {num_params:,}")
    
    # Extract train file name for checkpoint naming
    train_file_name = os.path.splitext(os.path.basename(args.train_path))[0]
    save_path = os.path.join(args.model_save_dir, f"best_{args.task}_model_{args.head_type}.pt")
    
    best_val_f1 = train_with_early_stopping(
        model, train_loader, val_loader,
        task_config, training_config,
        logger, save_path,
        train_sampler=train_sampler
    )
    
    # Save training info (for tracking multi-phase training)
    if is_main_process():
        train_info = {
            'task': args.task,
            'train_path': args.train_path,
            'train_file': train_file_name,
            'val_path': args.val_path,
            'best_val_f1': best_val_f1,
            'resumed_from': args.resume_from_checkpoint if args.resume_from_checkpoint else None,
            'model_saved_to': save_path,
            'config': vars(args)
        }
        
        train_info_path = os.path.join(args.output_dir, f"{args.task}_train_info_{train_file_name}.json")
        with open(train_info_path, 'w', encoding='utf-8') as f:
            json.dump(train_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Training info saved to: {train_info_path}")
        logger.info("="*70)
        logger.info("🎉 TRAINING COMPLETE!")
        logger.info(f"  Best Val F1: {best_val_f1:.4f}")
        logger.info(f"  Model saved: {save_path}")
        logger.info(f"  Train file: {train_file_name}")
        logger.info("="*70)
        logger.info("")
        logger.info("To evaluate on test set, run:")
        logger.info(f"  python evaluate.py --task {args.task} --model_path {save_path} --test_path <test_path>")
        logger.info("="*70)
    
    # Cleanup DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()

