#!/usr/bin/env python3
"""
SikuBERT Fine-tuning with CNN for Token Classification
Converted from Jupyter notebook for SLURM execution
"""

import argparse
import json
import logging
import os
import random
import math
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
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
    """SikuBERT with CNN classification head"""
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        use_extra_layer: bool = False,
        extra_layer_type: str = None,
        cnn_kernel_sizes: list = None,
        cnn_num_filters: int = 128
    ):
        super().__init__()
        
        # Backbone: SikuBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Extra layers
        self.extra_layer = None
        self.extra_layer_type = extra_layer_type
        
        if use_extra_layer and extra_layer_type == 'cnn':
            if cnn_kernel_sizes is None:
                cnn_kernel_sizes = [3, 5, 7]
            
            self.extra_layer = MultiKernelCNN(
                hidden_size=self.hidden_size,
                kernel_sizes=cnn_kernel_sizes,
                num_filters=cnn_num_filters
            )
            classifier_input_size = self.extra_layer.output_size
        else:
            classifier_input_size = self.hidden_size
        
        # Classification head
        self.classifier = nn.Linear(classifier_input_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # Extra layer
        if self.extra_layer is not None:
            sequence_output = self.extra_layer(sequence_output)
            sequence_output = self.dropout(sequence_output)
        
        # Classification
        logits = self.classifier(sequence_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}


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
    training_config: TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    train_dataset = ClassicalChineseDataset(
        train_texts, train_labels, tokenizer, config, training_config.max_length
    )
    val_dataset = ClassicalChineseDataset(
        val_texts, val_labels, tokenizer, config, training_config.max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


def evaluate_model(model, dataloader, task_config, device):
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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs['loss'].item()
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
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(model, train_loader, optimizer, scheduler, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        
        loss.backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    task_config,
    training_config,
    logger,
    save_path
):
    """Train model with early stopping"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
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
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING START")
    logger.info("="*70)
    
    for epoch in range(training_config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                                training_config.device, training_config)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, task_config, training_config.device)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}")
        logger.info(f"Val Recall: {val_metrics['recall']:.4f}")
        logger.info(f"Val F1: {val_metrics['f1']:.4f}")
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), save_path)
            logger.info(f"✓ New best F1: {best_val_f1:.4f} - Model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{training_config.early_stopping_patience}")
            
            if patience_counter >= training_config.early_stopping_patience:
                logger.info(f"\n⚠ Early stopping triggered!")
                break
    
    logger.info("\n" + "="*70)
    logger.info(f"Training complete! Best Val F1: {best_val_f1:.4f}")
    logger.info("="*70)
    
    return best_val_f1


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train SikuBERT with CNN')
    
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
    
    # CNN configuration
    parser.add_argument('--cnn_kernel_sizes', type=int, nargs='+', default=[3, 5, 7],
                       help='CNN kernel sizes')
    parser.add_argument('--cnn_num_filters', type=int, default=256,
                       help='Number of CNN filters')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--model_save_dir', type=str, default='models',
                       help='Model save directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.log_dir, f"train_{args.task}.log")
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
    
    # Log arguments
    logger.info("="*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*70)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("="*70)
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n✓ Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
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
        seed=args.seed
    )
    
    # Load tokenizer
    logger.info("\n✓ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    logger.info("✓ Loading data...")
    train_texts, train_labels = load_data(args.train_path)
    val_texts, val_labels = load_data(args.val_path)
    
    logger.info(f"  Train samples: {len(train_texts)}")
    logger.info(f"  Val samples: {len(val_texts)}")
    
    # Create dataloaders
    logger.info("✓ Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_texts, train_labels,
        val_texts, val_labels,
        tokenizer, task_config, training_config
    )
    
    # Create model
    logger.info("✓ Creating model...")
    model = SikuBERTForTokenClassification(
        model_name=args.model_name,
        num_labels=task_config.num_labels,
        dropout=args.dropout,
        use_extra_layer=True,
        extra_layer_type='cnn',
        cnn_kernel_sizes=args.cnn_kernel_sizes,
        cnn_num_filters=args.cnn_num_filters
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {num_params:,}")
    
    # Train
    save_path = os.path.join(args.model_save_dir, f"best_{args.task}_model_cnn.pt")
    best_val_f1 = train_with_early_stopping(
        model, train_loader, val_loader,
        task_config, training_config,
        logger, save_path
    )
    
    # Evaluate on test set
    logger.info("\n" + "="*70)
    logger.info("TEST SET EVALUATION")
    logger.info("="*70)
    
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    test_texts, test_labels = load_data(args.test_path)
    logger.info(f"Test samples: {len(test_texts)}")
    
    test_dataset = ClassicalChineseDataset(
        test_texts, test_labels, tokenizer, task_config, training_config.max_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False
    )
    
    test_metrics = evaluate_model(model, test_loader, task_config, device)
    
    logger.info(f"\nTest Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    
    # Save results
    results = {
        'task': args.task,
        'test_metrics': test_metrics,
        'config': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, f"{args.task}_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Results saved to: {results_path}")
    logger.info("="*70)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
