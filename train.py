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
import random
import math
import sys
import pyarrow.parquet as pq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset, IterableDataset
from transformers import (
    AutoModel, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    BertForTokenClassification
)
from transformers.trainer_callback import TrainerCallback
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from torchcrf import CRF
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# ============================================================================
# CONFIGURATION CLASSES (Keep existing for compatibility)
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
    """Training hyperparameters - Extended version"""
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

    # Step-based training (NEW)
    max_steps: int = -1  # -1 means use num_epochs instead
    eval_steps: int = 500  # Evaluate every N steps
    save_steps: int = 500  # Save checkpoint every N steps
    logging_steps: int = 100  # Log every N steps
    
    # Eval strategy (NEW)
    max_eval_samples: int = 1000  # Max samples for intermediate evaluation
    
    # Early stopping
    early_stopping_patience: int = 10  # Changed from 3 to 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    head_type: str = "cnn"
    output_dir: str = "outputs"
    
    # QLoRA configuration (NEW)
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # Streaming configuration (NEW)
    use_streaming: bool = True
    streaming_buffer_size: int = 10000


# ============================================================================
# MODEL ARCHITECTURE (Keep existing classes)
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
        # Gọi BERT (đã bọc QLoRA). 
        # Nhờ bước này, 'labels' sẽ KHÔNG bao giờ bị lọt vào trong self.bert

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
# STREAMING DATA UTILITIES (NEW)
# ============================================================================

def load_streaming_dataset(data_dir: str, split: str = "train"):
    """Load dataset from multiple parquet files in streaming mode
    
    Args:
        data_dir: Directory containing parquet files (e.g., data/train/*.parquet)
        split: Split name (train/val/test)
    
    Returns:
        IterableDataset or Dataset
    """
    data_path = Path(data_dir) / split
    
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_path}")
    
    parquet_files = list(data_path.glob("*.parquet"))
    

    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(f) for f in parquet_files]},
        streaming=True
    )["train"]
    
    return dataset


def preprocess_function(examples, tokenizer, task_config, max_length=256):
    """Preprocess function for token classification
    
    Compatible with both streaming and regular datasets
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
    return tokenized_inputs


# ============================================================================
# METRICS COMPUTATION (NEW - for Trainer)
# ============================================================================

def compute_metrics(eval_pred, task_config):
    """Compute metrics for Trainer"""
    predictions, labels = eval_pred
    
    # Handle both logits and predictions
    if len(predictions.shape) == 3:  # logits
        predictions = np.argmax(predictions, axis=2)
    
    # Flatten and filter valid labels
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                # Filter ignore labels
                if task_config.ignore_labels:
                    ignore_ids = [task_config.label2id[l] for l in task_config.ignore_labels]
                    if label in ignore_ids:
                        continue
                true_labels.append(label)
                pred_labels.append(pred)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# ============================================================================
# CUSTOM CALLBACKS (NEW)
# ============================================================================

class LimitedEvalCallback(TrainerCallback):
    """Callback to limit evaluation samples during training"""
    
    def __init__(self, max_eval_samples: int = 1000):
        self.max_eval_samples = max_eval_samples
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called during evaluation"""
        # This is handled by setting max_eval_samples in eval_dataset
        pass


# ============================================================================
# TRAINING UTILITIES (Keep existing functions)
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    """Run predictions on test set and save results"""
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
# MAIN TRAINING FUNCTION (NEW - using Trainer API)
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
        seed=args.seed,
        # [FIX QUAN TRỌNG CHO DDP + QLoRA]
        # Tắt tính năng tự động tìm tham số không sử dụng của DDP
        # để tránh xung đột với Gradient Checkpointing.
        ddp_find_unused_parameters=False
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
