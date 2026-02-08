#!/usr/bin/env python3
"""
SikuBERT Evaluation Script with Multi-GPU (DDP) Support
Evaluates trained model on test set and generates predictions
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
from dataclasses import dataclass
from typing import List, Dict

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


# ============================================================================
# DDP UTILITIES
# ============================================================================

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


def gather_results(local_results: list, world_size: int) -> list:
    """Gather results from all processes to main process"""
    if not dist.is_initialized():
        return local_results
    
    # Gather all results to rank 0
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    
    if is_main_process():
        # Flatten the list of lists
        all_results = []
        for r in gathered:
            all_results.extend(r)
        return all_results
    return []


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
                label_ids.append(-100)
            else:
                label = label_seq[word_id]
                label_ids.append(self.config.label2id[label])
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'idx': idx  # Store index for gathering results
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
        cnn_num_filters: int = 128
    ):
        super().__init__()
        
        self.head_type = head_type
        self.num_labels = num_labels
        
        self.bert = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
            add_pooling_layer=False
        )

        self.hidden_size = self.bert.config.hidden_size
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
        
        self.classifier = nn.Linear(classifier_input_size, num_labels)
        
        # CRF layer (only for 'crf' head type)
        self.crf = None
        if head_type == 'crf':
            self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        if self.cnn_layer is not None:
            sequence_output = self.cnn_layer(sequence_output)
            sequence_output = self.dropout(sequence_output)
        
        outputs = {'logits': emissions}
        loss = None
        
        if self.head_type == 'crf':
            # CRF head
            if labels is not None:
                # Use attention_mask as CRF mask (ensures first token is masked in)
                crf_mask = attention_mask.bool()
                
                # Replace -100 with 0 for CRF (will be ignored by mask effectively, but needed for input validity)
                # Note: labels might have -100 at [CLS] which is masked IN by attention_mask, so we must have valid label there.
                crf_labels = labels.clone()
                crf_labels[labels == -100] = 0
                
                # CRF forward returns negative log-likelihood
                loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction='mean')
                outputs['loss'] = loss
            
            # Viterbi decoding
            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=crf_mask)
            
            # Convert list of lists to tensor (padding with -100)
            max_len = emissions.size(1)
            batch_size = emissions.size(0)
            pred_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long, device=emissions.device)
            
            for i, pred_seq in enumerate(predictions):
                pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=emissions.device)
            
            outputs['predictions'] = pred_tensor

        else:
            # Softmax or CNN head (both use CrossEntropyLoss)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
                outputs['loss'] = loss
            
            # Argmax for softmax/cnn heads
            outputs['predictions'] = torch.argmax(emissions, dim=-1)
            
        return outputs
    
    def decode(self, input_ids, attention_mask):
        """Decode predictions (especially useful for CRF)"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        if self.cnn_layer is not None:
            sequence_output = self.cnn_layer(sequence_output)
            sequence_output = self.dropout(sequence_output)
        
        emissions = self.classifier(sequence_output)
        
        if self.head_type == 'crf':
            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=crf_mask)
            return predictions
        else:
            predictions = torch.argmax(emissions, dim=-1)
            return predictions.tolist()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_data(json_path: str) -> tuple:
    """Load data from JSON file"""
    texts = []
    labels = []
    with open(json_path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
            for item in data:
                texts.append(item["text"])
                labels.append(item["labels"])
        else:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(item["labels"])

    return texts, labels


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
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model_ddp(model, dataloader, task_config, device, use_amp: bool = False):
    """Evaluate model on a dataset with DDP support"""
    model.eval()
    
    local_preds = []
    local_labels = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", disable=not is_main_process())
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp and torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs['loss']
            if loss is not None:
                if loss.dim() > 0:
                    loss = loss.mean()
                total_loss += loss.item()
            num_batches += 1
            
            if 'predictions' in outputs:
                predictions = outputs['predictions']
            else:
                predictions = torch.argmax(outputs['logits'], dim=-1)
            mask = labels != -100
            
            for pred, label, m in zip(predictions, labels, mask):
                valid_preds = pred[m].cpu().numpy()
                valid_labels = label[m].cpu().numpy()
                local_preds.extend(valid_preds.tolist())
                local_labels.extend(valid_labels.tolist())
    
    # Gather results from all processes
    if dist.is_initialized():
        # Gather predictions and labels
        all_preds_gathered = [None] * dist.get_world_size()
        all_labels_gathered = [None] * dist.get_world_size()
        dist.all_gather_object(all_preds_gathered, local_preds)
        dist.all_gather_object(all_labels_gathered, local_labels)
        
        if is_main_process():
            all_preds = []
            all_labels = []
            for p, l in zip(all_preds_gathered, all_labels_gathered):
                all_preds.extend(p)
                all_labels.extend(l)
        else:
            all_preds = local_preds
            all_labels = local_labels
        
        # Gather loss
        loss_tensor = torch.tensor([total_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor[0].item() / loss_tensor[1].item() if loss_tensor[1].item() > 0 else 0.0
    else:
        all_preds = local_preds
        all_labels = local_labels
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Only main process calculates metrics
    if is_main_process():
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
    else:
        return {'loss': avg_loss, 'precision': 0, 'recall': 0, 'f1': 0, 'per_label': {}}


def run_test_set_ddp(
    model,
    tokenizer,
    config: TaskConfig,
    device: str,
    test_texts: List[str],
    test_labels: List[List[str]],
    dataloader,
    output_path: str,
    max_length: int = 256,
    logger=None
):
    """Run predictions on test set with DDP support"""
    model.eval()
    
    local_results = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Running predictions", disable=not is_main_process())
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            indices = batch['idx'].tolist()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if 'predictions' in outputs:
                predictions = outputs['predictions']
            else:
                predictions = torch.argmax(outputs['logits'], dim=-1)
            
            # Process each sample in batch
            for i, idx in enumerate(indices):
                text = test_texts[idx]
                gold_labels = test_labels[idx]
                
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
                
                local_results.append({
                    "idx": idx,
                    "text": text,
                    "gold_labels": gold_labels,
                    "pred_labels": pred_labels,
                    "gold_text_labeled": gold_text,
                    "pred_text_labeled": pred_text,
                })
    
    # Gather results from all processes
    if dist.is_initialized():
        all_results_gathered = [None] * dist.get_world_size()
        dist.all_gather_object(all_results_gathered, local_results)
        
        if is_main_process():
            all_results = []
            for r in all_results_gathered:
                all_results.extend(r)
            # Sort by original index to maintain order
            all_results.sort(key=lambda x: x['idx'])
            # Remove idx field
            for r in all_results:
                del r['idx']
        else:
            all_results = []
    else:
        all_results = local_results
        for r in all_results:
            del r['idx']
    
    # Only main process saves results
    if is_main_process():
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        if logger:
            logger.info(f"✓ Predictions saved to: {output_path}")
            logger.info(f"  Total samples: {len(all_results)}")
            
            logger.info("\n" + "="*70)
            logger.info("SAMPLE PREDICTIONS")
            logger.info("="*70)
            for idx in range(min(3, len(all_results))):
                sample = all_results[idx]
                logger.info(f"\n--- Sample {idx + 1} ---")
                logger.info(f"Original:  {sample['text'][:100]}...")
                logger.info(f"Gold:      {sample['gold_text_labeled'][:100]}...")
                logger.info(f"Predicted: {sample['pred_text_labeled'][:100]}...")


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
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_path', type=str, required=True,
                       help='Path to test data')
    
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
    log_file = os.path.join(args.log_dir, f"evaluate_{args.task}.log")
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
    
    # Create model and load weights
    if is_main_process():
        logger.info(f"✓ Loading model with head_type={args.head_type}...")
    model = SikuBERTForTokenClassification(
        model_name=args.model_name,
        num_labels=task_config.num_labels,
        dropout=args.dropout,
        head_type=args.head_type,
        cnn_kernel_sizes=args.cnn_kernel_sizes,
        cnn_num_filters=args.cnn_num_filters
    )
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Wrap with DDP for inference (ensures consistent behavior)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process():
            logger.info(f"  Model wrapped with DistributedDataParallel")
    
    if is_main_process():
        logger.info(f"  Loaded from: {args.model_path}")
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Total parameters: {num_params:,}")
    
    # Load test data
    if is_main_process():
        logger.info("\n✓ Loading test data...")
    test_texts, test_labels = load_data(args.test_path)
    if is_main_process():
        logger.info(f"  Test samples: {len(test_texts)}")
    
    # Create test dataloader with DistributedSampler
    test_dataset = ClassicalChineseDataset(
        test_texts, test_labels, tokenizer, task_config, args.max_length
    )
    
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if use_ddp else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    # Evaluate
    if is_main_process():
        logger.info("\n" + "="*70)
        logger.info("TEST SET EVALUATION")
        logger.info("="*70)
    
    test_metrics = evaluate_model_ddp(
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
    
    predictions_path = os.path.join(args.output_dir, f"{args.task}_predictions.json")
    run_test_set_ddp(
        model=model,
        tokenizer=tokenizer,
        config=task_config,
        device=device,
        test_texts=test_texts,
        test_labels=test_labels,
        dataloader=test_loader,
        output_path=predictions_path,
        max_length=args.max_length,
        logger=logger if is_main_process() else None
    )
    
    # Save results (only main process)
    if is_main_process():
        results = {
            'task': args.task,
            'model_path': args.model_path,
            'test_path': args.test_path,
            'test_metrics': test_metrics,
            'config': vars(args)
        }
        
        results_path = os.path.join(args.output_dir, f"{args.task}_eval_results.json")
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
