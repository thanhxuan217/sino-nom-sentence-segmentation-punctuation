#!/usr/bin/env python3
"""
Configuration classes for SikuBERT Fine-tuning
"""

from dataclasses import dataclass
from typing import List, Dict

import torch


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
