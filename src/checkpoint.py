#!/usr/bin/env python3
"""
Checkpoint loading utilities for SikuBERT
"""

from pathlib import Path

import torch
import torch.nn as nn


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
