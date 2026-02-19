#!/usr/bin/env python3
"""
Custom callbacks for SikuBERT training
"""

from transformers.trainer_callback import TrainerCallback


# ============================================================================
# CUSTOM CALLBACKS
# ============================================================================

class LimitedEvalCallback(TrainerCallback):
    """Callback to limit evaluation samples during training"""
    
    def __init__(self, max_eval_samples: int = 1000):
        self.max_eval_samples = max_eval_samples
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called during evaluation"""
        # This is handled by setting max_eval_samples in eval_dataset
        pass
