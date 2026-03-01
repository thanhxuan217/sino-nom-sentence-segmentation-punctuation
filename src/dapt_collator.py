#!/usr/bin/env python3
"""
Custom Data Collator with Punctuation-Aware MLM Masking for DAPT.

Extends HuggingFace's DataCollatorForLanguageModeling to bias masking
toward punctuation tokens, improving downstream punctuation restoration.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


@dataclass
class PunctuationAwareMLMCollator(DataCollatorForLanguageModeling):
    """MLM collator that masks punctuation tokens with higher probability.

    Standard MLM masks 15% of tokens uniformly. For DAPT targeting punctuation
    restoration, we boost punctuation masking to ensure the model receives
    enough training signal for rare punctuation marks.

    Behavior:
        1. Punctuation tokens are masked with probability `punct_mask_prob` (default 30%).
        2. Remaining masking budget fills non-punctuation tokens to maintain
           the overall `mlm_probability` rate (~15%).
        3. The 80/10/10 replacement rule (mask/random/keep) is preserved.

    Args:
        tokenizer: The tokenizer used for encoding.
        mlm_probability: Overall masking probability (default 0.15).
        punct_tokens: Set of punctuation characters to boost.
        punct_mask_prob: Masking probability for punctuation tokens (default 0.30).
    """

    punct_tokens: Optional[List[str]] = field(
        default_factory=lambda: ["，", "。", "：", "、", "；", "？", "！"]
    )
    punct_mask_prob: float = 0.30

    def __post_init__(self):
        super().__post_init__()
        # Convert punctuation characters to token IDs
        self._punct_ids: Set[int] = set()

    def _ensure_punct_ids(self):
        """Lazily compute punctuation token IDs."""
        if not self._punct_ids and self.tokenizer is not None:
            ids = self.tokenizer.convert_tokens_to_ids(self.punct_tokens)
            # Filter out UNK tokens (in case punctuation not in vocabulary)
            unk_id = self.tokenizer.unk_token_id
            self._punct_ids = {tid for tid in ids if tid != unk_id}

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """Override to implement punctuation-aware masking.

        Args:
            inputs: Tensor of input token IDs [batch_size, seq_len].
            special_tokens_mask: Optional mask indicating special tokens.

        Returns:
            Tuple of (masked_inputs, labels) tensors.
        """
        self._ensure_punct_ids()

        labels = inputs.clone()
        batch_size, seq_len = inputs.shape

        # Build special tokens mask if not provided
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool, device=inputs.device
            )
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # Identify punctuation positions
        punct_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for pid in self._punct_ids:
            punct_mask |= (inputs == pid)

        # Non-special, non-punctuation positions
        normal_mask = ~special_tokens_mask & ~punct_mask

        # --- Step 1: Mask punctuation with boosted probability ---
        punct_random = torch.rand(inputs.shape, device=inputs.device)
        punct_selected = punct_mask & (punct_random < self.punct_mask_prob)

        # --- Step 2: Calculate remaining budget for normal tokens ---
        total_budget = int(seq_len * self.mlm_probability)  # per sequence
        masked_inputs = inputs.clone()

        # Process each sequence in the batch
        for i in range(batch_size):
            # Count how many punct tokens selected in this sequence
            n_punct_masked = punct_selected[i].sum().item()
            remaining_budget = max(0, total_budget - n_punct_masked)

            # Select remaining normal tokens
            normal_indices = normal_mask[i].nonzero(as_tuple=True)[0]
            if len(normal_indices) > 0 and remaining_budget > 0:
                perm = torch.randperm(len(normal_indices), device=inputs.device)
                n_select = min(remaining_budget, len(normal_indices))
                selected_normal = normal_indices[perm[:n_select]]

                # Mark selected normal tokens
                normal_selected_mask = torch.zeros(seq_len, dtype=torch.bool, device=inputs.device)
                normal_selected_mask[selected_normal] = True
            else:
                normal_selected_mask = torch.zeros(seq_len, dtype=torch.bool, device=inputs.device)

            # Combine: all selected positions for this sequence
            all_selected = punct_selected[i] | normal_selected_mask

            # --- Apply 80/10/10 replacement ---
            replace_random = torch.rand(seq_len, device=inputs.device)

            # 80% → [MASK]
            mask_positions = all_selected & (replace_random < 0.8)
            masked_inputs[i, mask_positions] = self.tokenizer.mask_token_id

            # 10% → random token
            random_positions = all_selected & (replace_random >= 0.8) & (replace_random < 0.9)
            random_tokens = torch.randint(
                len(self.tokenizer), (random_positions.sum().item(),),
                device=inputs.device,
            )
            masked_inputs[i, random_positions] = random_tokens

            # 10% → keep original (no change needed)

            # Labels: -100 for non-masked positions
            labels[i, ~all_selected] = -100

        return masked_inputs, labels
