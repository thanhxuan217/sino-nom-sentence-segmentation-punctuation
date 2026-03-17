#!/usr/bin/env python3
"""
Model loading and single-text inference logic for the API.

Wraps the existing SikuBERT + CNN model for on-demand inference
on individual text strings. Long inputs are handled with a simple
sliding window (overlap) and batched inference to avoid truncation.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from src.checkpoint import load_model_from_trainer_checkpoint
from src.config import TaskConfig
from src.model import SikuBERTForTokenClassification
from src.utils import apply_punctuation_labels, apply_segmentation_inline

logger = logging.getLogger(__name__)


# ============================================================================
# TASK CONFIGURATIONS (mirrors infer_sliding_window.py)
# ============================================================================

TASK_CONFIGS: Dict[str, TaskConfig] = {
    "segmentation": TaskConfig.create(
        task_name="segmentation",
        labels=["B", "M", "E", "S"],
        ignore_labels=[],
    ),
    "punctuation": TaskConfig.create(
        task_name="punctuation",
        labels=["O", "，", "。", "：", "、", "；", "？", "！"],
        ignore_labels=["O"],
    ),
}

# ============================================================================
# SLIDING-WINDOW STITCHING HELPERS (adapted from infer_sliding_window.py)
# ============================================================================

_SENTENCE_ENDING_PUNCTS = set("。？！，：、；")
_SEG_SEP = " | "


def build_raw_to_pred_map(raw_text: str, pred_text: str) -> List[int]:
    """Map each raw character index to its index in the predicted string."""
    raw_to_pred: List[int] = []
    pred_idx = 0

    for raw_char in raw_text:
        while pred_idx < len(pred_text) and pred_text[pred_idx] != raw_char:
            pred_idx += 1
        raw_to_pred.append(pred_idx)
        pred_idx += 1

    return raw_to_pred


def strip_last_sentence(pred_text: str, task_name: str) -> Tuple[str, bool]:
    """Remove the last sentence/segment from the predicted text.

    Returns (stripped_text, has_boundary).
    """
    if task_name == "punctuation":
        last_idx = -1
        for i in range(len(pred_text) - 1, -1, -1):
            if pred_text[i] in _SENTENCE_ENDING_PUNCTS:
                last_idx = i
                break
        if last_idx == -1:
            return "", False
        return pred_text[: last_idx + 1], True

    last_sep = pred_text.rfind(_SEG_SEP)
    if last_sep == -1:
        return "", False
    return pred_text[: last_sep + len(_SEG_SEP)], True


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """Loads and manages a single SikuBERT model for inference."""

    def __init__(
        self,
        task_name: str,
        model_path: str,
        model_name: str = "SIKU-BERT/sikubert",
        tokenizer_name: Optional[str] = None,
        max_length: int = 256,
        overlap: int = 128,
        head_type: str = "cnn",
        cnn_kernel_sizes: Optional[List[int]] = None,
        cnn_num_filters: int = 256,
        dropout: float = 0.1,
        use_qlora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
    ):
        self.task_name = task_name
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.max_length = max_length
        self.overlap = overlap
        self.head_type = head_type
        self.cnn_kernel_sizes = cnn_kernel_sizes or [3, 5, 7]
        self.cnn_num_filters = cnn_num_filters
        self.dropout = dropout
        self.use_qlora = use_qlora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["query", "key", "value"]

        self.task_config = TASK_CONFIGS[task_name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load the tokenizer and model weights."""
        logger.info(f"Loading {self.task_name} model from {self.model_path} ...")

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # --- QLoRA Config ---
        qlora_config = None
        if self.use_qlora:
            from peft import LoraConfig, TaskType
            qlora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )

        # --- Model ---
        self.model = SikuBERTForTokenClassification(
            model_name=self.model_name,
            num_labels=self.task_config.num_labels,
            dropout=self.dropout,
            head_type=self.head_type,
            cnn_kernel_sizes=self.cnn_kernel_sizes,
            cnn_num_filters=self.cnn_num_filters,
            use_qlora=self.use_qlora,
            qlora_config=qlora_config,
        )

        # Resize embeddings if the tokenizer has extra tokens
        if len(self.tokenizer) != self.model.bert.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        load_model_from_trainer_checkpoint(
            checkpoint_path=self.model_path,
            model=self.model,
            logger=logger,
        )

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info(f"✓ {self.task_name} model loaded on {self.device}.")

    def _max_chars_per_window(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        return max(1, self.max_length - special_tokens)

    @torch.no_grad()
    def _predict_labels_single(self, text: str) -> List[str]:
        """Predict labels for a single window (no truncation expected)."""
        # Tokenize character-by-character
        chars = list(text)
        tokenized = self.tokenizer(
            chars,
            is_split_into_words=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if "predictions" in outputs:
            predictions = outputs["predictions"][0]
        else:
            predictions = torch.argmax(outputs["logits"], dim=-1)[0]

        # Map token predictions → character labels
        word_ids = tokenized.word_ids()
        pred_labels: List[str] = []
        preds_cpu = predictions.cpu()

        for token_idx, word_id in enumerate(word_ids):
            if word_id is not None and token_idx < len(preds_cpu):
                label_id = preds_cpu[token_idx].item()
                pred_labels.append(self.task_config.id2label[label_id])

        return pred_labels

    @torch.no_grad()
    def _predict_labels_batch(self, texts: List[str]) -> List[List[str]]:
        """Predict labels for multiple windows in a single batch."""
        if not texts:
            return []

        tokenized = self.tokenizer(
            [list(t) for t in texts],
            is_split_into_words=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if "predictions" in outputs:
            predictions = outputs["predictions"]
        else:
            predictions = torch.argmax(outputs["logits"], dim=-1)

        preds_cpu = predictions.cpu()
        all_labels: List[List[str]] = []

        for i in range(len(texts)):
            word_ids = tokenized.word_ids(batch_index=i)
            pred_labels: List[str] = []
            preds_i = preds_cpu[i]
            for token_idx, word_id in enumerate(word_ids):
                if word_id is not None and token_idx < len(preds_i):
                    label_id = preds_i[token_idx].item()
                    pred_labels.append(self.task_config.id2label[label_id])
            all_labels.append(pred_labels)

        return all_labels

    def _apply_labels(self, text: str, labels: List[str]) -> str:
        if self.task_name == "punctuation":
            return apply_punctuation_labels(text, labels)
        if self.task_name == "segmentation":
            return apply_segmentation_inline(text, labels)
        return text

    def _has_any_boundary(self, labels_list: List[List[str]]) -> bool:
        if self.task_name == "punctuation":
            for labels in labels_list:
                if any(lbl in _SENTENCE_ENDING_PUNCTS for lbl in labels):
                    return True
            return False
        for labels in labels_list:
            if any(lbl in ("E", "S") for lbl in labels):
                return True
        return False

    def _stitch_fragments(
        self,
        fragments: List[Tuple[str, str, List[str]]],
        overlap: int,
    ) -> Tuple[str, List[str]]:
        """Stitch sliding-window fragments into a single output."""
        doc_parts: List[str] = []
        doc_pred_labels: List[str] = []

        prev_raw: Optional[str] = None
        prev_pred: Optional[str] = None
        prev_pred_labels: Optional[List[str]] = None
        prev_overlap_start_raw = 0

        for raw_text, pred_text, pred_labels in fragments:
            if prev_raw is None:
                prev_raw = raw_text
                prev_pred = pred_text
                prev_pred_labels = pred_labels
                prev_overlap_start_raw = 0
                continue

            valid_prev_pred, _ = strip_last_sentence(prev_pred, self.task_name)

            if valid_prev_pred:
                if prev_overlap_start_raw > 0:
                    r2p = build_raw_to_pred_map(prev_raw, prev_pred)
                    clamped = min(prev_overlap_start_raw, len(r2p) - 1)
                    if clamped < 0 or prev_overlap_start_raw >= len(prev_raw):
                        valid_prev_pred = ""
                    else:
                        pred_start = r2p[clamped]
                        valid_prev_pred = valid_prev_pred[pred_start:]

                if valid_prev_pred:
                    doc_parts.append(valid_prev_pred)

            if valid_prev_pred and prev_pred:
                r2p_full = build_raw_to_pred_map(prev_raw, prev_pred)
                pred_start = (
                    r2p_full[prev_overlap_start_raw]
                    if prev_overlap_start_raw > 0
                    else 0
                )
                pred_end = pred_start + len(valid_prev_pred)

                valid_raw_end = prev_overlap_start_raw
                for ri in range(prev_overlap_start_raw, len(prev_raw)):
                    if r2p_full[ri] >= pred_end:
                        break
                    valid_raw_end = ri + 1
                else:
                    valid_raw_end = len(prev_raw)

                doc_pred_labels.extend(
                    prev_pred_labels[prev_overlap_start_raw:valid_raw_end]
                )

                overlap_zone_start = len(prev_raw) - overlap
                kept_in_overlap = max(0, valid_raw_end - overlap_zone_start)
                curr_overlap_start_raw = (
                    min(kept_in_overlap, len(raw_text) - 1)
                    if len(raw_text) > 0
                    else 0
                )
            else:
                curr_overlap_start_raw = overlap

            prev_raw = raw_text
            prev_pred = pred_text
            prev_pred_labels = pred_labels
            prev_overlap_start_raw = curr_overlap_start_raw

        if prev_pred is not None:
            if prev_overlap_start_raw > 0:
                r2p = build_raw_to_pred_map(prev_raw, prev_pred)
                if prev_overlap_start_raw < len(r2p):
                    pred_start = r2p[prev_overlap_start_raw]
                    doc_parts.append(prev_pred[pred_start:])
            else:
                doc_parts.append(prev_pred)

            doc_pred_labels.extend(prev_pred_labels[prev_overlap_start_raw:])

        result_text = "".join(doc_parts)
        return result_text, doc_pred_labels

    def _stitch_fragments_no_strip(
        self,
        fragments: List[Tuple[str, str, List[str]]],
        overlap: int,
    ) -> Tuple[str, List[str]]:
        """Stitch fragments by simple overlap removal (no sentence stripping)."""
        doc_parts: List[str] = []
        doc_pred_labels: List[str] = []

        prev_raw: Optional[str] = None
        prev_pred: Optional[str] = None
        prev_pred_labels: Optional[List[str]] = None
        prev_overlap_start_raw = 0

        for raw_text, pred_text, pred_labels in fragments:
            if prev_raw is None:
                prev_raw = raw_text
                prev_pred = pred_text
                prev_pred_labels = pred_labels
                prev_overlap_start_raw = 0
                continue

            if prev_overlap_start_raw > 0:
                r2p = build_raw_to_pred_map(prev_raw, prev_pred)
                if prev_overlap_start_raw < len(r2p):
                    pred_start = r2p[prev_overlap_start_raw]
                    doc_parts.append(prev_pred[pred_start:])
            else:
                doc_parts.append(prev_pred)

            doc_pred_labels.extend(prev_pred_labels[prev_overlap_start_raw:])

            prev_raw = raw_text
            prev_pred = pred_text
            prev_pred_labels = pred_labels
            prev_overlap_start_raw = overlap

        if prev_pred is not None:
            if prev_overlap_start_raw > 0:
                r2p = build_raw_to_pred_map(prev_raw, prev_pred)
                if prev_overlap_start_raw < len(r2p):
                    pred_start = r2p[prev_overlap_start_raw]
                    doc_parts.append(prev_pred[pred_start:])
            else:
                doc_parts.append(prev_pred)

            doc_pred_labels.extend(prev_pred_labels[prev_overlap_start_raw:])

        result_text = "".join(doc_parts)
        return result_text, doc_pred_labels

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, List[str]]:
        """Run inference on a text string (supports long inputs via sliding window).

        Returns
        -------
        result_text : str
            Text with labels applied (punctuation marks or segment separators).
        labels : list[str]
            Per-character predicted labels.
        """
        if not self._loaded:
            raise RuntimeError(f"{self.task_name} model is not loaded.")

        max_chars = self._max_chars_per_window()
        overlap = min(self.overlap, max_chars - 1) if max_chars > 1 else 0
        step = max_chars - overlap

        if len(text) <= max_chars:
            pred_labels = self._predict_labels_single(text)
            result_text = self._apply_labels(text, pred_labels)
            return result_text, pred_labels

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start += step

        # Run all windows in parallel (single batch), then stitch.
        batch_labels = self._predict_labels_batch(chunks)
        has_any_boundary = self._has_any_boundary(batch_labels)
        fragments: List[Tuple[str, str, List[str]]] = []
        for chunk, chunk_labels in zip(chunks, batch_labels):
            chunk_pred_text = self._apply_labels(chunk, chunk_labels)
            fragments.append((chunk, chunk_pred_text, chunk_labels))

        if has_any_boundary:
            result_text, pred_labels = self._stitch_fragments(fragments, overlap)
        else:
            result_text, pred_labels = self._stitch_fragments_no_strip(fragments, overlap)

        return result_text, pred_labels
