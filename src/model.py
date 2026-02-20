#!/usr/bin/env python3
"""
Model architecture for SikuBERT Token Classification
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torchcrf import CRF


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
