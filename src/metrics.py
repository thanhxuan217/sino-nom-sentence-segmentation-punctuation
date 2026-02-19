#!/usr/bin/env python3
"""
Metrics computation for SikuBERT training
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report


# ============================================================================
# METRICS COMPUTATION (for Trainer)
# ============================================================================

def compute_metrics(eval_pred, task_config):
    """
    Tính metrics tổng hợp VÀ in ra chi tiết từng nhãn (Per-label metrics).
    Tối ưu hóa bộ nhớ: Tính toán trên Numpy int, không convert list string.
    """
    predictions, labels = eval_pred
    
    # [1] Xử lý tuple (logits, hidden_states) nếu có
    if isinstance(predictions, tuple):
        found_preds = None
        for item in predictions:
            if isinstance(item, np.ndarray) and item.ndim == 2:
                found_preds = item
                break
        predictions = found_preds if found_preds is not None else predictions[0]

    # [2] Argmax nếu là logits 3D
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=2)

    # [3] Flatten & Masking (Thao tác RAM thấp)
    true_predictions = predictions.flatten()
    true_labels = labels.flatten()
    
    mask = true_labels != -100
    
    # Lấy dữ liệu sạch (vẫn là dạng số ID)
    filtered_preds = true_predictions[mask]
    filtered_labels = true_labels[mask]
    
    # ========================================================================
    # IN BÁO CÁO CHI TIẾT TỪNG LABEL (Classification Report)
    # ========================================================================
    
    # Lấy danh sách các ID xuất hiện trong tập này để map tên
    # (Để tránh lỗi nếu tập eval thiếu một số nhãn hiếm)
    unique_ids = sorted(list(set(filtered_labels) | set(filtered_preds)))
    target_names = [task_config.id2label[i] for i in unique_ids]
    
    # In ra màn hình console (Log)
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT (Per-Label Metrics)")
    print("-" * 60)
    try:
        print(classification_report(
            filtered_labels, 
            filtered_preds, 
            labels=unique_ids,
            target_names=target_names, 
            digits=4, # Hiển thị 4 chữ số thập phân
            zero_division=0
        ))
    except Exception as e:
        print(f"Error printing report: {e}")
    print("="*60 + "\n")

    # ========================================================================
    # TRẢ VỀ METRICS TỔNG HỢP CHO TRAINER
    # ========================================================================
    
    # Vẫn cần trả về số tổng (Macro Average) để Trainer chọn best model
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_labels, filtered_preds, average='macro', zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
