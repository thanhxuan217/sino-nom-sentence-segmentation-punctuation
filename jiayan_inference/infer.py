import argparse
import pandas as pd
import yaml
import os
from tqdm import tqdm
from jiayan import load_lm, CRFPunctuator
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Chỉ xét các dấu câu thuộc danh sách yêu cầu
VALID_PUNC = {'，', '。', '：', '、', '；', '？', '！'}

def text_to_labels(text):
    """
    Trích xuất list các ký tự raw (không chứa dấu câu) và label tương ứng của chúng.
    Mỗi ký tự text sẽ được gán label bằng dấu câu ngay sau nó. 
    Nếu không có dấu câu nào hoặc dấu câu không thuộc list VALID_PUNC, label là 'O'.
    """
    chars = []
    labels = []
    for char in text:
        if char in VALID_PUNC:
            if len(labels) > 0 and labels[-1] == 'O':
                labels[-1] = char
        elif char.strip() == '':
            # Bỏ khoảng trắng nếu có để alignment chuẩn hơn
            continue
        else:
            # Xem mỗi char không phải khoảng trắng và không phải dấu câu target là con chữ
            chars.append(char)
            labels.append('O')
    return chars, labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate Jiayan Punctuation on Parquet Dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--test_path", type=str, help="Override test path passed from config")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    test_path = args.test_path if args.test_path else config['data']['test_path']
    text_col = config['data'].get('text_col', 'text')

    print(f"Loading data from {test_path}...")
    if not os.path.exists(test_path):
        print(f"Error: Dataset not found at {test_path}")
        return
        
    df = pd.read_parquet(test_path)

    print("Loading Jiayan models...")
    lm = load_lm(config['model']['lm_path'])
    punctuator = CRFPunctuator(lm, config['model']['cut_model_path'])
    punctuator.load(config['model']['punc_model_path'])

    all_true_labels = []
    all_pred_labels = []

    print("Running inference and aligning labels...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[text_col])
        
        # 1. Parse ground truth (lấy chuỗi raw không dấu và các nhãn thực)
        chars, true_labels = text_to_labels(text)
        
        # 2. Xây lại chuỗi raw text không chứa space và không chứa target punctuations
        raw_text = "".join(chars)
        if not raw_text:
            continue
            
        # 3. Model Inference cho text raw không dấu
        pred_text = punctuator.punctuate(raw_text)
        
        # 4. Parse prediction (lấy nhãn dự đoán)
        pred_chars, pred_labels = text_to_labels(pred_text)

        # 5. Căn chỉnh Sequence Length 
        # (Jiayan thường chỉ chèn thêm dấu câu nên length chars = nhau)
        min_len = min(len(true_labels), len(pred_labels))
        all_true_labels.extend(true_labels[:min_len])
        all_pred_labels.extend(pred_labels[:min_len])

        if len(true_labels) != len(pred_labels):
            print(f"\nWarning on row {idx}: length mismatch! True: {len(true_labels)}, Pred: {len(pred_labels)}")

    print("\n" + "="*50)
    print("1. Evaluation Results for Punctuation (Exact Match):")
    print("="*50)
    
    # Chỉ xét label list ('O', '，', '。', '：', '、', '；', '？', '！') cho Precision, Recall, F1
    labels_to_eval = ['，', '。', '：', '、', '；', '？', '！'] # Bỏ 'O' để tính average P/R/F1 của target
    
    report_punc = classification_report(
        all_true_labels, 
        all_pred_labels, 
        labels=labels_to_eval,
        zero_division=0,
        digits=4
    )
    print(report_punc)
    
    # Calculate Micro and Macro average
    precision_p, recall_p, f1_p, _ = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, labels=labels_to_eval, average='micro', zero_division=0
    )
    print(f"\nMicro Average -> Precision: {precision_p:.4f}, Recall: {recall_p:.4f}, F1: {f1_p:.4f}")
    
    precision_p, recall_p, f1_p, _ = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, labels=labels_to_eval, average='macro', zero_division=0
    )
    print(f"Macro Average -> Precision: {precision_p:.4f}, Recall: {recall_p:.4f}, F1: {f1_p:.4f}")

    print("\n" + "="*50)
    print("2. Evaluation Results for Sentence Segmentation (Binary Boundary):")
    print("="*50)
    
    # Map tất cả các dấu câu hợp lệ thành 'B' (Boundary - Vị trí ngắt) và 'O' giữ nguyên
    true_seg_labels = ['B' if l in VALID_PUNC else 'O' for l in all_true_labels]
    pred_seg_labels = ['B' if l in VALID_PUNC else 'O' for l in all_pred_labels]
    
    report_seg = classification_report(
        true_seg_labels, 
        pred_seg_labels, 
        labels=['B'],
        zero_division=0,
        digits=4
    )
    print(report_seg)
    
    precision_s, recall_s, f1_s, _ = precision_recall_fscore_support(
        true_seg_labels, pred_seg_labels, labels=['B'], average='micro', zero_division=0
    )
    print(f"\nSegmentation -> Precision: {precision_s:.4f}, Recall: {recall_s:.4f}, F1: {f1_s:.4f}")

if __name__ == "__main__":
    main()
