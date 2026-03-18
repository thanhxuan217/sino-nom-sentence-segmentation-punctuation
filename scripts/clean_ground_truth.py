import unicodedata
import sys

# Đảm bảo in được tiếng Việt/Trung trên Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def clean_ground_truth(text: str) -> str:
    """
    Giữ lại các dấu câu model dự đoán: ，。：、；？！
    Loại bỏ tất cả các loại dấu câu khác (ngoặc, nháy, dấu chấm Latin, v.v.)
    """
    # Tập hợp các dấu câu muốn giữ lại (model-predicted characters)
    allowed_punct = set("，。：、；？！")
    
    # Loại bỏ ký tự nếu:
    # 1. Nó thuộc nhóm Punctuation ('P*') và không nằm trong tập được cho phép
    # 2. Hoặc nó là ký tự khoảng trắng/xuống hàng
    cleaned_text = "".join(
        ch for ch in text 
        if not (unicodedata.category(ch).startswith("P") and ch not in allowed_punct)
        and not ch.isspace()
    )
    
    return cleaned_text.strip()

if __name__ == "__main__":
    # Ví dụ sử dụng
    sample_text = ""
    print("Dữ liệu gốc:", sample_text)
    print("Dữ liệu sạch:", clean_ground_truth(sample_text))
    # Kết quả mong đợi: "Lê Lợi tui là model thanh xuân : ví dụ: Đây là câu trả lời? Vâng。"
    # Các dấu ngoặc 「」, ( ), và dấu phẩy Latin ',' đã bị xóa.
    # Các dấu ':', '?', '。' được giữ lại.
