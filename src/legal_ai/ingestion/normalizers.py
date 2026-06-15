import re
import unicodedata
def fix_encoding(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()

def normalize_legal_terms(text: str) -> str:
    text = re.sub(r'[Đđ]iều\s*\.?\s*(\d+[a-z]?)', r'Điều \1', text)
    text = re.sub(r'[Kk]hoản\s*\.?\s*(\d+)', r'Khoản \1', text)
    text = re.sub(r'[Đđ]iểm\s*([a-z])\b', r'Điểm \1', text)
    text = re.sub(r'[Cc]hương\s+([IVXLCDM]+|\d+)', r'Chương \1', text)
    text = re.sub(r'[Mm]ục\s+(\d+|[IVXLCDM]+)', r'Mục \1', text)
    return text

def remove_page_artifacts(text: str) -> str:
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r'\d{1,3}', stripped):
            continue
        if re.search(r'ThuVienPhapLuat|thuvienphapluat|028-\d+|\+84|\bTrang\b', stripped, re.IGNORECASE):
            continue
        if re.fullmatch(r'[-_.=*]{5,}', stripped):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def fix_ocr_artifacts(text: str) -> str:
    text = re.sub(r'\b1uật\b', 'luật', text)
    text = re.sub(r'\bLuâ1\b', 'Luật', text)
    text = re.sub(r'\bdoanh nghiêp\b', 'doanh nghiệp', text)
    text = re.sub(r'([Đđ])i ều', r'\1iều', text)
    text = re.sub(r'kho ản', 'khoản', text)
    return text

def normalize(text: str) -> str:
    text = fix_encoding(text)
    text = fix_ocr_artifacts(text)
    text = remove_page_artifacts(text)
    text = normalize_legal_terms(text)
    text = normalize_whitespace(text)
    return text