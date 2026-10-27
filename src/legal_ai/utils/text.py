import re
import hashlib
import unicodedata
def normalize_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return '\n'.join(line.strip() for line in text.split('\n')).strip()

def strip_control_chars(text: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def normalize_quotes(text: str) -> str:
    text = re.sub(r'[""„"]', '"', text)
    text = re.sub(r"[''‚']", "'", text)
    text = re.sub(r'[–—]', '-', text)
    return text

def clean_vietnamese_punct(text: str) -> str:
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'\s+([,;:\.!\?])', r'\1', text)
    return text

def clean(text: str) -> str:
    text = normalize_unicode(text)
    text = strip_control_chars(text)
    text = normalize_quotes(text)
    text = clean_vietnamese_punct(text)
    text = normalize_whitespace(text)
    return text

def truncate(text: str, max_chars: int = 800, suffix: str = "...") -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars - len(suffix)]
    last_space = cut.rfind(' ')
    if last_space > max_chars // 2:
        cut = cut[:last_space]
    return cut + suffix

def truncate_context(chunks: list[dict], max_total_chars: int = 6000) -> list[dict]:
    result = []
    total  = 0
    for chunk in chunks:
        chunk_len = len(chunk["text"])
        if total + chunk_len > max_total_chars:
            break
        result.append(chunk)
        total += chunk_len
    return result

def safe_sentence_split(text: str) -> list[str]:
    text = re.sub(r'(Điều\s+\d+[a-z]?)\.', r'\1<DOT>', text)
    text = re.sub(r'(Khoản\s+\d+)\.', r'\1<DOT>', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    return [s.strip() for s in sentences if s.strip()]

def hash_text(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def make_chunk_id(law_id: str, article_id: str, suffix: str = "") -> str:
    parts = [law_id, article_id]
    if suffix:
        parts.append(suffix)
    return "|".join(parts)

def extract_article_refs(text: str) -> list[str]:
    return re.findall(r'Điều\s+\d+[a-z]?', text)

def has_article_citation(text: str) -> bool:
    return bool(re.search(r'Điều\s+\d+', text))


# Tiêu đề Điều hợp lệ: BẮT BUỘC đầu dòng (cho phép 1 vài ký tự rác OCR đứng
# trước, vi du '| ' hay U+201E trong ban OCR 400dpi). Sau số điều phải là dấu
# chấm/hai chấm/phẩy, hết dòng, hoặc một đoạn tiêu đề KHÔNG phải cụm tham
# chiếu ("... tại Điều 17 của Luật này") — đó là trích dẫn giữa câu.
# Dùng chung cho parsers.split_by và metadata.extract_structure để hai chỗ
# không bao giờ lệch định nghĩa nhau nữa.
_REF_FOLLOWERS = (
    r'của\s+(?:Luật|Nghị|Thông|Bộ\s+luật|Quyết\s+định)\b'
    r'|Luật\b|Nghị\b|Thông\b'
    r'|khoản\b|điều\b|Điều\b|Mục\b|Chương\b|nay\b|sau\b'
)
# Ký tự rác OCR hay đứng trước heading. str.strip() không bỏ được vì chúng
# không phải whitespace. Thu hẹp cố ý: ban dau mo rộng them 'l' va '.' đã
# đánh thức lại 4 heading giả (Điều 1ê;, Điều 51, 52 và 53..., ...), nên
# chỉ giữ các ký tự đã QUAN SÁT được trong corpus OCR thật.
_LEADING_JUNK = r'[ \t]*[|¦‖„“]*[ \t]*'
LEADING_JUNK = _LEADING_JUNK
# Dấu kết thúc số điều: chấm, hai chấm, phẩy (OCR hay đọc '89.' thành '89,').
_STOP = r'[.:,]'
# Heading GIẢ do OCR gần ra: dấu câu bất thường, hoặc số điều nối tiếp cụm
# liệt kê/tham chiếu. Quan sát thực tế: 'Điều 1ê;', 'Điều 88 của Luật nảy.',
# 'Điều 51, 52 và 53 của Luật này.', 'Điều 94 của Luật này. :'
_BOGUS = r'(?:,[ \t]*\d|[a-zà-ỹ]{1,3};|của\s+Luật\s+\S{0,12}\.)'
ARTICLE_HEADING = (
    r'Điều\s+\d+[a-z]?[ \t]*'
    # Chặn cross-ref TRƯỚC, rồi mới tới dấu ngắt. Nếu để dấu ngắt trước thì
    # backtracking của \s+ tìm được đường khác và lookahead không còn tác dụng.
    r'(?!(?:' + _REF_FOLLOWERS + r'|' + _BOGUS + r'))'
    r'(?:' + _STOP + r'|$|[A-ZÀ-ỸÂÊÔĐƠƯ][^\n]{0,80})'
)

# Chỉ lấy cụm "Điều N" khi nó thực sự là heading (dùng cho structure_map).
# Chấp nhận "Điều 47", "Điều 47.", "Điều 47:", "Điều 47 Tên điều";
# LOẠI "Điều 17 của Luật này" (trích dẫn giữa dòng, không phải heading).
# group(1) = article_id. Cùng định nghĩa heading với parsers.split_by.
HEADING_FOLLOWERS_RE = re.compile(
    _LEADING_JUNK + r'(Điều\s+\d+[a-z]?)[ \t]*'
    r'(?!(?:' + _REF_FOLLOWERS + r'|' + _BOGUS + r'))'
    r'(?:' + _STOP + r'|$|[A-ZÀ-ỸÂÊÔĐƠƯ][^\n]{0,80})'
)