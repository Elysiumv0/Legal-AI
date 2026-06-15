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