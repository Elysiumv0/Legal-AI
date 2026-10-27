import re
import json
import os
import shutil
from pathlib import Path
from legal_ai.ingestion.normalizers import normalize
from legal_ai.utils.text import (
    ARTICLE_HEADING,
    HEADING_FOLLOWERS_RE,
    LEADING_JUNK,
    has_article_citation,
)


def _resolve_tesseract():
    """Tìm tesseract.exe + tessdata (vie) dù terminal chưa nạp PATH mới.

    Thứ tự ưu tiên: TESSERACT_CMD env < TESSDATA_PREFIX env < quét đường dẫn
    cài đặt quen thuộc trên Windows. Chỉ cần làm 1 lần rồi cấu hình cho
    pytesseract, nên gọi lại nhiều lần không tốn kém.
    """
    import pytesseract

    exe = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not exe:
        for cand in (
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        ):
            if cand and cand.exists():
                exe = str(cand)
                break
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe

    if not os.environ.get("TESSDATA_PREFIX"):
        for cand in (
            Path(os.environ.get("LOCALAPPDATA", "")) / "tessdata",
            Path(exe).parent / "tessdata" if exe else None,
        ):
            if cand and cand.exists():
                os.environ["TESSDATA_PREFIX"] = str(cand)
                break
    return exe


def _resolve_poppler() -> str | None:
    """Tìm thư mục bin của poppler (chứa pdfinfo/pdftoppm).

    Cùng lý do với _resolve_tesseract: terminal đang mở chưa nạp PATH mới.
    """
    exe = shutil.which("pdfinfo") or shutil.which("pdftoppm")
    if exe:
        return str(Path(exe).parent)
    for cand in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "poppler",
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "poppler",
        Path("C:/poppler"),
    ):
        if cand.exists():
            hits = sorted(cand.rglob("pdfinfo.exe"))
            if hits:
                return str(hits[0].parent)
    return None


def parse_pdf_ocr(path: str, dpi: int = 200) -> str:
    from pdf2image import convert_from_path
    import pytesseract

    _resolve_tesseract()
    poppler_path = _resolve_poppler()
    pages = convert_from_path(path, dpi=dpi, poppler_path=poppler_path)
    # Giu ranh gioi trang = \f. Khong co no, khi OCR miss 1 Dieu thi khong
    # the biet Dieu do nam o trang nao de re-OCR — gap=2 trong danh sach
    # heading la dau hieu duy nhat, va no khong cho biet vi tri cu the.
    text = ""
    for i, page in enumerate(pages, 1):
        if i > 1:
            text += "\f"
        text += pytesseract.image_to_string(page, lang="vie") + "\n"
    return text

_SPLIT_RE = re.compile(
    '^' + LEADING_JUNK + '(' + ARTICLE_HEADING + r'.*?)(?=^' + LEADING_JUNK + ARTICLE_HEADING + r'|\Z)',
    re.DOTALL | re.MULTILINE,
)


def split_by(text: str) -> list[dict]:
    matches = _SPLIT_RE.findall(text)
    articles = []
    for match in matches:
        lines = match.strip().split('\n')
        header = lines[0].strip()
        # Cung mot dinh nghia heading voi metadata.extract_structure:
        # regex nay vua nhan "Dieu 48 Ten dieu", vua loai "Dieu 17 cua Luat nay".
        num = HEADING_FOLLOWERS_RE.match(header)
        if not num:
            continue
        articles.append({
            "article_id": num.group(1).strip(),
            "header": header,
            "content": match.strip()
        })
    return articles

def _read_as_text(path: str) -> str | None:
    """Phát hiện file .pdf/.txt thực chất là VĂN BẢN (HTML "Save as" từ
    thuvienphapluat hay text qua clipboard). PDF thật bắt đầu b'%PDF';
    không phải thì đây là text — thường UTF-16LE kèm BOM (tên file còn
    hiện 'ÿþ' = BOM bị lỗi font).
    Trả None nếu đúng là PDF.
    """
    with open(path, "rb") as f:
        if f.read(8).startswith(b"%PDF"):
            return None
    raw = Path(path).read_bytes()
    for enc in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_pdf(path: str) -> str:
    # text-đóng-đuôi-pdf đọc trực tiếp; pdfplumber chỉ chạy khi đúng là PDF
    text_file = _read_as_text(path)
    if text_file is not None:
        return text_file

    import pdfplumber
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    dieu_count = len(re.findall(r'Điều\s+\d+', text, re.IGNORECASE))
    if dieu_count < 2 or len(text) < 500:
        text = parse_pdf_ocr(path)
    return text

def parse_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def parse_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_legal_file(
    file_path: str,
    law_id: str,
    law_name: str
) -> dict:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw_text = parse_pdf(file_path)
    elif suffix == ".docx":
        raw_text = parse_docx(file_path)
    elif suffix == ".txt":
        raw_text = parse_txt(file_path)
    elif suffix == ".doc":
        raise ValueError(f"Không hỗ trợ định dạng legacy .doc: {file_path}")
    else:
        raise ValueError(f"Không hỗ trợ định dạng: {suffix}")
    raw_text = normalize(raw_text)
    articles = split_by(raw_text)
    for art in articles:
        art["law_id"] = law_id
        art["law_name"] = law_name
        art["full_id"] = f"{law_id}|{law_name}|{art['article_id']}"
    return {
        "law_id": law_id,
        "law_name": law_name,
        "content": raw_text,
        "total_articles": len(articles),
        "articles": articles
    }

def parse_all(
    config_path: str = "configs/data_sources.yaml",
    output_path: str = "data/processed/corpus.json",
):
    from legal_ai.config.loader import load_data_sources

    try:
        data_sources = load_data_sources(config_path)
    except FileNotFoundError:
        print(f"⚠️  Config not found: {config_path}")
        print("   Tạo config mẫu tại configs/data_sources.yaml")
        _write_sample_config()
        return []

    if not data_sources.sources:
        print("⚠️  Không có source nào trong config. Thêm file vào configs/data_sources.yaml")
        return []

    corpus = []
    failed = []
    for source in data_sources.sources:
        if not Path(source.path).exists():
            print(f"Không tìm thấy: {source.path}")
            failed.append(source.path)
            continue
        try:
            result = parse_legal_file(source.path, source.law_id, source.law_name)
            corpus.append(result)
            print(f"{source.law_id}: {result['total_articles']} điều — {source.path}")
        except Exception as e:
            print(f"{source.law_id}: {e}")
            failed.append(source.path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    total_articles = sum(d["total_articles"] for d in corpus)
    print(f"\nCorpus: {total_articles} điều từ {len(corpus)} văn bản")
    if failed:
        print(f"Lỗi: {failed}")
    return corpus

def _write_sample_config():
    sample = """# Khai báo văn bản pháp luật cần parse
# Đặt file .pdf / .docx vào data/raw/ rồi khai báo path ở đây
sources:
  - path: data/raw/luat-doanh-nghiep-2020.docx
    law_id: 59/2020/QH14
    law_name: Luật Doanh nghiệp 2020
    document_type: law
    issued_date: "2020-06-17"
    effective_date: "2021-01-01"
    source_url: ""
    status: active
"""
    Path("configs/data_sources.yaml").parent.mkdir(parents=True, exist_ok=True)
    with open("configs/data_sources.yaml", "w", encoding="utf-8") as f:
        f.write(sample)
    print("   Đã tạo configs/data_sources.yaml — điền path file thật vào rồi chạy lại.")

if __name__ == "__main__":
    parse_all()
