import re
import json
from pathlib import Path
import pdfplumber
from legal_ai.ingestion.normalizers import normalize
from legal_ai.utils.text import has_article_citation
from legal_ai.config.loader import load_data_sources

def parse_pdf_ocr(path: str) -> str:
    from pdf2image import convert_from_path
    import pytesseract
    pages = convert_from_path(path, dpi=200)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="vie") + "\n"
    return text

def split_by(text: str) -> list[dict]:
    pattern = r'(Điều\s+\d+[a-z]?[\.\:].*?)(?=Điều\s+\d+[a-z]?[\.\:]|\Z)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    articles = []
    for match in matches:
        lines = match.strip().split('\n')
        header = lines[0].strip()
        num = re.search(r'Điều\s+(\d+[a-z]?)', header, re.IGNORECASE)
        if not num:
            continue
        articles.append({
            "article_id": f"Điều {num.group(1)}",
            "header": header,
            "content": match.strip()
        })
    return articles

def parse_pdf(path: str) -> str:
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
    """
    Load file mappings từ data_sources.yaml và parse tất cả văn bản pháp luật.
    Không cần sửa code — chỉ cần edit file YAML.
    """
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