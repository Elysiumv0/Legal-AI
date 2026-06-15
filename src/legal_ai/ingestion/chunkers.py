import re
import json
from pathlib import Path
from legal_ai.ingestion.metadata import build_metadata, extract_structure, enrich_chunks
MAX_CHARS = 800   
MIN_CHARS = 100  
def split(content: str) -> list[str]:
    pattern = r'(?=^\d+\.\s)'
    parts = re.split(pattern, content, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]

def merge(khoans: list[str], min_chars: int = MIN_CHARS) -> list[str]:
    merged = []
    buffer = ""
    for k in khoans:
        buffer = (buffer + "\n" + k).strip() if buffer else k
        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] += "\n" + buffer  
        else:
            merged.append(buffer)
    return merged

def chunk_article(article: dict, max_chars: int = MAX_CHARS) -> list[dict]:
    content = article["content"]
    header  = article["header"]
    full_id = article["full_id"]
    if len(content) <= max_chars:
        return [{
            "chunk_id": f"{full_id}|khoản_0",
            "text": content,
            "law_id": article["law_id"],
            "law_name": article["law_name"],
            "article_id": article["article_id"],
            "header": header,
            "full_id": full_id,
            "khoan": None,
        }]
    khoans = split(content)
    if len(khoans) <= 1:
        return sliding_window(article, max_chars)
    khoans = merge(khoans)
    chunks = []
    for i, khoan_text in enumerate(khoans):
        khoan_num = re.match(r'^(\d+)\.', khoan_text)
        khoan_label = f"khoản_{khoan_num.group(1)}" if khoan_num else f"khoản_{i}"
        text_with_context = f"{header}\n{khoan_text}"
        if len(text_with_context) > max_chars:
            sub_article = {**article, "content": text_with_context, "header": header}
            sub_chunks = sliding_window(sub_article, max_chars)
            for j, sc in enumerate(sub_chunks):
                sc["chunk_id"] = f"{full_id}|{khoan_label}_w{j}"
                sc["khoan"] = khoan_label
            chunks.extend(sub_chunks)
        else:
            chunks.append({
                "chunk_id": f"{full_id}|{khoan_label}",
                "text": text_with_context,
                "law_id": article["law_id"],
                "law_name": article["law_name"],
                "article_id": article["article_id"],
                "header": header,
                "full_id": full_id,
                "khoan": khoan_label,
            })
    return chunks

def sliding_window(article: dict, max_chars: int = MAX_CHARS, overlap: int = 100) -> list[dict]:
    content = article["content"]
    full_id = article["full_id"]
    chunks = []
    start = 0
    i = 0
    while start < len(content):
        end = start + max_chars
        if end < len(content):
            last_nl = content.rfind('\n', start, end)
            last_sp = content.rfind(' ', start, end)
            end = last_nl if last_nl > start + max_chars // 2 else (last_sp if last_sp > start + max_chars // 2 else end)
        chunk_text = content[start:end].strip()
        if chunk_text:
            chunks.append({
                "chunk_id": f"{full_id}|window_{i}",
                "text": chunk_text,
                "law_id": article["law_id"],
                "law_name": article["law_name"],
                "article_id": article["article_id"],
                "header": article["header"],
                "full_id": full_id,
                "khoan": None,
            })
        start = end - overlap
        i += 1
    return chunks

def is_noise(article: dict) -> bool:
    content = article["content"]
    noise_patterns = [
        r'Chữ ký',
        r'Họ và tên',
        r'ngày.*tháng.*năm',
        r'CHỦ TỊCH',
        r'Ghi chú',
        r'\.{5,}',    
        r'_{5,}',      
    ]
    noise_count = sum(1 for p in noise_patterns if re.search(p, content))
    if noise_count >= 2:
        return True
    body = content.replace(article["header"], "").strip()
    if len(body) < 20:
        return True
    return False

def chunk_corpus(
    corpus_path: str = "data/processed/corpus.json",
    output_path: str = "data/processed/chunks.json"
) -> list[dict]:
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    all_chunks = []
    noise_count = 0
    for doc in corpus:
        law_id   = doc["law_id"]
        law_name = doc["law_name"] 
        doc_chunks = []
        for article in doc["articles"]:
            if is_noise(article):
                noise_count += 1
                continue
            chunks = chunk_article(article)
            doc_chunks.extend(chunks)
        raw_text = "\n".join(a["content"] for a in doc["articles"])
        law_meta      = build_metadata(law_id, law_name)
        structure_map = extract_structure(raw_text)
        doc_chunks    = enrich_chunks(doc_chunks, structure_map, law_meta)
        all_chunks.extend(doc_chunks)
        print(f"{law_id}: {len(doc['articles'])} điều: {len(doc_chunks)} chunks")
    all_chunks = [c for c in all_chunks if len(c["text"].strip()) >= 50]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\n{len(all_chunks)} chunks")
    return all_chunks

if __name__ == "__main__":
    chunks = chunk_corpus()
    sizes = [len(c["text"]) for c in chunks]
    print(f"\nChunk size stats:")
    print(f"  Min:  {min(sizes)}")
    print(f"  Max:  {max(sizes)}")
    print(f"  Avg:  {sum(sizes)//len(sizes)}")
    print(f"  >800: {sum(1 for s in sizes if s > 800)}")