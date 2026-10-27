import re
from dataclasses import dataclass, field
from legal_ai.utils.text import HEADING_FOLLOWERS_RE
@dataclass
class LawMetadata:
    law_id:       str
    law_name:     str
    law_type:     str       
    effective_date: str | None = None
    status:       str = "Còn hiệu lực"  

@dataclass
class ArticleMetadata:
    article_id:   str      
    chapter:      str | None = None   
    chapter_name: str | None = None  
    section:      str | None = None  
    section_name: str | None = None

def detect_law_type(law_name: str) -> str:
    name_lower = law_name.lower()
    if "bộ luật" in name_lower:
        return "Bộ luật"
    if "luật" in name_lower:
        return "Luật"
    if "nghị định" in name_lower:
        return "Nghị định"
    if "thông tư" in name_lower:
        return "Thông tư"
    if "quyết định" in name_lower:
        return "Quyết định"
    return "Văn bản pháp luật"

def extract_structure(full_text_or_articles) -> dict[str, ArticleMetadata]:
    result: dict[str, ArticleMetadata] = {}
    current_chapter      = None
    current_chapter_name = None
    current_section      = None
    current_section_name = None
    if isinstance(full_text_or_articles, list):
        raw_text = "\n".join(a.get("content", "") for a in full_text_or_articles)
    else:
        raw_text = full_text_or_articles
    lines = raw_text.split('\n')
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Không strip trước khi match ARTICLE_HEADING: regex cần thấy
        # "Điều 17 của Luật này" ở giữa dòng để loại, và "Điều 48 Tiêu chuẩn..."
        # ở đầu dòng để nhận — cả hai phải cùng một định nghĩa với parsers.split_by.
        # OCR co the mat dau ('Chuong II') hoac viet hoa thuong. Van ban
        # thuc te: OCR cua Luat DN 2020 mat 13/16 dong tieu de Chuong, nen
        # chapter se la None — do du lieu, khong phai loi code.
        ch = re.match(r'^[Cc][Hh][UuƯ][OoƠ][Nn][Gg]\s+([IVXLCDM\d]+)[.:]?\s*(.*)', stripped, re.IGNORECASE)
        if ch:
            current_chapter      = ch.group(1).strip()
            current_chapter_name = ch.group(2).strip() or None
            current_section      = None
            current_section_name = None
            continue

        sec = re.match(r'^(Mục\s+\d+)[\.:]?\s*(.*)', stripped)
        if sec:
            current_section      = sec.group(1).strip()
            current_section_name = sec.group(2).strip() or None
            continue

        # Chấp nhận "Điều 47", "Điều 47.", "Điều 48 Tiêu chuẩn ...";
        # loại "Điều 17 của Luật này" (trích dẫn giữa dòng).
        art = HEADING_FOLLOWERS_RE.match(line)
        if art:
            article_id = art.group(1).strip()
            result[article_id] = ArticleMetadata(
                article_id=article_id,
                chapter=current_chapter,
                chapter_name=current_chapter_name,
                section=current_section,
                section_name=current_section_name,
            )
    return result

def enrich_chunks(
    chunks: list[dict],
    structure_map: dict[str, ArticleMetadata],
    law_metadata: LawMetadata,
) -> list[dict]:
    for chunk in chunks:
        article_id = chunk.get("article_id", "")
        meta = structure_map.get(article_id, ArticleMetadata(article_id=article_id))
        chunk["law_type"]     = law_metadata.law_type
        chunk["status"]       = law_metadata.status
        chunk["chapter"]      = meta.chapter
        chunk["chapter_name"] = meta.chapter_name
        chunk["section"]      = meta.section
        chunk["section_name"] = meta.section_name
        parts = []
        if meta.chapter:
            parts.append(meta.chapter)
            if meta.chapter_name:
                parts.append(meta.chapter_name)
        parts.append(article_id)
        parts.append(law_metadata.law_name)
        chunk["context_prefix"] = " | ".join(parts)
    return chunks

def build_metadata(law_id: str, law_name: str) -> LawMetadata:
    return LawMetadata(
        law_id=law_id,
        law_name=law_name,
        law_type=detect_law_type(law_name),
    )