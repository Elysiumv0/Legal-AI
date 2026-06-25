import re
from dataclasses import dataclass, field

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
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        # 1. Match Chapter
        ch = re.match(r'^(Chương\s+[IVXLCDM\d]+)[\.:]?\s*(.*)', line, re.IGNORECASE)
        if ch:
            current_chapter = ch.group(1).strip()
            name_part = ch.group(2).strip()
            
            lookahead_lines = []
            if name_part:
                lookahead_lines.append(name_part)
                
            j = i + 1
            while j < n:
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue
                # Stop if next line is a structural boundary or clause
                if (re.match(r'^(Chương|Mục|Điều)\s+', next_line, re.IGNORECASE) or 
                    re.match(r'^\d+\.\s+', next_line)):
                    break
                # Stop if next line is too long (titles are usually short, < 150 chars)
                if len(next_line) > 150:
                    break
                lookahead_lines.append(next_line)
                j += 1
            
            current_chapter_name = " ".join(lookahead_lines).strip() or None
            current_section = None
            current_section_name = None
            i = j  # Fast forward to where lookahead stopped
            continue
            
        # 2. Match Section
        sec = re.match(r'^(Mục\s+\d+)[\.:]?\s*(.*)', line, re.IGNORECASE)
        if sec:
            current_section = sec.group(1).strip()
            name_part = sec.group(2).strip()
            
            lookahead_lines = []
            if name_part:
                lookahead_lines.append(name_part)
                
            j = i + 1
            while j < n:
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue
                if (re.match(r'^(Chương|Mục|Điều)\s+', next_line, re.IGNORECASE) or 
                    re.match(r'^\d+\.\s+', next_line)):
                    break
                if len(next_line) > 150:
                    break
                lookahead_lines.append(next_line)
                j += 1
                
            current_section_name = " ".join(lookahead_lines).strip() or None
            i = j
            continue
            
        # 3. Match Article
        art = re.match(r'^(Điều\s+\d+[a-z]?)[\.\:]', line, re.IGNORECASE)
        if art:
            article_id = art.group(1).strip()
            result[article_id] = ArticleMetadata(
                article_id=article_id,
                chapter=current_chapter,
                chapter_name=current_chapter_name,
                section=current_section,
                section_name=current_section_name,
            )
            
        i += 1
        
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