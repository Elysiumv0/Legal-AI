"""Ghep OCR dpi=200 (main) va dpi=400 (recovered) thanh mot van ban co PAGE marker.

Ban main khong co page marker (lỗi cu: parse_pdf_ocr ghep bang '\\n'), nen ta
khong the biet Điều nào o trang nào. Ban recovered CO marker.

Cach ghep an toàn: chi lay tu ban recovered nhung Điều mà main KHONG co
(40, 72, va bat ky Điều nào con thieu), roi ghep vao theo đung vi tri sắp xếp
theo số Điều. Khong ghep ca trang → tranh duplicate Điều da co o main.

Chay: python -I scripts/merge_ocr_pages.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from legal_ai.ingestion.normalizers import normalize
from legal_ai.ingestion.parsers import split_by

MAIN = ROOT / "data" / "processed" / "ocr_sample.txt"
REC = ROOT / "data" / "processed" / "ocr_recovered.txt"
OUT = ROOT / "data" / "processed" / "corpus_final.json"

main_text = MAIN.read_text(encoding="utf-8")
rec_text = REC.read_text(encoding="utf-8")

main_arts = split_by(normalize(main_text))
have = {a["article_id"] for a in main_arts}
print(f"main: {len(main_arts)} articles, thiếu: ", end="")

# Tìm số thiếu từ main
nums = sorted({int(a["article_id"].split()[1]) for a in main_arts})
missing = [n for n in range(nums[0], nums[-1] + 1) if n not in set(nums)]
print(missing)

# Điều từ recovered
rec_pages = {}
parts = re.split(r"\n=== PAGE (\d+) \(dpi=400\) ===\n", rec_text)
for i in range(1, len(parts), 2):
    rec_pages[int(parts[i])] = parts[i + 1]

added = {}
for p, text in sorted(rec_pages.items()):
    for a in split_by(normalize(text)):
        if a["article_id"] not in have:
            added[a["article_id"]] = a
            have.add(a["article_id"])

print(f"recovered bổ sung: {sorted(added, key=lambda s: int(s.split()[1]))}")

# Sắp xếp tất cả theo số Điều
def key(a):
    n, suffix = a["article_id"].split()[1], ""
    m = re.match(r"(\d+)([a-z]*)", n)
    return (int(m.group(1)), m.group(2))

final = sorted(main_arts + list(added.values()), key=key)
fnums = [int(a["article_id"].split()[1]) for a in final]
print(f"FINAL: {len(final)} articles | range {min(fnums)}-{max(fnums)}")
still = [n for n in range(min(fnums), max(fnums) + 1) if n not in set(fnums)]
print(f"vẫn thiếu: {still}")

# Ghi corpus.json
articles = []
for a in final:
    articles.append(
        {
            "article_id": a["article_id"],
            "header": a["header"],
            "content": a["content"],
            "law_id": "59/2020/QH14",
            "law_name": "Luật Doanh nghiệp 2020",
            "full_id": f"59/2020/QH14|Luật Doanh nghiệp 2020|{a['article_id']}",
        }
    )
corpus = [
    {
        "law_id": "59/2020/QH14",
        "law_name": "Luật Doanh nghiệp 2020",
        "content": main_text,
        "total_articles": len(articles),
        "articles": articles,
    }
]
OUT.parent.mkdir(parents=True, exist_ok=True)
import json

OUT.write_text(
    json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"wrote {OUT}")
