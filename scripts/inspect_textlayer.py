"""Dump text layer cua tung trang PDF + kiem chat luong.

Ly do: file scan cung co 8 font + 4 text-op o dau file (pdfplumber/tesseract
deu khong thay), tuc CO text layer kha nang. Neu text layer du tot va du
113 Dieu thi dung no thay cho OCR hoan toan - chinh xac hon, va khong con
mat 40/72/89/98.

Chay:  python -I scripts/inspect_textlayer.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "luat-doanh-nghiep-2020.pdf"
OUT = ROOT / "data" / "processed" / "textlayer.txt"

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Can PyMuPDF: python -m pip install pymupdf")

doc = fitz.open(PDF)
print(f"pages: {len(doc)}")

pages = []
for i, page in enumerate(doc, 1):
    pages.append(page.get_text() or "")

# 1. Text layer co du manh khong?
total_chars = sum(len(t) for t in pages)
empty_pages = [i for i, t in enumerate(pages, 1) if len(t.strip()) < 20]
print(f"textlayer chars: {total_chars}")
print(f"pages co text <20 chars (tuc van la anh scan): {len(empty_pages)}")
print(f"  chi tiet: {empty_pages[:20]}{'...' if len(empty_pages) > 20 else ''}")

# 2. Tim so Dieu trong text layer (chi heading dau dong co dau cham/hai cham)
full = "\n".join(pages)
heading_re = re.compile(r"^[ \t]*Điều\s+(\d+[a-z]?)[ \t]*[.:]", re.MULTILINE)
tl_ids = heading_re.findall(full)
tl_nums = sorted({int(n) for n in tl_ids if n.isdigit()})
print(f"\ntextlayer heading ids: {len(tl_ids)} (unique: {len(tl_nums)})")
if tl_nums:
    lo, hi = tl_nums[0], tl_nums[-1]
    missing = [n for n in range(lo, hi + 1) if n not in set(tl_nums)]
    print(f"range: {lo}..{hi} | missing: {missing}")

# 3. So sanh voi OCR
ocr_path = ROOT / "data" / "processed" / "ocr_sample.txt"
if ocr_path.exists():
    ocr = ocr_path.read_text(encoding="utf-8")
    print(f"\nOCR chars: {len(ocr)}")
    for n in (40, 72, 89, 98):
        # textlayer co heading Dieu n khong
        hit = re.search(rf"^[ \t]*Điều\s+{n}\b", full, re.MULTILINE)
        print(f"  Điều {n}: textlayer={'CO' if hit else 'KHONG'}")
    # Điều 4 bi cat trong OCR — textlayer co du khong
    m = re.search(r"Điều\s+4\b(.*?)(?=Điều\s+5\b)", full, re.DOTALL)
    if m:
        print(f"  Điều 4 trong textlayer: {len(m.group(1))} chars")
else:
    print("\n(khong co ocr_sample.txt de so sanh)")

if total_chars > 0:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = []
    for i, t in enumerate(pages, 1):
        body.append(f"\n=== PAGE {i} ===\n{t}")
    OUT.write_text("".join(body), encoding="utf-8")
    print(f"\nwrote {OUT}")
