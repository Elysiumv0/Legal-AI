"""OCR lai cac trang nghi ngo voi chat luong cao hon, tim 4 Dieu bi mat.

Ban OCR cu: dpi=200 → mat Điều 40, 72, 89, 98 (gap=2 giua cac heading
39→41, 71→73, 88→90, 97→99).
Ban nay: dpi=400 + psm=6 (gia dinh 1 khoi van ban thong nhat), va luu
kem PAGE marker de biet chinh xac trang nao ra Điều nao.

Chay: python -I scripts/recover_missing_articles.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "luat-doanh-nghiep-2020.pdf"
OUT = ROOT / "data" / "processed" / "ocr_recovered.txt"

# Dải trang quanh 4 điểm nghi, +-2 trang cho an toàn
PAGE_RANGES = [(22, 28), (42, 48), (51, 57), (56, 63)]
DPI = 400


def resolve_poppler():
    import os
    import shutil

    exe = shutil.which("pdftoppm") or shutil.which("pdfinfo")
    if exe:
        return str(Path(exe).parent)
    from pathlib import Path as P

    for cand in (
        P(os.environ.get("LOCALAPPDATA", "")) / "poppler",
        P(os.environ.get("ProgramFiles", "C:/Program Files")) / "poppler",
        P("C:/poppler"),
    ):
        if cand.exists():
            hits = sorted(cand.rglob("pdftoppm.exe"))
            if hits:
                return str(hits[0].parent)
    return None


def resolve_tesseract():
    import os
    import shutil

    exe = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not exe:
        for cand in (
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        ):
            if cand.exists():
                exe = str(cand)
                break
    if not os.environ.get("TESSDATA_PREFIX"):
        cand = Path(os.environ.get("LOCALAPPDATA", "")) / "tessdata"
        if cand.exists():
            os.environ["TESSDATA_PREFIX"] = str(cand)
    return exe


def main():
    from pdf2image import convert_from_path
    import pytesseract

    pop = resolve_poppler()
    exe = resolve_tesseract()
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
    print(f"poppler: {pop}\ntesseract: {exe}\nTESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX')}")

    # gop cac dai trang trung lap
    pages = sorted({p for lo, hi in PAGE_RANGES for p in range(lo, hi + 1)})
    print(f"re-OCRing {len(pages)} pages at dpi={DPI}: {pages}")

    chunks = []
    for p in pages:
        imgs = convert_from_path(
            PDF, dpi=DPI, first_page=p, last_page=p, poppler_path=pop
        )
        txt = pytesseract.image_to_string(imgs[0], lang="vie", config="--psm 6")
        chunks.append(f"\n=== PAGE {p} (dpi={DPI}) ===\n{txt}")
        print(f"  page {p}: {len(txt.strip())} chars")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(chunks), encoding="utf-8")
    print(f"\nwrote {OUT}")

    # Bao cao: tim thay cac Dieu bi mat khong
    full = "".join(chunks)
    for n in (40, 72, 89, 98):
        m = re.search(rf"Điều\s+{n}\b", full)
        if m:
            start = m.start()
            snippet = re.sub(r"\s+", " ", full[start : start + 220])
            print(f"\nFOUND Điều {n}: {snippet}")
        else:
            print(f"\nSTILL MISSING Điều {n}")


if __name__ == "__main__":
    import os

    main()
