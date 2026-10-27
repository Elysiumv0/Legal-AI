"""Cat block dump vbpl_update khoi configs/data_sources.yaml.

Block dump (305 entry, tro den data/raw/vbpl_update/ khong ton tai, co 2 entry
homoglyph Cyrillic) nam tu dong 529 den het file. Ghi lai file chi voi phan
thuc te co y nghi a (33 entry ky tay o dau), roi validate.

Chay: .venv/Scripts/python.exe scripts/prune_data_sources.py
"""
import sys
import unicodedata
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "configs" / "data_sources.yaml"

CUT_AFTER = 527  # giu dong 1..527 (1-indexed); bo tu 528 tro di


def main():
    lines = YAML.read_text(encoding="utf-8").splitlines(keepends=True)
    print(f"trước khi cắt: {len(lines)} dòng")
    keep = lines[:CUT_AFTER]

    out = "".join(keep).rstrip() + "\n"
    YAML.write_text(out, encoding="utf-8")
    print(f"sau khi cắt:  {len(keep)} dòng")

    # Validate: doc lai tu chinh file vua ghi
    import yaml

    d = yaml.safe_load(out)
    srcs = d.get("sources", [])
    print(f"entries: {len(srcs)}")

    homoglyph = [
        s["law_id"]
        for s in srcs
        if any(ord(c) == 0x0421 for c in s["law_id"])  # CYRILLIC ES
    ]
    print(f"homoglyph Cyrillic còn lại: {homoglyph if homoglyph else 'khong'}")

    missing = [s["path"] for s in srcs if not (ROOT / s["path"]).exists()]
    print(f"path khong ton tai: {len(missing)}/{len(srcs)}")

    # Kiem tra từng entry con co homoglyph khac
    ALLOWED = set("ĐđƯưƠơÂâÊêÔô")
    suspicious = []
    for s in srcs:
        for field in ("law_id", "law_name", "path"):
            for c in s.get(field, ""):
                if (
                    ord(c) > 127
                    and unicodedata.category(c) != "Mn"
                    and c not in ALLOWED
                ):
                    suspicious.append((s["law_id"], field, c, hex(ord(c))))
    print(f"ky tu nghi ngoai tieng Viet: {len(suspicious)}")
    for x in suspicious[:5]:
        print("  ", x)

    print()
    print("==> OK" if not homoglyph and not suspicious else "==> CÒN VẤN ĐỀ")


if __name__ == "__main__":
    main()
