"""Tim entry homoglyph / trung NFKC trong configs/data_sources.yaml."""
import re
import sys
import unicodedata
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

t = Path("configs/data_sources.yaml").read_text(encoding="utf-8")
ids = re.findall(r"^    law_id: (.+)$", t, re.MULTILINE)
print("tổng entries:", len(ids))

ALLOWED = set("ĐđƯưƠơÂâÊêÔô")
weird = []
for x in ids:
    bad = [
        c for c in x
        if ord(c) > 127
        and unicodedata.category(c) != "Mn"
        and c not in ALLOWED
    ]
    if bad:
        weird.append((x, bad))

print("entry có ký tự đáng nghi:", len(weird))
for x, bad in weird[:15]:
    print(" ", repr(x), "->", [unicodedata.name(c, hex(ord(c))) for c in bad])

norm = {}
for x in ids:
    n = unicodedata.normalize("NFKC", x)
    norm.setdefault(n, []).append(x)
dups = {k: v for k, v in norm.items() if len(v) > 1}
print()
print("cặp trùng sau NFKC:", len(dups))
for k, v in list(dups.items())[:15]:
    print(" ", v)
