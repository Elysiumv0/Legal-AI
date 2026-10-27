"""Audit bo cau hoi eval truoc khi dung.

Kiem tra 5 thu: leak so Dieu, cau hoi khong dau, cau hoi gop nhieu y,
expected_articles ngoai pham vi, expected rong.

Chay: python -I scripts/audit_eval_set.py
"""
import json
import re
from pathlib import Path

SRC = Path(
    "C:/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/"
    "0fae9f7c-10aa-4328-8c07-ed527fa8a73c/tasks/w3rvw345x.output"
)
OUT = Path("data/processed/eval_raw.json")


def extract_result(text: str) -> dict:
    i = text.find('"result": ')
    body = text[i + len('"result": ') :]
    depth = 0
    instr = False
    esc = False
    for idx, ch in enumerate(body):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            instr = not instr
            continue
        if instr:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(body[: idx + 1])
    raise ValueError("khong tim thay JSON result")


text = SRC.read_text(encoding="utf-8")
d = extract_result(text)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

print("total:", d["total"], "| byDifficulty:", d["byDifficulty"])
print()

DIACRITIC = re.compile(
    r"[ăâêôơưđàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ]"
)

# 1. Leak that su: cau hoi chua "Dieu N"
real_leak = [
    q
    for q in d["questions"]
    if re.search(r"[Đđ]iề\s*u\s+\d", q["question"], re.IGNORECASE)
]
print("1. LEAK so Dieu trong cau hoi:", len(real_leak))
for q in real_leak:
    print("   id", q["id"], ":", q["question"][:110])
print()

# 2. Câu hoi khong dau
nodiac = [q for q in d["questions"] if not DIACRITIC.search(q["question"])]
print("2. Câu hoi KHONG CO DAU:", len(nodiac), "/", d["total"])
for q in nodiac:
    print("   id", q["id"], ":", q["question"][:100])
print()

# 3. Cau hoi gop nhieu y
long_q = [
    q
    for q in d["questions"]
    if q["question"].count("?") > 1 or len(q["question"]) > 180
]
print("3. Cau hoi gop nhieu y (>180 chars hoac >1 dau ?):", len(long_q))
for q in long_q:
    print("   id", q["id"], f"({len(q['question'])} chars):", q["question"][:150])
print()

# 4. expected ngoai pham vi
valid = {f"Điều {i}" for i in range(1, 114)}
bad_exp = [
    q
    for q in d["questions"]
    if any(a not in valid for a in q["expected_articles"])
]
print("4. expected_articles ngoai Dai 1-113:", len(bad_exp))
for q in bad_exp:
    print("   id", q["id"], q["expected_articles"])
print()

# 5. expected rong
empty = [q for q in d["questions"] if not q["expected_articles"]]
print("5. expected_articles RONG:", len(empty))
for q in empty:
    print("   id", q["id"], ":", q["question"][:100])
print()

# 6. Cai nao expected nhieu nhat (hard nhat)
print("6. Top 5 cau co nhieu expected nhat:")
for q in sorted(
    d["questions"], key=lambda x: -len(x["expected_articles"])
)[:5]:
    print(
        f"   id {q['id']} ({len(q['expected_articles'])} dieu, {q['difficulty']}):",
        q["question"][:80],
    )
