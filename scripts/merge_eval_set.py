"""Gop 12 cau da viet lai vao bo 48 cau, ghi ra eval_set.json cuoi cung."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

raw = json.loads((ROOT / "data/processed/eval_raw.json").read_text(encoding="utf-8"))
fixed = json.loads(
    (
        ROOT
        / "C:/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/"
        "0fae9f7c-10aa-4328-8c07-ed527fa8a73c/tasks/wjxkshmdz.output"
    ).read_text(encoding="utf-8").split('"result": ', 1)[1].rsplit('},\n  "workflowProgress"', 1)[0].rstrip() + "\n}"
) if False else None

# Cach don gian: doc file output va trich JSON result
text = (
    ROOT
    / "C:/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/"
    "0fae9f7c-10aa-4328-8c07-ed527fa8a73c/tasks/wjxkshmdz.output"
).read_text(encoding="utf-8")
i = text.find('"result": ')
body = text[i + len('"result": ') :]
depth, instr, esc, end = 0, False, False, None
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
            end = idx + 1
            break
fixed = json.loads(body[:end])

by_id = {q["id"]: q for q in fixed["questions"]}
merged = []
for q in raw["questions"]:
    if q["id"] in by_id:
        merged.append(by_id[q["id"]])
    else:
        merged.append(q)
merged.sort(key=lambda q: q["id"])

DIACRITIC = re.compile(r"[ăâêôơưđàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ]")
nodiac = [q for q in merged if not DIACRITIC.search(q["question"])]
leak = [
    q
    for q in merged
    if re.search(r"[Đđ]iề\s*u\s+\d", q["question"], re.IGNORECASE)
]

out = ROOT / "data/processed/eval_set.json"
out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"gop xong: {len(merged)} cau -> {out.name}")
print(f"  khong con cau thieu dau: {len(nodiac)} (truoc khi sua: 12)")
print(f"  leak so Dieu: {len(leak)}")
print(f"  difficulty: easy={sum(1 for q in merged if q['difficulty']=='easy')} "
      f"medium={sum(1 for q in merged if q['difficulty']=='medium')} "
      f"hard={sum(1 for q in merged if q['difficulty']=='hard')}")
print(f"  do dai trung binh: {sum(len(q['question']) for q in merged)//len(merged)} ky tu")
