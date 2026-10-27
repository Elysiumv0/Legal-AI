"""Đo baseline BM25: macro Precision / Recall / F2 tren bo cau hoi eval.

F2 = (5 x P x R) / (4 x P + R)  -- recall trong hon precision gap đoi.

Chay: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/run_bm25_baseline.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_ai.retrieval.sparse.bm25 import BM25Retriever  # noqa: E402
from legal_ai.evaluation.metrics import (  # noqa: E402
    compute_retrieval_metrics,
    macro_f2,
)

EVAL = ROOT / "data/processed/eval_set.json"
if not EVAL.exists():
    EVAL = ROOT / "data/processed/eval_raw.json"

TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def load_eval():
    raw = EVAL.read_text(encoding="utf-8")
    d = json.loads(raw)
    return d["questions"] if isinstance(d, dict) else d


def main():
    questions = load_eval()
    chunks = json.loads(
        (ROOT / "data/processed/chunks.json").read_text(encoding="utf-8")
    )
    print(f"eval: {len(questions)} cau | chunks: {len(chunks)} | top_k={TOP_K}")

    bm = BM25Retriever()
    bm.build(chunks)

    per_difficulty = defaultdict(list)
    rows = []
    misses = []

    for q in questions:
        # Top-K article_id tu BM25, khong trung
        results = bm.retrieve(q["question"], top_k=TOP_K)
        retrieved = []
        seen = set()
        for r in results:
            art = r["article_id"]
            if art not in seen:
                seen.add(art)
                retrieved.append(art)

        m = compute_retrieval_metrics(retrieved, q["expected_articles"])
        per_difficulty[q["difficulty"]].append(m)
        rows.append((q, m, retrieved))

        if m.f2 < 1.0:
            misses.append(
                {
                    "id": q["id"],
                    "difficulty": q["difficulty"],
                    "question": q["question"],
                    "expected": q["expected_articles"],
                    "retrieved_top": retrieved[:TOP_K],
                    "p": round(m.precision, 3),
                    "r": round(m.recall, 3),
                    "f2": round(m.f2, 3),
                    "hit_at_10": any(
                        a in set(q["expected_articles"]) for a in retrieved
                    ),
                }
            )

    print()
    print("=" * 64)
    print(f"MACRO-F2 @ top_{TOP_K}: {macro_f2([m for _, m, _ in rows]):.4f}")
    print("-" * 64)
    print(f"{'difficulty':10} {'n':>4} {'P':>7} {'R':>7} {'F2':>7}")
    for diff in ("easy", "medium", "hard"):
        ms = per_difficulty.get(diff, [])
        if not ms:
            continue
        n = len(ms)
        p = sum(x.precision for x in ms) / n
        r = sum(x.recall for x in ms) / n
        f2 = sum(x.f2 for x in ms) / n
        print(f"{diff:10} {n:>4} {p:>7.3f} {r:>7.3f} {f2:>7.3f}")
    print("-" * 64)

    hit10 = sum(1 for x in misses if x["hit_at_10"])
    zero = sum(1 for x in misses if x["r"] == 0.0)
    print(f"cau co it nhat 1 expected trong top-{TOP_K}: "
          f"{len(questions) - zero}/{len(questions)}")
    print(f"cau F2 < 1.0 : {len(misses)}/{len(questions)}")
    print(f"  trong do recall = 0 (khong lay duoc cai nao): {zero}")
    print(f"  con lai lay duoc 1 phan: {len(misses) - zero}")
    print()

    out = ROOT / "data/processed/baseline_misses.json"
    out.write_text(
        json.dumps(misses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"chi tiet cau chua dat: {out}")

    # In 5 cau te nhat de xem huong
    print()
    print("5 cau F2 thap nhat:")
    for x in sorted(misses, key=lambda y: y["f2"])[:5]:
        print(f"  id {x['id']} [{x['difficulty']}] F2={x['f2']:.2f}")
        print(f"     Q: {x['question'][:100]}")
        print(f"     expected: {x['expected']}")
        print(f"     got     : {x['retrieved_top'][:5]}")


if __name__ == "__main__":
    main()
