"""Đo MỘT điểm còn thiếu: retrieve=20, rerank=3.

Bối cảnh: sweep 6 config quá chậm (~45 phút/config trên CPU) nên đã kill sau
khi có hàng (10,3) = F2 0.723. Chỉ còn 1 câu hỏi thực sự mở: tăng retrieve
lên 20 có cứu được câu id 15 (ung vien Điều 34 chưa từng vào top-10) không,
và có làm F2 tệ hơn không.

Chay: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -u scripts/test_retrieve20.py
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_ai.retrieval.sparse.bm25 import BM25Retriever  # noqa: E402
from legal_ai.models.reranker import Reranker  # noqa: E402
from legal_ai.retrieval.config import RERANK_BATCH_SIZE  # noqa: E402

RETRIEVE_K = 20
RERANK_K = 3


def unique_articles(results):
    seen, out = set(), []
    for r in results:
        if r["article_id"] not in seen:
            seen.add(r["article_id"])
            out.append(r["article_id"])
    return out


def prf(retrieved, expected):
    exp = set(expected)
    if not retrieved:
        return 0.0, 0.0, 0.0
    hits = len(exp & set(retrieved))
    p = hits / len(retrieved)
    r = hits / len(exp) if exp else 1.0
    d = 4 * p + r
    return p, r, (5 * p * r) / d if d else 0.0


def main():
    questions = json.loads((ROOT / "data/processed/eval_set.json").read_text(encoding="utf-8"))
    chunks = json.loads((ROOT / "data/processed/chunks.json").read_text(encoding="utf-8"))
    bm = BM25Retriever()
    bm.build(chunks)

    print("loading reranker...")
    rr = Reranker()
    print(f"reranker: device={rr.device}")

    P, R, F = [], [], []
    for i, q in enumerate(questions, 1):
        cand = bm.retrieve(q["question"], top_k=RETRIEVE_K)
        reranked = rr.rerank(
            q["question"], cand, top_k=RERANK_K, batch_size=RERANK_BATCH_SIZE
        )
        top = unique_articles(reranked)[:RERANK_K]
        p, r, f2 = prf(top, q["expected_articles"])
        P.append(p)
        R.append(r)
        F.append(f2)
        if i % 12 == 0:
            print(f"  {i}/{len(questions)}")

    n = len(questions)
    p, r, f2 = sum(P) / n, sum(R) / n, sum(F) / n

    print()
    print("=" * 58)
    print(f"{'':22} {'P':>8} {'R':>8} {'F2':>8}")
    print("-" * 58)
    print(f"{'retrieve=10 rerank=3':22} {'0.500':>8} {'0.858':>8} {'0.723':>8}")
    print(f"{'retrieve=20 rerank=3':22} {p:>8.3f} {r:>8.3f} {f2:>8.3f}")
    print("=" * 58)

    q15 = next(q for q in questions if q["id"] == 15)
    cand = bm.retrieve(q15["question"], top_k=RETRIEVE_K)
    rr15 = unique_articles(
        rr.rerank(q15["question"], cand, top_k=RERANK_K, batch_size=RERANK_BATCH_SIZE)
    )
    print()
    print(f"cau id 15 (expected Đieu 34): top3 = {rr15[:3]}")
    print("  =>", "retrieve=20 CỨU ĐƯỢC" if "Điều 34" in rr15[:3] else "vẫn không cứu được")


if __name__ == "__main__":
    main()
