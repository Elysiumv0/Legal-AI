"""Đo BM25 + reranker trên 48 câu eval, so voi BM25-only.

Pipeline thực tế: BM25 lấy TOP_K_RETRIEVE=10 ứng viên -> reranker xếp lại ->
cắt TOP_K_FINAL=3. Đo cả P/R/F2 của cả 2 chặng để thấy reranker đóng góp gì.

Chạy: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/run_bm25_rerank.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_ai.retrieval.sparse.bm25 import BM25Retriever  # noqa: E402
from legal_ai.models.reranker import Reranker  # noqa: E402
from legal_ai.evaluation.metrics import (  # noqa: E402
    compute_retrieval_metrics,
    macro_f2,
)
from legal_ai.retrieval.config import (  # noqa: E402
    RERANK_BATCH_SIZE,
    TOP_K_FINAL,
    TOP_K_RETRIEVE,
)

EVAL = ROOT / "data/processed/eval_set.json"


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
    questions = json.loads(EVAL.read_text(encoding="utf-8"))
    chunks = json.loads(
        (ROOT / "data/processed/chunks.json").read_text(encoding="utf-8")
    )

    print(f"eval: {len(questions)} cau | retrieve={TOP_K_RETRIEVE} -> rerank -> {TOP_K_FINAL}")
    bm = BM25Retriever()
    bm.build(chunks)

    print("loading reranker (lần đầu tải ~1GB)...")
    rr = Reranker()
    print(f"reranker ready: device={rr.device} dtype={next(rr.model.parameters()).dtype}")

    m_bm25, m_rerank = [], []
    improved = []
    hurt = []

    for i, q in enumerate(questions, 1):
        cand = bm.retrieve(q["question"], top_k=TOP_K_RETRIEVE)
        # BM25-only: cắt top 3 từ chính danh sách ứng viên
        bm25_top = unique_articles(cand)[:TOP_K_FINAL]
        # Rerank: reranker xếp lại toàn bộ ứng viên rồi cắt
        reranked = rr.rerank(q["question"], cand, top_k=TOP_K_FINAL,
                             batch_size=RERANK_BATCH_SIZE)
        rr_top = unique_articles(reranked)[:TOP_K_FINAL]

        p1, r1, f1 = prf(bm25_top, q["expected_articles"])
        p2, r2, f2 = prf(rr_top, q["expected_articles"])
        m_bm25.append((p1, r1, f1))
        m_rerank.append((p2, r2, f2))

        tag = ""
        if f2 > f1 + 1e-9:
            tag = "TỐT HƠN"
            improved.append((q["id"], f1, f2))
        elif f2 < f1 - 1e-9:
            tag = "TỆ HƠN"
            hurt.append((q["id"], f1, f2))

        if i % 12 == 0 or i == len(questions):
            print(f"  đã chạy {i}/{len(questions)}")
        if tag and f2 != f1:
            print(f"    id {q['id']}: {tag} F2 {f1:.2f} -> {f2:.2f}")

    def agg(rows):
        n = len(rows)
        return (
            sum(x[0] for x in rows) / n,
            sum(x[1] for x in rows) / n,
            sum(x[2] for x in rows) / n,
        )

    p_b, r_b, f_b = agg(m_bm25)
    p_r, r_r, f_r = agg(m_rerank)

    print()
    print("=" * 62)
    print(f"{'':22} {'P':>8} {'R':>8} {'F2':>8}")
    print("-" * 62)
    print(f"{'BM25-only (top 3)':22} {p_b:>8.3f} {r_b:>8.3f} {f_b:>8.3f}")
    print(f"{'BM25 + reranker':22} {p_r:>8.3f} {r_r:>8.3f} {f_r:>8.3f}")
    print("=" * 62)
    print(f"Δ F2 = {f_r - f_b:+.4f}  ({(f_r - f_b) / f_b * 100:+.1f}%)" if f_b else "")
    print(f"câu tốt hơn: {len(improved)} | câu tệ hơn: {len(hurt)} | không đổi: "
          f"{len(questions) - len(improved) - len(hurt)}")

    if improved:
        print("  câu cải thiện:", [f"id{i}: {a:.2f}->{b:.2f}" for i, a, b in improved[:8]])
    if hurt:
        print("  câu tệ hơn:", [f"id{i}: {a:.2f}->{b:.2f}" for i, a, b in hurt[:8]])

    # id 15 — câu miss nổi tiếng
    q15 = next(q for q in questions if q["id"] == 15)
    cand = bm.retrieve(q15["question"], top_k=TOP_K_RETRIEVE)
    rr_top = unique_articles(rr.rerank(q15["question"], cand, top_k=TOP_K_FINAL,
                                      batch_size=RERANK_BATCH_SIZE))
    print()
    print(f"câu id 15 (BM25 F2=0): expected={q15['expected_articles']}")
    print(f"  sau rerank: {rr_top[:TOP_K_FINAL]}")
    print("  =>", "RERANKER SỬA ĐƯỢC" if q15["expected_articles"][0] in rr_top else "vẫn chưa sửa được")


if __name__ == "__main__":
    main()
