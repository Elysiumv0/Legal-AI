"""Phan tich precision theo top-k va tim dai do precision.

Precision = so Điều đúng / so Điều trả về. Có 3 cách tăng:
  1. Trả về ít kết quả hơn (giảm k) — đánh đổi recall
  2. Xếp hạng tốt hơn (reranker) — không đánh đổi
  3. Truy vấn khớp đúng hơn (trọng số tiêu đề, hybrid)

Script này đo (1) trước, vì nó miễn phí và cho biết trần precision.

Chay: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/analyze_precision.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_ai.retrieval.sparse.bm25 import BM25Retriever  # noqa: E402

EVAL = ROOT / "data/processed/eval_set.json"


def prf(retrieved, expected):
    """Precision / Recall / F2 cho 1 câu. retrieved = list article_id (top-k)."""
    exp = set(expected)
    if not retrieved:
        return 0.0, 0.0, 0.0
    hits = len(exp & set(retrieved))
    p = hits / len(retrieved)
    r = hits / len(exp) if exp else 1.0
    denom = 4 * p + r
    f2 = (5 * p * r) / denom if denom else 0.0
    return p, r, f2


def main():
    questions = json.loads(EVAL.read_text(encoding="utf-8"))
    chunks = json.loads(
        (ROOT / "data/processed/chunks.json").read_text(encoding="utf-8")
    )
    bm = BM25Retriever()
    bm.build(chunks)

    # Lấy top-20 cho mỗi câu, rồi cắt theo từng k
    KS = [1, 2, 3, 5, 7, 10, 15, 20]
    per_k = {k: [] for k in KS}

    for q in questions:
        results = bm.retrieve(q["question"], top_k=20)
        retrieved = []
        seen = set()
        for r in results:
            if r["article_id"] not in seen:
                seen.add(r["article_id"])
                retrieved.append(r["article_id"])
        for k in KS:
            p, r, f2 = prf(retrieved[:k], q["expected_articles"])
            per_k[k].append((p, r, f2))

    print("=" * 68)
    print(f"{'k':>3} {'P':>7} {'R':>7} {'F2':>7}   {'ghi chú':}")
    print("-" * 68)
    rows = []
    for k in KS:
        n = len(per_k[k])
        p = sum(x[0] for x in per_k[k]) / n
        r = sum(x[1] for x in per_k[k]) / n
        f2 = sum(x[2] for x in per_k[k]) / n
        rows.append((k, p, r, f2))
        note = ""
        if k == 1:
            note = "chỉ lấy đúng 1 Điều"
        elif k == 5:
            note = "đề xuất hiện tại của pipeline"
        elif k == 10:
            note = "baseline đã đo"
        print(f"{k:>3} {p:>7.3f} {r:>7.3f} {f2:>7.3f}   {note}")
    print("=" * 68)

    best_f2 = max(rows, key=lambda x: x[3])
    print(f"F2 cao nhất ở k={best_f2[0]} ({best_f2[3]:.3f})")

    # Điểm then chốt: k=1 có P bao nhiêu?
    p1 = rows[0][1]
    p3 = rows[2][1]
    print()
    print(f"P@1  = {p1:.3f}  -> Điều đúng xếp hạng 1 ở {p1*100:.0f}% câu hỏi")
    print(f"P@3  = {p3:.3f}")
    print()
    print("Neu P@1 thap (ví dụ < 0.3) nghia la Điều đúng HAY bị xếp sau")
    print("nhieu Điều sai -> giảm k se mat recall nhanh, phải sửa RANKING.")

    # Phân tích nhiễu: Điều sai có cùng luật không?
    same_law = 0
    diff_law = 0
    for q in questions:
        results = bm.retrieve(q["question"], top_k=10)
        exp = set(q["expected_articles"])
        for r in results:
            if r["article_id"] in exp:
                continue
            if r["law_id"] == "59/2020/QH14":
                same_law += 1
            else:
                diff_law += 1
    print()
    print(f"Nhiễu trong top-10: {same_law} cung luat / {diff_law} khac luat")
    if diff_law == 0 and same_law > 0:
        print("-> Toan bo nhieu la 'Điều sai trong ĐÚNG luật'.")
        print("   Nghia la loi o xep hang, khong phai o nguon du lieu.")

    # Phân bố score: correct vs incorrect
    print()
    print("Phan bo score (Điều đúng vs sai):")
    correct_scores = []
    wrong_scores = []
    for q in questions:
        results = bm.retrieve(q["question"], top_k=20)
        exp = set(q["expected_articles"])
        for r in results:
            if r["article_id"] in exp:
                correct_scores.append(r["score"])
            else:
                wrong_scores.append(r["score"])
    correct_scores.sort(reverse=True)
    wrong_scores.sort(reverse=True)
    print(
        f"  đúng : n={len(correct_scores)} "
        f"median={correct_scores[len(correct_scores)//2]:.2f} "
        f"max={correct_scores[0]:.2f}"
    )
    print(
        f"  sai  : n={len(wrong_scores)} "
        f"median={wrong_scores[len(wrong_scores)//2]:.2f} "
        f"max={wrong_scores[0]:.2f}"
    )
    overlap = sum(1 for s in correct_scores if s >= correct_scores[len(correct_scores) // 2])
    print()
    print("Neu median(đúng) gan median(sai) -> score BM25 khong phân biệt")
    print("được đúng/sai, nên cat threshold khong giup được. Phải rerank.")


if __name__ == "__main__":
    main()
