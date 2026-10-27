"""Đo code GỐC (từ git HEAD) trên CÙNG input, để so với code đã sửa.

Input dùng chung: data/processed/ocr_sample.txt (text OCR đã có).
Eval dùng chung: data/processed/eval_set.json (48 câu).
Hàm metric dùng chung: legal_ai.evaluation.metrics.

Mục đích: trả lời "so với code ban đầu thì sao" bằng số, không bằng lời.

Chạy: PYTHONPATH=src_orig:src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/measure_original.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# QUAN TRỌNG: code gốc nằm ở src_orig/src/legal_ai (vì git archive giữ
# đường dẫn src/legal_ai). Nếu chỉ trỏ tới src_orig thì import rơi vào code
# ĐANG SỬA trong src/ và kết quả sẽ "bằng nhau" một cách giả tạo — đã gặp
# đúng lỗi đó: báo cáo sai 111/0-dup thay vì 187/64-dup.
sys.path.insert(0, str(ROOT / "src_orig" / "src"))

from legal_ai.ingestion.parsers import split_by as split_by_orig  # noqa: E402
from legal_ai.ingestion.normalizers import normalize as normalize_orig  # noqa: E402
from legal_ai.ingestion.metadata import extract_structure as extract_structure_orig  # noqa: E402
from legal_ai.retrieval.sparse.bm25 import (  # noqa: E402
    BM25Retriever,
    tokenize_vi as tokenize_vi_orig,
)

del sys.path[1]  # bỏ src_orig/src, chỉ giữ src (code mới) cho phần còn lại
sys.path.insert(0, str(ROOT / "src"))
from legal_ai.evaluation.metrics import (  # noqa: E402
    compute_retrieval_metrics,
    macro_f2,
)


def unique_arts(results):
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
    raw = (ROOT / "data/processed/ocr_sample.txt").read_text(encoding="utf-8")
    questions = json.loads((ROOT / "data/processed/eval_set.json").read_text(encoding="utf-8"))

    # ── Ingest bằng code GỐC ────────────────────────────────────────────────
    print("[1] Ingest voi code GOC (git HEAD)...")
    clean = normalize_orig(raw)
    arts = split_by_orig(clean)
    ids = [a["article_id"] for a in arts]
    from collections import Counter

    dup = {k: v for k, v in Counter(ids).items() if v > 1}
    print(f"    split_by : {len(arts)} article | unique {len(set(ids))}")
    print(f"    duplicate: {len(dup)} -> {sorted(dup.items())[:6]}")
    sm = extract_structure_orig(arts)
    print(f"    structure_map: {len(sm)} (chunk co chapter: {sum(1 for c in arts if c.get('article_id') in sm)})")

    # Chuẩn bị chunk cho BM25 gốc
    chunks = []
    for a in arts:
        chunks.append(
            {
                "chunk_id": f"59/2020/QH14|Luật Doanh nghiệp 2020|{a['article_id']}",
                "text": f"Luật Doanh nghiệp 2020 - {a['header']}\n{a['content']}",
                "law_id": "59/2020/QH14",
                "law_name": "Luật Doanh nghiệp 2020",
                "article_id": a["article_id"],
                "header": a["header"],
                "full_id": f"59/2020/QH14|Luật Doanh nghiệp 2020|{a['article_id']}",
                "khoan": None,
            }
        )

    # ── BM25 gốc trên eval ──────────────────────────────────────────────────
    print()
    print("[2] BM25 code GOC tren 48 cau eval...")
    bm = BM25Retriever()
    bm.build(chunks)

    # Tokenize gốc có xử lý được lệch dấu không?
    tk = tokenize_vi_orig("thời hạn sử dụng đất nông nghiệp")
    print(f"    tokenize_vi goc: {tk}")
    has_plain = any("thoi" in t for t in tk)
    print(f"    xu ly lech dau: {'CO' if has_plain else 'KHONG (se miss neu van ban lech dau)'}")

    for K in (3, 10):
        P, R, F = [], [], []
        for q in questions:
            res = bm.retrieve(q["question"], top_k=K)
            top = unique_arts(res)[:K]
            p, r, f2 = prf(top, q["expected_articles"])
            P.append(p)
            R.append(r)
            F.append(f2)
        n = len(questions)
        print()
        print(f"  ── code GOC, top_k={K} ──")
        print(f"  P={sum(P)/n:.3f}  R={sum(R)/n:.3f}  F2={sum(F)/n:.3f}")

    # COLUMN CẢNH BÁO: KHÔNG in số của code đã sửa ở đây. Lần trước bảng này
    # có dòng GOC k=3 = 0.312/0.938/0.337 là số VIIẾT CỨNG, không đo được —
    # đã báo cáo sai vì nó. Chỉ in những gì script này thực sự đo.
    print()
    print("=" * 62)
    print("So sanh voi code ĐÃ SỬA: chay scripts/run_bm25_baseline.py")
    print("(tat ca tren cung ocr_sample.txt, cung eval_set.json, cung metric)")
    print("=" * 62)


if __name__ == "__main__":
    main()
