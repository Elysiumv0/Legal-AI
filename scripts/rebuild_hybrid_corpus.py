"""Thí nghiệm: dùng text OCR dpi=400 (27 trang) thay cho dpi=200 ở chỗ có.

Bối cảnh: lần re-OCR 27 trang ở dpi=400 + psm=6 chỉ lấy 2 Điều (40, 72) bị
thiếu. Nhưng vùng trang đó phủ ~45 Điều khác — chúng vẫn đang dùng text
dpi=200, vốn có lỗi OCR ("đăng_lý", "trìnhộ", thiếu dấu...).
Script này dựng corpus "hybrid": article nào có bản dpi=400 thì lấy bản đó,
không thì giữ dpi=200. Rồi chunk + đo F2 để xem text tốt hơn có giúp gì không.

Chạy: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -u scripts/rebuild_hybrid_corpus.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from legal_ai.ingestion.parsers import split_by  # noqa: E402
from legal_ai.ingestion.normalizers import normalize  # noqa: E402
from legal_ai.ingestion.chunkers import chunk_corpus  # noqa: E402
from legal_ai.retrieval.sparse.bm25 import BM25Retriever  # noqa: E402
from legal_ai.evaluation.metrics import compute_retrieval_metrics  # noqa: E402

LAW_ID = "59/2020/QH14"
LAW_NAME = "Luật Doanh nghiệp 2020"


def main():
    # ── 1. Trạng thái hiện tại ────────────────────────────────────────────────
    chunks_now = json.loads(
        (ROOT / "data/processed/chunks.json").read_text(encoding="utf-8")
    )
    arts_now = sorted({c["article_id"] for c in chunks_now})
    print(f"[state] chunks.json: {len(chunks_now)} chunk, {len(arts_now)} Điều")

    # ── 2. Corpus hybrid ──────────────────────────────────────────────────────
    corpus = json.loads(
        (ROOT / "data/processed/corpus_final.json").read_text(encoding="utf-8")
    )
    articles = corpus[0]["articles"]

    rec_text = (ROOT / "data/processed/ocr_recovered.txt").read_text(encoding="utf-8")
    # Tách từng trang rồi split để lấy article theo đúng ranh giới trang
    rec_arts = {}
    parts = rec_text.split("\n=== PAGE ")
    for p in parts[1:]:
        body = p.split("===\n", 1)[1] if "===\n" in p else p
        for a in split_by(normalize(body)):
            rec_arts.setdefault(a["article_id"], a)

    print(f"[hybrid] bản dpi=400 có {len(rec_arts)} Điều: "
          f"{sorted(rec_arts, key=lambda s: int(s.split()[1]))[:8]}...")

    replaced = []
    for a in articles:
        hi = rec_arts.get(a["article_id"])
        if hi and len(hi["content"]) > len(a["content"]) * 0.8:
            a["content"] = hi["content"]
            a["header"] = hi["header"]
            replaced.append(a["article_id"])

    print(f"[hybrid] thay {len(replaced)}/{len(articles)} Điều bằng bản dpi=400")

    corpus[0]["articles"] = articles
    corpus[0]["total_articles"] = len(articles)
    (ROOT / "data/processed/corpus_hybrid.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 3. Chunk ──────────────────────────────────────────────────────────────
    chunks = chunk_corpus(
        corpus_path="data/processed/corpus_hybrid.json",
        output_path="data/processed/chunks_hybrid.json",
    )
    print(f"[chunk] {len(chunks)} chunk | unique chunk_id "
          f"{len({c['chunk_id'] for c in chunks})}")

    # ── 4. Đo F2 ──────────────────────────────────────────────────────────────
    questions = json.loads(
        (ROOT / "data/processed/eval_set.json").read_text(encoding="utf-8")
    )
    bm = BM25Retriever()
    bm.build(chunks)
    print(f"[index] BM25 trên {len(chunks)} chunk hybrid")

    rows = []
    for q in questions:
        res = bm.retrieve(q["question"], top_k=3)
        seen, top = set(), []
        for r in res:
            if r["article_id"] not in seen:
                seen.add(r["article_id"])
                top.append(r["article_id"])
        rows.append(compute_retrieval_metrics(top[:3], q["expected_articles"]))
    f2 = sum(r.f2 for r in rows) / len(rows)
    p = sum(r.precision for r in rows) / len(rows)
    r_ = sum(r.recall for r in rows) / len(rows)

    print()
    print("=" * 56)
    print(f"{'':34} {'P':>7} {'R':>7} {'F2':>7}")
    print("-" * 56)
    print(f"{'chunks.json (text dpi=200)':34} {'0.458':>7} {'0.799':>7} {'0.669':>7}")
    print(f"{'chunks_hybrid (400 nơi có)':34} {p:>7.3f} {r_:>7.3f} {f2:>7.3f}")
    print("=" * 56)
    delta = f2 - 0.669
    print(f"Δ F2 = {delta:+.4f}")
    if delta > 0.005:
        print("=> text dpi=400 CÓ giúp. Nên: đổi chunks.json sang hybrid.")
    elif delta < -0.005:
        print("=> text dpi=400 LÀM TỆ HƠN. Giữ nguyên bản dpi=200.")
    else:
        print("=> không đáng kể. Không cần đổi.")


if __name__ == "__main__":
    main()
