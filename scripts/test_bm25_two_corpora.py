"""Kiem tra BM25 sau khi doi tokenize_vi sang dang khong dau thong nhat.

Chay: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/test_bm25_two_corpora.py
"""
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legal_ai.retrieval.sparse.bm25 import BM25Retriever

print("=== A. CORPUS THAT (Luat Doanh nghiep 2020) ===")
chunks = json.loads(
    (Path(__file__).resolve().parents[1] / "data/processed/chunks.json").read_text(
        encoding="utf-8"
    )
)
bm = BM25Retriever()
bm.build(chunks)
bm.save(
    str(Path(__file__).resolve().parents[1] / "data/indexes/bm25/bm25.pkl")
)

qs_a = [
    "điều kiện thành lập công ty TNHH một thành viên",
    "hồ sơ đăng ký doanh nghiệp",
    "Chủ tịch công ty TNHH có quyền gì",
    "dieu kien thanh lap cong ty TNHH mot thanh vien",  # khong dau
]
for q in qs_a:
    print("Q:", q)
    for r in bm.retrieve(q, top_k=3):
        art = r["full_id"].split("|")[-1]
        print(f"   [{r['score']:6.2f}] {art}")
    print()

print("=== B. CORPUS LECH DAU (Luat Dat đai gialap) ===")
chunks2 = json.loads(
    (Path(__file__).resolve().parents[1] / "data/processed/chunks_gen_test.json").read_text(
        encoding="utf-8"
    )
)
bm2 = BM25Retriever()
bm2.build(chunks2)
qs_b = [
    "thời hạn sử dụng đất nông nghiệp",
    "thu hồi đất vì mục đích quốc phòng",
    "thanh lập công ty TNHH",
]
for q in qs_b:
    res = bm2.retrieve(q, top_k=3)
    print("Q:", q)
    for r in res:
        print(f"   [{r['score']:6.2f}] {r['full_id'].split('|')[-1]}")
    if not res:
        print("   (0 ket qua - dung thiet ke, van ban dat đai khong co cong ty)")
