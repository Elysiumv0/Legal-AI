"""
Fixture dựng y chang failure mode thật trong corpus văn bản pháp luật VN.
Chạy: python -I test_fixtures/fixture_article_split.py
"""
import sys
from pathlib import Path

# Windows console mac dinh cp1252, print tieng Viet co dau bi UnicodeEncodeError.
# Ep utf-8 de fixture chay duoc tren powershell/bash thong thuong.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_ai.ingestion.parsers import split_by  # noqa: E402
from legal_ai.ingestion.normalizers import normalize  # noqa: E402
from legal_ai.ingestion.metadata import extract_structure  # noqa: E402

# ── Case A: cross-reference giữa câu (bug cũ: cắt cụt Điều 93, sinh article giả) ──
# Lấy nguyên từ reranker.py mock, đó là dạng OCR/pdf thật:
#   "Điều 93. ... Không thuộc đối tượng quy định tại khoản 2 Điều 17 của Luật này."
CASE_A = """Điều 93. Tiêu chuẩn và điều kiện của thành viên Hội đồng thành viên
1. Không thuộc đối tượng quy định tại khoản 2 Điều 17 của Luật này.
2. Có trình độ trung cấp trở lên chuyên ngành tài chính, kế toán.

Điều 94. Nhiệm vụ và quyền hạn của thành viên Hội đồng thành viên
1. Chấp thuận nghị quyết của Hội đồng thành viên về góp vốn.
"""

# ── Case B: footer "Trang 2" phải mất, nhưng "Trang thiết bị" phải ở lại ──
CASE_B = """Trang 2

Trang thiết bị phải bảo đảm an toàn lao động theo Điều 108 của Luật này.
Người sử dụng lao động niêm yết quy định về an toàn trên trang thông tin điện tử.
ThuVienPhapLuat
028-38265549
+84 24 3826 5549
--------- page break ---------
"""

# ── Case C: văn bản không dùng dấu chấm sau số điều (metadata cũ rơi entry) ──
CASE_C = """Điều 47
1. Vốn điều lệ của công ty trách nhiệm hữu hạn hai thành viên trở lên.

Điều 48 Tiêu chuẩn của thành viên góp vốn
1. Thành viên góp vốn phải đủ 18 tuổi.
"""


def show(title, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {title}")
    if not cond and detail:
        print(f"         {detail}")
    return cond


results = []

for name, text in [("A cross-ref giữa câu", CASE_A),
                   ("B footer trang", CASE_B),
                   ("C heading không dấu", CASE_C)]:
    print(f"\n── {name} ──")
    clean = normalize(text)
    arts = split_by(clean)

    ids = [a["article_id"] for a in arts]
    print(f"  split → {len(arts)} article: {ids}")

    if name.startswith("A"):
        # Điều 93 không được cắt cụt: nội dung phải chứa cả 2 khoản.
        a93 = next((a for a in arts if a["article_id"] == "Điều 93"), None)
        results.append(show("Điều 93 được nhận ra", a93 is not None, f"ids={ids}"))
        results.append(show(
            "Điều 93 KHÔNG bị cắt cụt (còn Khoản 2)",
            a93 is not None and "trình độ trung cấp" in a93["content"],
        ))
        # Không được sinh article giả "Điều 17" từ cross-ref mid-sentence.
        results.append(show(
            "không sinh article giả 'Điều 17'",
            ids.count("Điều 17") == 0,
            f"count={ids.count('Điều 17')}",
        ))
        results.append(show("Điều 94 được nhận ra", "Điều 94" in ids, f"ids={ids}"))

    if name.startswith("B"):
        kept = normalize(CASE_B)
        results.append(show("footer 'đúng dạng Trang + số' bị xoá", "Trang 2" not in kept))
        results.append(show(
            "câu 'Trang thiết bị phải bảo đảm an toàn' còn nguyên",
            "Trang thiết bị phải bảo đảm an toàn" in kept,
            "bị mất → vẫn còn bug xoá quá tay",
        ))
        results.append(show(
            "câu 'trang thông tin điện tử' còn nguyên",
            "trang thông tin điện tử" in kept,
            "bị mất → vẫn còn bug xoá quá tay",
        ))
        results.append(show("watermark tvpl bị xoá", "ThuVienPhapLuat" not in kept))

    if name.startswith("C"):
        structure = extract_structure(arts)
        print(f"  structure_map keys: {sorted(structure.keys())}")
        for aid in ["Điều 47", "Điều 48"]:
            results.append(show(
                f"{aid} vào structure_map (không None) dù không có dấu chấm",
                aid in structure,
                f"keys={sorted(structure.keys())}",
            ))

print("\n" + "=" * 62)
ok = sum(results)
print(f"  {ok}/{len(results)} PASS")
print("=" * 62)
sys.exit(0 if ok == len(results) else 1)
