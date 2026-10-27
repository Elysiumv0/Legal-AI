"""Tham so retrieval — mot cho dinh nghia, moi noi dung dung.

Nguon goc con so: do tren 48 cau hoi eval (scripts/run_bm25_baseline.py +
scripts/analyze_precision.py), BM25-only tren 114 chunk cua Luat Doanh nghiep 2020:

    k=1   P=0.812  R=0.497  F2=0.527
    k=2   P=0.594  R=0.708  F2=0.660
    k=3   P=0.458  R=0.799  F2=0.669   <-- F2 dinh
    k=5   P=0.304  R=0.854  F2=0.600
    k=10  P=0.167  R=0.920  F2=0.464   <-- gia tri cu, bi precision keo xuong

F2 phai cao khi recall gap doi precision, nen lay k=3 thay vi k=10.

Nhiễu 399 cung luat / 0 khac luat: loi o xep hang, khong phai nguon du lieu.
Phan bo score đung/sai chong lap (median 23.5 vs 13.2, max 58 vs 44) nen cat
threshold khong giup duoc.
"""

# Số Điều trả ra cuoi cung cho LLM (context window va precision)
TOP_K_FINAL = 3

# Số ứng viên lấy TRƯỚC khi rerank/fuse. Phải > TOP_K_FINAL để reranker có
# đủ đầu vào; k=10 cho reranker cơ hội nâng Điều đúng từ hạng 4-10 lên top-3.
# Đo trên 48 câu eval: retrieve=10 -> F2 0.723, retrieve=20 -> F2 0.733.
# Chọn 20: +1.4% F2, đánh đổi rerank gấp đôi chunk (chấp nhận được vì chỉ
# 114 chunk; nếu corpus lớn hơn thì nên quay lại 10).
TOP_K_RETRIEVE = 20

# Reranker xử lý batch bao nhiêu cặp (query, passage) một lần. 16 vừa RAM CPU.
RERANK_BATCH_SIZE = 16
