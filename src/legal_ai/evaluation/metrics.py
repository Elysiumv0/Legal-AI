"""
Metrics — Đánh giá chất lượng retrieval và QA
Metric chính: F2 (recall quan trọng hơn precision)
"""

import re
import time
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    precision:  float
    recall:     float
    f2:         float


@dataclass
class QAMetrics:
    has_citation:    bool    # Có "Điều X" không
    has_disclaimer:  bool    # Có cảnh báo AI không
    citation_correct: float  # Tỷ lệ điều trích dẫn đúng
    latency_ms:      float


# ── Retrieval Metrics ─────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], expected: list[str]) -> float:
    if not retrieved:
        return 0.0
    correct = len(set(retrieved) & set(expected))
    return correct / len(retrieved)


def recall_at_k(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    correct = len(set(retrieved) & set(expected))
    return correct / len(expected)


def f2_score(precision: float, recall: float) -> float:
    """
    F2 = (5 × P × R) / (4 × P + R)
    Đây là metric chính của cuộc thi — recall quan trọng gấp đôi precision
    """
    denom = 4 * precision + recall
    if denom == 0:
        return 0.0
    return (5 * precision * recall) / denom


def compute_retrieval_metrics(
    retrieved_articles: list[str],
    expected_articles:  list[str],
) -> RetrievalMetrics:
    """Tính P, R, F2 cho 1 query"""
    # Normalize: bỏ khoản suffix, chỉ giữ Điều
    def normalize(articles):
        result = set()
        for a in articles:
            # "59/2020/QH14|Luật DN|Điều 47|khoản_1" → "59/2020/QH14|Luật DN|Điều 47"
            parts = a.split("|")
            result.add("|".join(parts[:3]))
        return result

    retrieved_norm = normalize(retrieved_articles)
    expected_norm  = normalize(expected_articles)

    p = precision_at_k(list(retrieved_norm), list(expected_norm))
    r = recall_at_k(list(retrieved_norm), list(expected_norm))
    f2 = f2_score(p, r)

    return RetrievalMetrics(precision=p, recall=r, f2=f2)


def macro_f2(results: list[RetrievalMetrics]) -> float:
    """Macro-average F2 — metric cuối cùng của cuộc thi"""
    if not results:
        return 0.0
    return sum(r.f2 for r in results) / len(results)


# ── QA Metrics ────────────────────────────────────────────────────────────────

DISCLAIMER_PATTERNS = [
    r'tư vấn sơ bộ',
    r'tham khảo luật sư',
    r'không phải tư vấn pháp lý chính thức',
    r'cần xác nhận',
]


def has_article_citation(answer: str) -> bool:
    return bool(re.search(r'Điều\s+\d+', answer))


def has_disclaimer(answer: str) -> bool:
    return any(re.search(p, answer, re.IGNORECASE) for p in DISCLAIMER_PATTERNS)


def citation_coverage(answer: str, expected_articles: list[str]) -> float:
    """Tỷ lệ điều luật expected được nhắc đến trong answer"""
    if not expected_articles:
        return 1.0
    mentioned = 0
    for article in expected_articles:
        # Lấy số điều: "59/2020/QH14|Luật DN|Điều 47" → "47"
        match = re.search(r'Điều\s+(\d+)', article)
        if match and re.search(rf'Điều\s+{match.group(1)}', answer):
            mentioned += 1
    return mentioned / len(expected_articles)


def compute_qa_metrics(
    answer: str,
    expected_articles: list[str],
    latency_ms: float = 0.0,
) -> QAMetrics:
    return QAMetrics(
        has_citation=has_article_citation(answer),
        has_disclaimer=has_disclaimer(answer),
        citation_correct=citation_coverage(answer, expected_articles),
        latency_ms=latency_ms,
    )


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(
    retrieval_results: list[RetrievalMetrics],
    qa_results: list[QAMetrics],
):
    macro = macro_f2(retrieval_results)
    avg_p = sum(r.precision for r in retrieval_results) / len(retrieval_results)
    avg_r = sum(r.recall    for r in retrieval_results) / len(retrieval_results)

    pct_citation   = sum(1 for q in qa_results if q.has_citation)   / len(qa_results) * 100
    pct_disclaimer = sum(1 for q in qa_results if q.has_disclaimer) / len(qa_results) * 100
    avg_coverage   = sum(q.citation_correct for q in qa_results)    / len(qa_results)
    avg_latency    = sum(q.latency_ms for q in qa_results)          / len(qa_results)

    print("=" * 50)
    print("RETRIEVAL")
    print(f"  Macro-F2:  {macro:.4f}")
    print(f"  Precision: {avg_p:.4f}")
    print(f"  Recall:    {avg_r:.4f}")
    print("QA")
    print(f"  Citation:   {pct_citation:.1f}%")
    print(f"  Disclaimer: {pct_disclaimer:.1f}%")
    print(f"  Coverage:   {avg_coverage:.4f}")
    print(f"  Latency:    {avg_latency:.0f}ms")
    print("=" * 50)