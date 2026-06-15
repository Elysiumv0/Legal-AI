"""
Datasets — Load và quản lý bộ câu hỏi đánh giá
"""

import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    id:                int
    question:          str
    expected_articles: list[str]   # ["59/2020/QH14|Luật DN|Điều 74"]
    reference_answer:  str = ""
    expected_docs:     list[str] = field(default_factory=list)


class EvalDataset:
    def __init__(self, samples: list[EvalSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @classmethod
    def from_json(cls, path: str) -> "EvalDataset":
        """Load từ file JSON"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        samples = [
            EvalSample(
                id=item["id"],
                question=item["question"],
                expected_articles=item.get("expected_articles", []),
                reference_answer=item.get("reference_answer", ""),
                expected_docs=item.get("expected_docs", []),
            )
            for item in data
        ]
        return cls(samples)

    @classmethod
    def from_corpus(cls, corpus_path: str, n: int = 50) -> "EvalDataset":
        """
        Tạo eval dataset synthetic từ corpus.
        Dùng khi không có ground truth — lấy n Điều ngẫu nhiên,
        tạo câu hỏi đơn giản để test retrieval.
        """
        import random
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)

        articles = [
            a for doc in corpus
            for a in doc["articles"]
        ]
        sampled = random.sample(articles, min(n, len(articles)))

        samples = [
            EvalSample(
                id=i,
                question=f"Nội dung {a['article_id']} trong {a['law_name']} quy định gì?",
                expected_articles=[a["full_id"]],
                expected_docs=[f"{a['law_id']}|{a['law_name']}"],
            )
            for i, a in enumerate(sampled)
        ]
        return cls(samples)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "id": s.id,
                        "question": s.question,
                        "expected_articles": s.expected_articles,
                        "reference_answer": s.reference_answer,
                        "expected_docs": s.expected_docs,
                    }
                    for s in self.samples
                ],
                f, ensure_ascii=False, indent=2
            )