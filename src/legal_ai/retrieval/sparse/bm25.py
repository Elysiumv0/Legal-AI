import re
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from legal_ai.retrieval.base import BaseRetriever
from legal_ai.utils.text import normalize_whitespace

def tokenize_vi(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'điều\s+(\d+)', r'điều_\1', text)
    text = re.sub(r'khoản\s+(\d+)', r'khoản_\1', text)
    tokens = word_tokenize(text, format="text").split()
    tokens = [t.replace(" ", "_") for t in tokens]
    stopwords = {"và", "của", "các", "có", "được", "trong",
                 "là", "theo", "về", "này", "tại", "đến"}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


class BM25Retriever(BaseRetriever):
    def __init__(self):
        self.bm25   = None
        self.chunks = []

    def get_citation(self, chunk: dict) -> str:
        return f"{chunk['article_id']}, {chunk['law_name']}"

    def build(self, chunks: list[dict]):
        print(f"Building BM25 index for {len(chunks)} chunks")
        self.chunks = chunks
        corpus = [tokenize_vi(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def save(self, path: str = "data/indexes/bm25/bm25.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        print(f"BM25 index saved to {path}")

    def load(self, path: str = "data/indexes/bm25/bm25.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25   = data["bm25"]
        self.chunks = data["chunks"]
        print(f"BM25 index loaded ({len(self.chunks)} chunks)")

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        tokens = tokenize_vi(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        return [
            {
                "chunk_id":   self.chunks[i]["chunk_id"],
                "text":       self.chunks[i]["text"],
                "law_id":     self.chunks[i]["law_id"],
                "law_name":   self.chunks[i]["law_name"],
                "article_id": self.chunks[i]["article_id"],
                "full_id":    self.chunks[i]["full_id"],
                "score":      float(scores[i]),
                "rank":       rank + 1,
            }
            for rank, i in enumerate(top_indices)
            if scores[i] > 0  
        ]
    
def build_bm25_index(
    chunks_path="data/processed/chunks.json",
    index_path="data/indexes/bm25/bm25.pkl",
):
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    retriever = BM25Retriever()
    retriever.build(chunks)
    retriever.save(index_path)
    return retriever

if __name__ == "__main__":
    retriever = build_bm25_index()
    test_queries = [
        "điều kiện thành lập công ty TNHH một thành viên",
        "mức lương tối thiểu vùng",
        "thuế giá trị gia tăng hàng hóa xuất khẩu",
        "59/2020/QH14 Điều 47",
    ]
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = retriever.retrieve(query, top_k=3)
        for r in results:
            print(f"  [{r['score']:.2f}] {r['full_id']}")
            print(f"  {r['text'][:100]}")