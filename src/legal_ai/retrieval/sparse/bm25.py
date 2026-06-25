"""
BM25 Retriever — sparse lexical retrieval using rank-bm25.
Supports save/load for pre-built index.
"""

import re
import pickle
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

# Stopwords tiếng Việt phổ biến — chiếm IDF score vô ích
_VN_STOPWORDS = frozenset({
    "của", "và", "là", "có", "được", "trong", "cho", "với", "các",
    "này", "đã", "để", "theo", "từ", "đến", "không", "những", "một",
    "về", "tại", "hoặc", "hay", "do", "khi", "nếu", "bị", "thì",
    "cũng", "như", "nhưng", "mà", "vì", "nên", "sẽ", "đó", "đây",
    "rằng", "lại", "còn", "ra", "vào", "trên", "qua", "sau",
})

# Regex loại bỏ dấu câu (giữ chữ, số, khoảng trắng)
_RE_PUNCT = re.compile(r'[^\w\s]', re.UNICODE)


class BM25Retriever:
    """
    BM25 lexical retrieval với index có thể lưu/load.
    Dùng rank_bm25 (Okapi BM25) — nhẹ, không cần GPU.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    # ── tokenizer tiếng Việt ──
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tách từ tiếng Việt: NFKC normalize → lowercase → bỏ dấu câu → split → lọc stopwords + token ngắn."""
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        text = _RE_PUNCT.sub(' ', text)
        return [t for t in text.split() if len(t) > 1 and t not in _VN_STOPWORDS]

    # ── build index ──
    def build(self, chunks: List[Dict[str, Any]]):
        """Build BM25 index từ list chunks (mỗi chunk có key 'text')."""
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 index built: {len(chunks)} documents")

    # ── save / load ──
    def save(self, path: str):
        """Lưu tokenized corpus ra pickle (nhẹ, nhanh load lại)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"BM25 index saved to {path}")

    def load(self, path: str):
        """Load BM25 index từ pickle.

        Hỗ trợ 2 format:
        - New: {"chunks": [...], "tokenized_corpus": [[...], ...]}
        - Old: {"bm25": BM25Okapi, "chunks": [...]}
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]

        if "tokenized_corpus" in data:
            # New format: có sẵn tokenized corpus, rebuild BM25
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        elif "bm25" in data:
            # Old format: dùng thẳng BM25 object đã được pickle
            self.bm25 = data["bm25"]
            self.tokenized_corpus = []  # không cần, BM25 đã ready
        else:
            # Fallback: rebuild từ chunks
            self.tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 index loaded from {path}: {len(self.chunks)} documents")

    # ── retrieve ──
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Search top-k chunks theo BM25 score.

        Args:
            query: câu hỏi pháp lý
            top_k: số kết quả trả về

        Returns:
            list[dict] mỗi phần tử có keys: chunk_id, text, law_id, law_name,
            article_id, full_id, score, rank
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built or loaded. Call build() or load() first.")

        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Lấy top-k indices
        top_k = min(top_k, len(doc_scores))
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id":   chunk.get("chunk_id", ""),
                "text":       chunk.get("text", ""),
                "full_id":    chunk.get("full_id", ""),
                "law_id":     chunk.get("law_id", ""),
                "law_name":   chunk.get("law_name", ""),
                "article_id": chunk.get("article_id", ""),
                "header":     chunk.get("header", ""),
                "khoan":      chunk.get("khoan"),
                "score":      float(doc_scores[idx]),
                "rank":       rank + 1,
            })

        return results
