import re
import yaml
from pathlib import Path
from dataclasses import dataclass
from legal_ai.retrieval.sparse.bm25 import BM25Retriever
from legal_ai.retrieval.dense.retriever import DenseRetriever, VectorStore
from legal_ai.models.embeddings import EmbeddingModel
from legal_ai.retrieval.hybrid.fusion import HybridRetriever
from legal_ai.models.reranker import Reranker
from legal_ai.models.llm import LLMClient
from legal_ai.rag.schemas  import RAGResult
from legal_ai.rag.citations import build_context, extract_articles
from legal_ai.rag.prompts   import RAG_PROMPT
from legal_ai.retrieval.config import TOP_K_FINAL, TOP_K_RETRIEVE

RAG_BLIND_KEYWORDS = [
    "không tìm thấy", "chưa có quy định", "chưa hỗ trợ",
    "không có thông tin", "không đủ thông tin",
    "nằm ngoài phạm vi", "không có căn cứ",
    "tôi không tìm thấy", "dữ liệu hiện tại chưa",
]

SCORE_THRESHOLD = 0.70
MIN_CHUNK_LEN = 150


class StandardRAGPipeline:
    def __init__(
        self,
        bm25_retriever:   BM25Retriever,
        dense_retriever:  DenseRetriever,
        reranker:         Reranker,
        llm:              LLMClient,
        top_k_retrieve:   int = TOP_K_RETRIEVE,
        top_k_rerank:     int = TOP_K_FINAL,
    ):
        self.hybrid = HybridRetriever(bm25_retriever, dense_retriever)
        self.reranker = reranker
        self.llm = llm
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank   = top_k_rerank

    def retrieve(self, query: str) -> list[dict]:
        fused = self.hybrid.retrieve(query, top_k=self.top_k_retrieve)
        reranked = self.reranker.rerank(query, fused, top_k=self.top_k_rerank)
        return reranked

    def augment(self, query: str, chunks: list[dict]) -> str:
        context = build_context(chunks)
        return RAG_PROMPT.format(context=context, question=query)

    def generate(self, prompt: str) -> str:
        return self.llm.complete(prompt)

    def _router_signals(self, answer: str, chunks: list[dict]) -> bool:
        """Return True nếu cần fallback sang Agent.

        LƯU Ý về score: trước đây dòng dưới so chunks[0]["score"] — key này
        là BM25 raw (0-20+) hoặc cosine (0-1) tuỳ nguồn nào surface trước,
        so với ngưỡng 0.70 thì vô nghĩa. Sau khi retrieve() chạy rerank,
        chunks[0] luôn mang rerank_score (sigmoid 0-1) — dùng đúng thang đó.
        """
        answer_lower = answer.lower()
        is_clueless = any(kw in answer_lower for kw in RAG_BLIND_KEYWORDS)
        top = chunks[0] if chunks else None
        if top is not None and "rerank_score" in top:
            low_score = float(top["rerank_score"]) < SCORE_THRESHOLD
        else:
            # Chưa qua rerank: không thể so thang BM25/cosine với 0.70,
            # coi như không có tín hiệu này thay vì đoán.
            low_score = False
        no_citation = not re.search(
            r'(Điều|Khoản|Chương|Mục|Nghị định|Thông tư)\s+\d+',
            answer, re.IGNORECASE
        )
        useful = [c for c in chunks[:3] if len(c.get("text", "")) >= MIN_CHUNK_LEN]
        too_sparse = len(useful) < 2
        return is_clueless or low_score or no_citation or too_sparse

    def run(self, question: str) -> RAGResult:
        chunks = self.retrieve(question)
        top_scores = [
            c.get("rerank_score", c.get("score", 0.0)) for c in chunks[:3]
        ]
        prompt = self.augment(question, chunks)
        answer = self.generate(prompt)
        docs, articles = extract_articles(chunks)
        needs_agent = self._router_signals(answer, chunks)
        return RAGResult(
            question=question,
            answer=answer,
            relevant_docs=docs,
            relevant_articles=articles,
            chunks_used=chunks,
            top_scores=top_scores,
            routed_to_agent=needs_agent,
        )

def build_pipeline(config_path: str = "configs/settings.yaml") -> StandardRAGPipeline:
    from legal_ai.config.loader import load_settings  # không "from src.legal_ai"
    cfg = load_settings(config_path)
    bm25 = BM25Retriever()
    bm25.load(str(cfg.indexes.bm25_path))
    embedder = EmbeddingModel(cfg.models.embedding.name)
    store    = VectorStore(str(cfg.indexes.qdrant_path))
    dense    = DenseRetriever(store, embedder)
    reranker = Reranker(cfg.models.reranker.name)
    llm      = LLMClient(cfg.models.llm.model_name)
    return StandardRAGPipeline(
        bm25_retriever=bm25,
        dense_retriever=dense,
        reranker=reranker,
        llm=llm,
        top_k_retrieve=cfg.retrieval.top_k_bm25,
        top_k_rerank=cfg.retrieval.top_k_rerank,
    )
