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
        top_k_retrieve:   int = 20,
        top_k_rerank:     int = 10,
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
        """Return True nếu cần fallback sang Agent."""
        # Signal 1: RAG tự nhận bó tay
        answer_lower = answer.lower()
        is_clueless = any(kw in answer_lower for kw in RAG_BLIND_KEYWORDS)

        # Signal 2: Cosine score gốc thấp (từ rrf_score hoặc score)
        top_score = chunks[0].get("score", chunks[0].get("rrf_score", 1.0)) if chunks else 1.0
        low_score = top_score < SCORE_THRESHOLD

        # Signal 3: Không có citation Điều/Khoản/Chương/Nghị định
        no_citation = not re.search(
            r'(Điều|Khoản|Chương|Mục|Nghị định|Thông tư)\s+\d+',
            answer, re.IGNORECASE
        )

        # Signal 4: Chunk ít nội dung hữu ích
        useful = [c for c in chunks[:3] if len(c.get("text", "")) >= MIN_CHUNK_LEN]
        too_sparse = len(useful) < 2

        return is_clueless or low_score or no_citation or too_sparse

    def run(self, question: str) -> RAGResult:
        chunks = self.retrieve(question)
        top_scores = [
            c.get("score", c.get("rrf_score", 0.0)) for c in chunks[:3]
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
    from src.legal_ai.config.loader import load_settings
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
