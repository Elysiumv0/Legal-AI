"""
Dense Retriever — wraps VectorStore + EmbeddingModel for dense vector search.
"""

from legal_ai.retrieval.dense.qdrant_store import VectorStore
from legal_ai.models.embeddings import EmbeddingModel


class DenseRetriever:
    """
    Dense (vector) retrieval using a Qdrant-backed VectorStore
    and a sentence-transformer style EmbeddingModel.
    """

    def __init__(self, store: VectorStore, embedding_model: EmbeddingModel):
        self.store = store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
        law_id_filter: str | None = None,
    ) -> list[dict]:
        """
        Encode query → search vector store → return top-k chunks.

        Args:
            query:         câu hỏi pháp lý
            top_k:         số kết quả trả về
            law_id_filter: lọc theo mã văn bản (VD: "59/2020/QH14")

        Returns:
            list[dict] mỗi phần tử có keys: chunk_id, text, law_id, law_name,
            article_id, full_id, score, rank
        """
        query_vec = self.embedding_model.encode([query], is_query=True)[0]
        return self.store.search(query_vec, top_k=top_k, law_id_filter=law_id_filter)
