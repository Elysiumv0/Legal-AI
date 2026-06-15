"""
ColBERT Retriever — load index đã build và search.
Token-level MaxSim matching, tốt với thuật ngữ pháp lý đa âm tiết tiếng Việt.
"""

from pathlib import Path

from ragatouille import RAGPretrainedModel

INDEX_NAME = "legal_colbert"
INDEX_ROOT = "data/colbert_index"


class ColBERTRetriever:
    def __init__(
        self,
        index_name: str = INDEX_NAME,
        index_root: str = INDEX_ROOT,
    ):
        self.index_name = index_name
        self.index_root = index_root
        self.model: RAGPretrainedModel | None = None

    def load(self):
        """Load index đã build."""
        index_path = Path(self.index_root) / self.index_name
        if not index_path.exists():
            raise FileNotFoundError(
                f"ColBERT index không tồn tại: {index_path}\n"
                f"Chạy indexer.py trước để build index."
            )
        try:
            self.model = RAGPretrainedModel.from_index(str(index_path))
        except Exception as e:
            raise RuntimeError(
                f"Không thể load ColBERT index tại {index_path}. "
                f"Index có thể bị hỏng, thử build lại với --recreate.\n"
                f"Lỗi gốc: {e}"
            ) from e
        print(f"ColBERT index loaded: {index_path}")

    def _ensure_loaded(self):
        if self.model is None:
            self.load()

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        law_id_filter: str | None = None,
    ) -> list[dict]:
        """
        Search top-k chunks theo ColBERT MaxSim score.

        Args:
            query:         câu hỏi pháp lý
            top_k:         số kết quả trả về
            law_id_filter: lọc theo mã văn bản (VD: "59/2020/QH14")

        Returns:
            list[dict] cùng schema với DenseRetriever và BM25Retriever
        """
        self._ensure_loaded()

        # Nếu có filter, lấy nhiều hơn để sau khi lọc vẫn đủ top_k
        fetch_k = top_k * 3 if law_id_filter else top_k
        results = self.model.search(query=query, k=fetch_k)

        output = []
        for i, r in enumerate(results):
            meta = r.get("document_metadata", {})

            # Lọc theo law_id nếu cần
            if law_id_filter and meta.get("law_id") != law_id_filter:
                continue

            output.append({
                "chunk_id":   r["document_id"],
                "text":       r["content"],
                "full_id":    meta.get("full_id", ""),
                "law_id":     meta.get("law_id", ""),
                "law_name":   meta.get("law_name", ""),
                "article_id": meta.get("article_id", ""),
                "header":     meta.get("header", ""),
                "khoan":      meta.get("khoan"),
                "score":      r["score"],
                "rank":       len(output) + 1,
            })

            if len(output) >= top_k:
                break

        return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ColBERT retriever")
    parser.add_argument("--query", type=str, required=True, help="Câu hỏi test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--law_filter", type=str, default=None, help="Lọc theo law_id")
    args = parser.parse_args()

    retriever = ColBERTRetriever()
    retriever.load()

    results = retriever.retrieve(args.query, top_k=args.top_k, law_id_filter=args.law_filter)
    print(f"\nQuery: {args.query}")
    print("-" * 60)
    for r in results:
        print(f"[{r['score']:.3f}] {r['full_id']}")
        print(f"  {r['text'][:100]}...")
        print()
