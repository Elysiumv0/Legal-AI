from collections import defaultdict
def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    weights: list[float] | None = None
) -> list[dict]:
    if weights is None:
        weights = [1.0] * len(result_lists)
    assert len(weights) == len(result_lists)
    rrf_scores = defaultdict(float)
    chunk_data  = {}
    for result_list, weight in zip(result_lists, weights):
        for rank, chunk in enumerate(result_list):
            chunk_id = chunk["chunk_id"]
            rrf_scores[chunk_id] += weight / (k + rank + 1)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = chunk
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [
        {
            **chunk_data[cid],
            "rrf_score": rrf_scores[cid],
            "rank": i + 1,
        }
        for i, cid in enumerate(sorted_ids)
    ]

class HybridRetriever:
    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        article_retriever=None,
        colbert_retriever=None,
        bm25_weight:    float = 1.0,
        dense_weight:   float = 1.5,
        article_weight: float = 2.0,
        colbert_weight: float = 2.0,
    ):
        self.bm25     = bm25_retriever
        self.dense    = dense_retriever
        self.article  = article_retriever
        self.colbert  = colbert_retriever
        self.weights  = {
            "bm25":    bm25_weight,
            "dense":   dense_weight,
            "article": article_weight,
            "colbert": colbert_weight,
        }

    def _retrieve_single(self, query: str, each_top_k: int) -> list[list[dict]]:
        result_lists  = []
        weight_list   = []
        result_lists.append(self.bm25.retrieve(query, top_k=each_top_k))
        weight_list.append(self.weights["bm25"])
        result_lists.append(self.dense.retrieve(query, top_k=each_top_k))
        weight_list.append(self.weights["dense"])
        if self.article:
            result_lists.append(self.article.retrieve(query, top_k=each_top_k // 2))
            weight_list.append(self.weights["article"])
        if self.colbert:
            result_lists.append(self.colbert.retrieve(query, top_k=each_top_k))
            weight_list.append(self.weights["colbert"])
        return result_lists, weight_list

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        each_top_k: int = 100,
        extra_queries: list[str] | None = None,
    ) -> list[dict]:
        all_lists   = []
        all_weights = []
        lists, weights = self._retrieve_single(query, each_top_k)
        all_lists.extend(lists)
        all_weights.extend(weights)
        if extra_queries:
            for eq in extra_queries:
                lists, weights = self._retrieve_single(eq, each_top_k // 2)
                # giảm weight xuống để không át query gốc
                all_lists.extend(lists)
                all_weights.extend([w * 0.7 for w in weights])

        fused = reciprocal_rank_fusion(all_lists, weights=all_weights)
        return fused[:top_k]
