from typing import Dict, List, Any
from legal_ai.retrieval.sparse.bm25 import BM25Retriever
from legal_ai.retrieval.dense.retriever import DenseRetriever
from legal_ai.models.reranker import Reranker
from legal_ai.utils.legal_ontology import LegalOntologyMapper
from legal_ai.rag.citations import build_context

class LegalRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: Reranker,
        kg_graph: Dict[str, List[str]],
        law_metadata: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]] | None = None,  
    ):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.kg = kg_graph
        self.law_metadata = law_metadata
        self.ontology = LegalOntologyMapper()
        self.chunks = chunks or []
        self.chunk_map = {c["full_id"]: c for c in self.chunks}

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        max_kg_hops: int = 1,
        entities: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        # Ưu tiên NER entities nếu có law_ids
        filtered_laws = None
        if entities:
            # Dùng law_ids từ NER — chính xác hơn domain keyword matching
            law_ids = entities.get('law_ids', [])
            primary = entities.get('primary_law_id')
            if primary and primary not in law_ids:
                law_ids = [primary] + law_ids
            if law_ids:
                filtered_laws = law_ids

        # Fallback: dùng ontology domain filter nếu NER không có law_ids
        if filtered_laws is None:
            domain = self.ontology.get_domain(query)
            filtered_laws = self._filter_laws_by_domain(domain)

        hybrid_results = self._hybrid_retrieval(query, filtered_laws, top_k * 2)
        expanded_results = self._expand_via_kg(hybrid_results, max_kg_hops)
        reranked_results = self.reranker.rerank(
            query,
            expanded_results,
            top_k=top_k,
        )
        return reranked_results

    def _filter_laws_by_domain(self, domain: str) -> List[str]:
        if not domain or domain == "general":
            return None 
        filtered = [
            l["law_id"] for l in self.law_metadata
            if l.get("domain") == domain and l.get("status") == "active"
        ]
        if len(filtered) < 2:
            print(f"Domain '{domain}' chỉ có {len(filtered)} luật, bỏ filter")
            return None
        return filtered

    def _hybrid_retrieval(
        self,
        query: str,
        law_ids: List[str] | None,
        top_k: int
    ) -> List[Dict[str, Any]]:
        bm25_results = self.bm25.retrieve(query, top_k=top_k)
        dense_results = self.dense.retrieve(query, top_k=top_k)
        if law_ids:
            law_ids_set = set(law_ids)
            bm25_results = [r for r in bm25_results if r.get("law_id") in law_ids_set]
            dense_results = [r for r in dense_results if r.get("law_id") in law_ids_set]
            if not bm25_results and not dense_results:
                print(f"Filter {len(law_ids_set)} luật không có kết quả, bỏ filter")
                bm25_results = self.bm25.retrieve(query, top_k=top_k)
                dense_results = self.dense.retrieve(query, top_k=top_k)
        return self._merge_results(bm25_results, dense_results)

    def _expand_via_kg(
        self,
        results: List[Dict[str, Any]],
        max_hops: int
    ) -> List[Dict[str, Any]]:
        expanded = []
        for r in results:
            full_id = r.get("full_id")
            if not full_id or full_id not in self.kg:
                continue
            for ref_id in list(self.kg[full_id])[:3]:
                ref_chunk = self._get_chunk_by_full_id(ref_id)
                if ref_chunk:
                    expanded.append({
                        **ref_chunk,
                        "score": r.get("score", r.get("rrf_score", 0)) * 0.8,
                        "kg_source": full_id,
                        "kg_hop": 1
                    })
        seen = set()
        unique_results = []
        for r in results + expanded:
            cid = r.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                unique_results.append(r)
        return unique_results

    def _merge_results(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        rrf_scores = {}
        chunk_data = {}
        K = 60  
        for rank, r in enumerate(bm25_results):
            chunk_id = r.get("chunk_id")
            if not chunk_id:
                continue
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (K + rank + 1)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = r
        for rank, r in enumerate(dense_results):
            chunk_id = r.get("chunk_id")
            if not chunk_id:
                continue
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (K + rank + 1)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = r
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [
            {**chunk_data[cid], "rrf_score": rrf_scores[cid], "rank": i + 1}
            for i, cid in enumerate(sorted_ids)
        ]

    def _get_chunk_by_full_id(self, full_id: str) -> Dict[str, Any] | None:
        return self.chunk_map.get(full_id)