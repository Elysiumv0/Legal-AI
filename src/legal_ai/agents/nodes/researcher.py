import re
import logging
from typing import Any
from legal_ai.agents.state import AgentState
from legal_ai.retrieval.hybrid.fusion import HybridRetriever
from legal_ai.models.reranker import Reranker
from legal_ai.models.llm import LLMClient
from legal_ai.utils.json_utils import parse_json
# Import DECOMPOSE_PROMPT từ nguồn duy nhất — tránh trùng lặp với prompts.py
from legal_ai.rag.prompts import DECOMPOSE_PROMPT

logger = logging.getLogger(__name__)


def deduplicate(chunks: list[dict]) -> list[dict]:
    seen, result = set(), []
    for c in chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            result.append(c)
    return result


def build_entity_context(entities: dict[str, Any]) -> str:
    """Tạo context string từ NER entities để inject vào prompt."""
    parts = []
    if entities.get('primary_law_id'):
        parts.append(f"Luật áp dụng: {entities['primary_law_id']}")
    if entities.get('legal_subjects'):
        parts.append(f"Chủ thể: {', '.join(entities['legal_subjects'])}")
    if entities.get('legal_actions'):
        parts.append(f"Hành vi: {', '.join(entities['legal_actions'])}")
    if entities.get('legal_objects'):
        parts.append(f"Đối tượng: {', '.join(entities['legal_objects'])}")
    if entities.get('penalties'):
        parts.append(f"Chế tài: {', '.join(entities['penalties'])}")
    if entities.get('article_nums'):
        parts.append(f"Điều khoản: {', '.join(entities['article_nums'])}")
    if parts:
        return "THỰC THỂ PHÁP LÝ ĐÃ XÁC ĐỊNH:\n" + "\n".join(f"- {p}" for p in parts) + "\n\n"
    return ""


def build_entity_weighted_query(question: str, entities: dict[str, Any]) -> str:
    """
    Tạo BM25 query: câu gốc + entity terms để boost.
    Giữ câu gốc làm neo chống lạc đề.
    Deduplicate: không thêm term đã có trong question.
    """
    question_lower = question.lower()
    boost_terms = []
    seen = set()
    # Ưu tiên legal_actions và legal_objects (mô tả chính xác hành vi)
    for field in ['legal_actions', 'legal_objects', 'penalties']:
        for term in entities.get(field, []):
            term_lower = term.lower()
            # Bỏ qua nếu term đã có trong câu gốc hoặc đã thêm
            if term_lower not in question_lower and term_lower not in seen:
                boost_terms.append(term)
                seen.add(term_lower)
    if boost_terms:
        return f"{question} {' '.join(boost_terms)}"
    return question


def compute_entity_match_bonus(chunk: dict, entities: dict[str, Any]) -> float:
    """
    Tính bonus score dựa trên mức độ khớp entity giữa chunk và NER.
    Trả về multiplier (0.0 – 0.5) để dùng multiplicative: score * (1 + bonus).
    Cap ở 0.5 để không overpower RRF score.
    
    Lưu ý: bonus cộng dồn theo thứ tự field (legal_actions → legal_objects →
    penalties → article_nums). Field xuất hiện trước có thể "ăn hết quota"
    0.5 nếu match nhiều — đây là hành vi chấp nhận được vì legal_actions
    là signal quan trọng nhất trong legal retrieval.
    """
    bonus = 0.0
    chunk_text = (
        (chunk.get('text', '') or '') + ' ' +
        (chunk.get('law_name', '') or '') + ' ' +
        (chunk.get('article_id', '') or '')
    ).lower()

    # Khớp legal_actions → bonus cao (cốt lõi hành vi pháp lý)
    for action in entities.get('legal_actions', []):
        if action.lower() in chunk_text:
            bonus += 0.10

    # Khớp legal_objects → bonus vừa
    for obj in entities.get('legal_objects', []):
        if obj.lower() in chunk_text:
            bonus += 0.06

    # Khớp penalties → bonus
    for penalty in entities.get('penalties', []):
        if penalty.lower() in chunk_text:
            bonus += 0.06

    # Khớp article_nums → bonus cao
    for art in entities.get('article_nums', []):
        if f'điều {art.lower()}' in chunk_text or art.lower() in chunk_text:
            bonus += 0.12

    return min(bonus, 0.5)  # Cap để không át RRF


def get_target_law_from_entities(entities: dict[str, Any]) -> str | None:
    """Lấy law_id để filter từ NER entities (ưu tiên primary_law_id)."""
    primary = entities.get('primary_law_id')
    if primary:
        return primary.lower()
    law_ids = entities.get('law_ids', [])
    if law_ids:
        return law_ids[0].lower()
    return None


class ResearcherNode:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        llm: LLMClient,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 15,
    ):
        self.hybrid   = hybrid_retriever
        self.reranker = reranker
        self.llm      = llm
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank   = top_k_rerank

    def decompose(self, question: str, entities: dict[str, Any] | None = None) -> list[str]:
        """Phân rã câu hỏi compound thành sub-queries.
        
        Lưu ý: Đây dùng LLMClient.chat() (API-style, ví dụ DeepSeek/OpenAI),
        KHÔNG PHẢI vLLM engine. Khác với QueryDecomposer trong query_expander.py
        (dùng vLLM batch generate). Hai luồng này phục vụ deployment khác nhau:
        - ResearcherNode.decompose() → khi dùng API LLM (DeepSeek, GPT)
        - QueryDecomposer → khi có vLLM local (self-hosted Qwen/Llama)
        """
        entity_ctx = build_entity_context(entities or {})
        prompt = DECOMPOSE_PROMPT.format(
            entity_context=entity_ctx,
            question=question,
        )
        response = self.llm.chat([
            {"role": "system", "content": "Bạn là chuyên gia pháp lý Việt Nam."},
            {"role": "user",   "content": prompt},
        ])
        parsed = parse_json(response)
        sub_queries = parsed.get("sub_queries", [question])
        if not isinstance(sub_queries, list):
            sub_queries = [sub_queries] if isinstance(sub_queries, str) else [question]
        # Chủ đích: luôn giữ câu gốc ở đầu danh sách sub-queries để đảm bảo
        # retrieval không bao giờ bỏ sót ngữ cảnh tổng thể. Nếu LLM trả 3 sub,
        # insert(0, question) → 4 phần tử → [:3] cắt sub cuối của LLM.
        # Đây là trade-off có chủ đích: ưu tiên giữ câu gốc > sub-query thứ 3.
        if question not in sub_queries:
            sub_queries.insert(0, question)
        return sub_queries[:3]

    def search(
        self,
        sub_queries: list[str],
        entities: dict[str, Any] | None = None,
    ) -> list[dict]:
        entities = entities or {}
        target_law = get_target_law_from_entities(entities)

        all_chunks = []
        for sq in sub_queries:
            # Dùng entity-weighted query cho BM25 (qua hybrid)
            bm25_query = build_entity_weighted_query(sq, entities)
            # Ưu tiên: nếu entity-weighted khác gốc, retrieve cả 2 rồi merge
            chunks = self.hybrid.retrieve(bm25_query, top_k=self.top_k_retrieve)
            if bm25_query != sq:
                # Thêm retrieval với câu gốc thuần để giữ semantic alignment
                extra_chunks = self.hybrid.retrieve(sq, top_k=self.top_k_retrieve // 2)
                chunks.extend(extra_chunks)

            # Hard filter theo primary_law_id nếu có
            if target_law:
                filtered = [c for c in chunks if target_law in (c.get('law_id', '') or '').lower()]
                if filtered:
                    chunks = filtered
                else:
                    # WARNING: entity filter trả về rỗng — có thể NER chọn sai law.
                    # Fallback về unfiltered list, nhưng log để debug.
                    logger.warning(
                        f"[ResearcherNode] Entity filter cho target_law='{target_law}' "
                        f"trả về 0 kết quả (sub_query='{sq[:60]}...'). "
                        f"Fallback về unfiltered list ({len(chunks)} chunks). "
                        f"Kiểm tra lại get_primary_law_id() hoặc LAW_NAME_ALIASES."
                    )

            all_chunks.extend(chunks)

        all_chunks = deduplicate(all_chunks)

        # Áp dụng entity match bonus trước khi rerank (multiplicative để giữ scale)
        if entities:
            for c in all_chunks:
                bonus = compute_entity_match_bonus(c, entities)
                existing_score = c.get('rrf_score', c.get('score', 0))
                c['rrf_score'] = existing_score * (1 + bonus)

        return self.reranker.rerank(sub_queries[0], all_chunks, top_k=self.top_k_rerank)

    def run(self, state: AgentState) -> dict:
        question = state["question"]
        entities = state.get("entities", {})
        attempts = state.get("retrieval_attempts", 0)

        if attempts > 0 and state.get("critic_feedback"):
            sub_queries = [question, f"{question} {state['critic_feedback']}"]
        else:
            sub_queries = self.decompose(question, entities)

        chunks = self.search(sub_queries, entities)
        return {
            "sub_queries":        sub_queries,
            "retrieved_chunks":   chunks,
            "retrieval_attempts": attempts + 1,
        }