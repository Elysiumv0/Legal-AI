import json
import re
from legal_ai.agents.state import AgentState
from legal_ai.retrieval.hybrid.fusion import HybridRetriever
from legal_ai.models.reranker import Reranker
from legal_ai.models.llm import LLMClient

DECOMPOSE_PROMPT = """Bạn là chuyên gia pháp lý Việt Nam.
Phân tích câu hỏi sau và tách thành các câu hỏi con đơn giản hơn nếu cần.
Nếu câu hỏi đã đơn giản, chỉ trả về câu hỏi gốc.

Trả về JSON:
{{"sub_queries": ["câu hỏi 1", "câu hỏi 2", ...]}}

Câu hỏi: {question}"""

def parse_json(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

def deduplicate(chunks: list[dict]) -> list[dict]:
    seen, result = set(), []
    for c in chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            result.append(c)
    return result

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

    def decompose(self, question: str) -> list[str]:
        response = self.llm.chat([
            {"role": "system", "content": "Bạn là chuyên gia pháp lý Việt Nam."},
            {"role": "user",   "content": DECOMPOSE_PROMPT.format(question=question)},
        ])
        parsed = parse_json(response)
        sub_queries = parsed.get("sub_queries", [question])
        if not isinstance(sub_queries, list):
            sub_queries = [sub_queries] if isinstance(sub_queries, str) else [question]
        if question not in sub_queries:
            sub_queries.insert(0, question)
        return sub_queries[:3]

    def search(self, sub_queries: list[str]) -> list[dict]:
        all_chunks = []
        for sq in sub_queries:
            chunks = self.hybrid.retrieve(sq, top_k=self.top_k_retrieve)
            all_chunks.extend(chunks)
        all_chunks = deduplicate(all_chunks)
        return self.reranker.rerank(sub_queries[0], all_chunks, top_k=self.top_k_rerank)

    def run(self, state: AgentState) -> dict:
        question = state["question"]
        attempts = state.get("retrieval_attempts", 0)
        if attempts > 0 and state.get("critic_feedback"):
            sub_queries = [question, f"{question} {state['critic_feedback']}"]
        else:
            sub_queries = self.decompose(question)
        chunks = self.search(sub_queries)
        return {
            "sub_queries":        sub_queries,
            "retrieved_chunks":   chunks,
            "retrieval_attempts": attempts + 1,
        }