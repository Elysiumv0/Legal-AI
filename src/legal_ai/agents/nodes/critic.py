import json
import re
from legal_ai.agents.state import AgentState
from legal_ai.models.llm import LLMClient
from legal_ai.utils.json_utils import parse_json, build_context
MAX_RETRIEVAL_ATTEMPTS = 3

CRITIC_PROMPT = """Bạn là chuyên gia pháp lý đánh giá chất lượng thông tin.
Đánh giá xem các điều luật sau có đủ để trả lời câu hỏi không.

Câu hỏi: {question}

Các điều luật đã tìm được:
{context}

Trả về JSON:
{{
  "sufficient": true/false,
  "feedback": "lý do nếu chưa đủ, hoặc ok nếu đủ",
  "missing": "thông tin còn thiếu nếu có"
}}"""

class CriticNode:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, state: AgentState) -> dict:
        question = state["question"]
        chunks   = state["retrieved_chunks"]
        attempts = state.get("retrieval_attempts", 1)
        if attempts >= MAX_RETRIEVAL_ATTEMPTS:
            return {"context_sufficient": bool(chunks), "critic_feedback": "Max attempts reached. Answer with limitations."}
        if not chunks:
            return {"context_sufficient": False, "critic_feedback": "Không tìm được điều luật liên quan"}
        context  = build_context(chunks[:5])
        response = self.llm.chat([
            {"role": "system", "content": "Bạn là chuyên gia pháp lý đánh giá thông tin."},
            {"role": "user",   "content": CRITIC_PROMPT.format(question=question, context=context)},
        ])
        parsed    = parse_json(response)
        sufficient = parsed.get("sufficient", False)
        feedback   = parsed.get("feedback", "")
        missing    = parsed.get("missing", "")
        return {
            "context_sufficient": sufficient,
            "critic_feedback":    f"{feedback}. Cần thêm: {missing}" if missing else feedback,
        }