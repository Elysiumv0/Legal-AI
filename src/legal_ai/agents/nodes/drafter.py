from legal_ai.agents.state import AgentState
from legal_ai.models.llm import LLMClient
DRAFTER_PROMPT = """Bạn là trợ lý pháp lý chuyên về pháp luật doanh nghiệp Việt Nam.
Dựa trên các điều luật được cung cấp, hãy trả lời câu hỏi chính xác và đầy đủ.

YÊU CẦU:
- Trích dẫn rõ số Điều và tên văn bản (VD: "theo Điều 47 Luật Doanh nghiệp 2020")
- Chỉ dựa trên điều luật được cung cấp, không tự suy diễn
- Nếu thiếu thông tin, nói rõ giới hạn
- Cuối câu trả lời thêm: "Lưu ý: Đây là tư vấn sơ bộ, vui lòng tham khảo luật sư để được tư vấn chính xác."

CÁC ĐIỀU LUẬT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""

def build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] {c['full_id']}\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )

class DrafterNode:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, state: AgentState) -> dict:
        question = state["question"]
        chunks   = state["retrieved_chunks"]
        context  = build_context(chunks)
        response = self.llm.chat([
            {"role": "system", "content": "Bạn là trợ lý pháp lý chuyên nghiệp Việt Nam."},
            {"role": "user",   "content": DRAFTER_PROMPT.format(context=context, question=question)},
        ])
        return {"draft_answer": response}