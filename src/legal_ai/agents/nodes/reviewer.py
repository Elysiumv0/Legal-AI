import re
from legal_ai.agents.state import AgentState
from legal_ai.models.llm import LLMClient
from legal_ai.utils.text import has_article_citation
from legal_ai.rag.citations import build_context, extract_articles

REVIEWER_PROMPT = """Bạn là chuyên gia pháp lý kiểm tra chất lượng câu trả lời.

Kiểm tra:
1. Có trích dẫn "Điều X" cụ thể không?
2. Có bịa điều luật không tồn tại trong context không?
3. Có rõ ràng, dễ hiểu không?

Nếu đã tốt → trả về nguyên văn.
Nếu cần sửa → trả về bản đã chỉnh sửa.

CÁC ĐIỀU LUẬT (để kiểm chứng):
{context}

CÂU TRẢ LỜI CẦN KIỂM TRA:
{draft}

KẾT QUẢ:"""

RETRY_SUFFIX = "\n\nLƯU Ý QUAN TRỌNG: Bắt buộc phải trích dẫn số Điều cụ thể trong câu trả lời."

DRAFTER_PROMPT = """Bạn là trợ lý pháp lý chuyên nghiệp Việt Nam.
Dựa trên các điều luật được cung cấp, hãy trả lời câu hỏi chính xác và đầy đủ.
Trích dẫn rõ số Điều và tên văn bản trong câu trả lời.

CÁC ĐIỀU LUẬT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""

class ReviewerNode:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _has_citation(self, text: str) -> bool:
        return bool(re.search(r'Điều\s+\d+[a-z]?', text))

    def run(self, state: AgentState) -> dict:
        question = state["question"]
        draft    = state["draft_answer"]
        chunks   = state["retrieved_chunks"]
        context  = build_context(chunks)
        if self._has_citation(draft):
            final = self.llm.chat([
                {"role": "system", "content": "Bạn là chuyên gia pháp lý kiểm tra câu trả lời."},
                {"role": "user",   "content": REVIEWER_PROMPT.format(context=context, draft=draft)},
            ])
        else:
            final = self.llm.chat([
                {"role": "system", "content": "Bạn là trợ lý pháp lý chuyên nghiệp Việt Nam."},
                {"role": "user",   "content": DRAFTER_PROMPT.format(
                    context=context, question=question
                ) + RETRY_SUFFIX},
            ])
        docs, articles = extract_articles(chunks)
        return {
            "final_answer":      final,
            "relevant_docs":     docs,
            "relevant_articles": articles,
        }