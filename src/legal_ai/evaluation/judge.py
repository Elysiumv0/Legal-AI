"""
Judge — LLM-as-judge + rule-based judge cho legal QA
"""

import re
import json
from dataclasses import dataclass


@dataclass
class JudgeScore:
    legal_grounding:  float   # Căn cứ pháp lý chính xác (0-1)
    accuracy:         float   # Tính chính xác nội dung (0-1)
    completeness:     float   # Tính đầy đủ (0-1)
    practicality:     float   # Tính thực tiễn (0-1)
    clarity:          float   # Tính rõ ràng (0-1)
    overall:          float   # Trung bình
    feedback:         str = ""


# ── Rule-based Judge ──────────────────────────────────────────────────────────

class RuleBasedJudge:
    """
    Chấm điểm nhanh bằng rule — không cần LLM call.
    Dùng để filter trước khi gửi LLM judge.
    """

    def score(
        self,
        question: str,
        answer: str,
        expected_articles: list[str],
        context: str = "",
    ) -> JudgeScore:

        # 1. Legal grounding — có cite đúng Điều không
        legal = self._score_legal_grounding(answer, expected_articles)

        # 2. Accuracy — answer có bám context không
        accuracy = self._score_accuracy(answer, context)

        # 3. Completeness — có đủ các khía cạnh không
        completeness = self._score_completeness(answer, question)

        # 4. Practicality — có disclaimer, checklist nếu cần
        practicality = self._score_practicality(answer, question)

        # 5. Clarity — câu trả lời có rõ ràng không
        clarity = self._score_clarity(answer)

        overall = (legal + accuracy + completeness + practicality + clarity) / 5

        return JudgeScore(
            legal_grounding=legal,
            accuracy=accuracy,
            completeness=completeness,
            practicality=practicality,
            clarity=clarity,
            overall=overall,
        )

    def _score_legal_grounding(self, answer: str, expected: list[str]) -> float:
        if not re.search(r'Điều\s+\d+', answer):
            return 0.0
        if not expected:
            return 0.8
        mentioned = 0
        for a in expected:
            m = re.search(r'Điều\s+(\d+)', a)
            if m and re.search(rf'Điều\s+{re.escape(m.group(1))}', answer):
                mentioned += 1
        return min(1.0, mentioned / len(expected))

    def _score_accuracy(self, answer: str, context: str) -> float:
        if not context:
            return 0.5
        # Kiểm tra answer không có claim quá xa context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        return min(1.0, overlap * 2)

    def _score_completeness(self, answer: str, question: str) -> float:
        score = 0.5
        # Câu trả lời đủ dài
        if len(answer) > 200:
            score += 0.3
        # Có đề cập đến các khía cạnh chính
        if re.search(r'ngoài ra|bên cạnh|đồng thời|thêm vào', answer, re.IGNORECASE):
            score += 0.2
        return min(1.0, score)

    def _score_practicality(self, answer: str, question: str) -> float:
        score = 0.5
        # Có disclaimer
        if re.search(r'tư vấn sơ bộ|tham khảo luật sư', answer, re.IGNORECASE):
            score += 0.3
        # Có checklist nếu câu hỏi về thủ tục
        if re.search(r'thủ tục|quy trình|bước', question, re.IGNORECASE):
            if re.search(r'^\d+\.|bước \d+', answer, re.MULTILINE | re.IGNORECASE):
                score += 0.2
        return min(1.0, score)

    def _score_clarity(self, answer: str) -> float:
        score = 0.6
        # Không quá ngắn hoặc quá dài
        length = len(answer)
        if 100 < length < 2000:
            score += 0.2
        # Có cấu trúc (xuống dòng, đánh số)
        if '\n' in answer:
            score += 0.2
        return min(1.0, score)


# ── LLM Judge ─────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """Bạn là chuyên gia pháp lý đánh giá chất lượng câu trả lời AI.

Câu hỏi: {question}

Các điều luật căn cứ:
{context}

Câu trả lời cần đánh giá:
{answer}

Chấm điểm từ 0.0 đến 1.0 cho từng tiêu chí:
1. legal_grounding:  Có trích dẫn đúng điều luật không?
2. accuracy:         Nội dung có chính xác theo luật không?
3. completeness:     Có bao quát đầy đủ các khía cạnh không?
4. practicality:     Có thể áp dụng thực tế không?
5. clarity:          Có rõ ràng, dễ hiểu không?

Trả về JSON:
{{
  "legal_grounding": 0.0-1.0,
  "accuracy": 0.0-1.0,
  "completeness": 0.0-1.0,
  "practicality": 0.0-1.0,
  "clarity": 0.0-1.0,
  "feedback": "nhận xét ngắn gọn"
}}"""


class LLMJudge:
    """LLM-as-judge — chính xác hơn rule-based nhưng tốn LLM call"""

    def __init__(self, llm):
        self.llm = llm

    def score(
        self,
        question: str,
        answer: str,
        context: str = "",
    ) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(
            question=question,
            context=context[:2000],
            answer=answer,
        )
        response = self.llm.chat([
            {"role": "system", "content": "Bạn là chuyên gia pháp lý đánh giá khách quan."},
            {"role": "user",   "content": prompt},
        ])

        # Parse JSON
        text = re.sub(r"```json|```", "", response).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback rule-based nếu LLM trả về sai format
            return RuleBasedJudge().score(question, answer, [], context)

        scores = [
            data.get("legal_grounding", 0.5),
            data.get("accuracy", 0.5),
            data.get("completeness", 0.5),
            data.get("practicality", 0.5),
            data.get("clarity", 0.5),
        ]

        return JudgeScore(
            legal_grounding=scores[0],
            accuracy=scores[1],
            completeness=scores[2],
            practicality=scores[3],
            clarity=scores[4],
            overall=sum(scores) / len(scores),
            feedback=data.get("feedback", ""),
        )