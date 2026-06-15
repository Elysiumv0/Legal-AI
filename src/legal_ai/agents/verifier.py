import re
from vllm import LLM, SamplingParams
class ArticleVerifier:
    def __init__(self, llm: LLM, max_candidates: int = 20):
        self.llm           = llm
        self.max_candidates = max_candidates
        self.sampling = SamplingParams(temperature=0.0, max_tokens=50)

    def _build_prompt(self, query: str, candidates: list[dict]) -> str:
        items = "\n".join([
            f"{i+1}. [{c['article_id']} - {c['law_name']}]:\n{c['text'][:400]}"
            for i, c in enumerate(candidates[:self.max_candidates])
        ])
        return (
            f"<|im_start|>system\n"
            f"Bạn là chuyên gia pháp lý. Chỉ trả lời bằng số thứ tự các điều luật "
            f"trực tiếp liên quan đến câu hỏi, cách nhau bởi dấu phẩy.\n"
            f"Ví dụ: 1,3,7\n"
            f"Nếu không có điều nào liên quan: 0\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Câu hỏi: {query}\n\n{items}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def verify_batch(
        self,
        queries: list[str],
        all_candidates: list[list[dict]],
    ) -> list[list[dict]]:
        prompts = [
            self._build_prompt(q, cands)
            for q, cands in zip(queries, all_candidates)
        ]
        outputs = self.llm.generate(prompts, self.sampling)
        results = []
        for output, candidates in zip(outputs, all_candidates):
            text = output.outputs[0].text.strip()
            indices = set()
            for m in re.finditer(r'\d+', text):
                idx = int(m.group()) - 1
                if 0 <= idx < len(candidates):
                    indices.add(idx)
            results.append([candidates[i] for i in sorted(indices)])
        return results