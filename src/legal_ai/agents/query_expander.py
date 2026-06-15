import re
from vllm import LLM, SamplingParams

class QueryExpander:
    def __init__(self, llm: LLM, n_variants: int = 2):
        self.llm       = llm
        self.n_variants = n_variants
        self.sampling  = SamplingParams(temperature=0.3, max_tokens=150)

    def expand_batch(self, queries: list[str]) -> list[list[str]]:
        prompts = [self._build_prompt(q) for q in queries]
        outputs = self.llm.generate(prompts, self.sampling)
        results = []
        for original, output in zip(queries, outputs):
            text     = output.outputs[0].text.strip()
            variants = [l.strip() for l in text.split('\n') if l.strip()]
            variants = variants[:self.n_variants]
            results.append([original] + variants)
        return results

    def _build_prompt(self, query: str) -> str:
        return (
            f"<|im_start|>system\n"
            f"Viết lại câu hỏi pháp lý sau thành {self.n_variants} cách "
            f"dùng thuật ngữ pháp lý chính xác hơn. Mỗi cách 1 dòng, "
            f"không đánh số, không giải thích.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n{query}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )