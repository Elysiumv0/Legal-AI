"""
DeepSeekLLMClient — adapter triển khai interface LLMClient.chat()
dùng DeepSeek API thay cho local vLLM.
"""

import os
import re
import json
from typing import List, Dict

from legal_ai.models.deepseek_client import DeepSeekClient


class DeepSeekLLMClient:
    """
    Wrapper triển khai giao diện .chat(messages) → str
    dùng DeepSeek API (deepseek-v4-flash / deepseek-v4-pro).

    Usage:
        llm = DeepSeekLLMClient(model="deepseek-v4-flash")
        answer = llm.chat([
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-v4-flash",
        temperature: float = 0.0,
        max_tokens: int = 1500,
        timeout: int = 120,
    ):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is required. "
                "Set environment variable: set DEEPSEEK_API_KEY=sk-..."
            )
        self._client = DeepSeekClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            thinking=False,  # tắt thinking để response nhanh + JSON sạch
        )
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Gọi DeepSeek chat API, trả về text response."""
        return self._client.chat(messages)

    def chat_json(self, messages: List[Dict[str, str]]) -> dict:
        """Gọi DeepSeek chat API, parse JSON response."""
        return self._client.chat_json(messages)

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> List[str]:
        """
        Sinh batch prompts (mô phỏng vLLM.generate()).

        Args:
            prompts: Danh sách prompt string (đã format sẵn ChatML).
            temperature: Nhiệt độ sampling.
            max_tokens: Max tokens cho mỗi response.

        Returns:
            List response text tương ứng.
        """
        from tqdm import tqdm

        results = []
        old_temp = self._client.temperature
        old_max = self._client.max_tokens
        self._client.temperature = temperature
        self._client.max_tokens = max_tokens

        try:
            for prompt in tqdm(prompts, desc='DeepSeek API'):
                messages = self._parse_chatml(prompt)
                try:
                    text = self._client.chat(messages)
                    results.append(text.strip())
                except Exception as e:
                    print(f"  DeepSeek error: {e}")
                    results.append("")  # fallback empty
        finally:
            self._client.temperature = old_temp
            self._client.max_tokens = old_max

        return results

    @staticmethod
    def _parse_chatml(prompt: str) -> List[Dict[str, str]]:
        """
        Parse ChatML format → OpenAI messages format.

        Input:  <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        Output: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        messages = []
        pattern = r'<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>'
        for match in re.finditer(pattern, prompt, re.DOTALL):
            role = match.group(1)
            content = match.group(2).strip()
            if role != "assistant":  # Bỏ assistant prefix (chỉ là prompt continuation)
                messages.append({"role": role, "content": content})
        if not messages:
            # Fallback: toàn bộ prompt là user message
            messages = [{"role": "user", "content": prompt.strip()}]
        return messages
