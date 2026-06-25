"""
DeepSeek API Client — dùng OpenAI SDK chuẩn theo https://api-docs.deepseek.com
- Model: deepseek-v4-pro / deepseek-v4-flash
- Dùng response_format="json_object" để ép JSON output
- thinking + reasoning_effort qua extra_body (vì OpenAI SDK chưa hỗ trợ)
"""

import os
import json
import time
import re
from typing import Any, List, Dict, Optional


class DeepSeekClient:
    """
    Client gọi DeepSeek API theo đúng spec:
    https://api-docs.deepseek.com/api/create-chat-completion
    
    Usage:
        client = DeepSeekClient()
        text = client.chat([{"role": "user", "content": "..."}])
        data = client.chat_json([{"role": "user", "content": "..."}])
    """

    BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-v4-pro"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: int = 60,
        response_format: Dict | None = None,
        reasoning_effort: str = "high",
        thinking: bool = True,
    ):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is required. "
                "Set environment variable or pass api_key=..."
            )

        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            timeout=timeout,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.response_format = response_format
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking

    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Dict | None = None,
    ) -> str:
        """Gửi 1 chat request → trả về text response.

        Args:
            messages: Danh sách messages (role + content).
            response_format: Override response_format cho lần gọi này
                (vd: {"type": "json_object"} cho chat_json).
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        # response_format — ưu tiên tham số truyền vào, nếu không thì dùng mặc định
        fmt = response_format if response_format is not None else self.response_format
        if fmt:
            kwargs["response_format"] = fmt

        # extra_body — luôn gửi thinking config (enabled hoặc disabled)
        # để tránh DeepSeek V4 mặc định bật thinking làm vỡ JSON output
        extra_body: Dict[str, Any] = {}
        if self.thinking:
            extra_body["thinking"] = {"type": "enabled"}
        else:
            extra_body["thinking"] = {"type": "disabled"}
        if self.thinking and self.reasoning_effort:
            extra_body["reasoning_effort"] = self.reasoning_effort
        kwargs["extra_body"] = extra_body

        max_retries = 5  # tăng retries vì API hay trả empty
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                # Empty response → retry
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  API returned empty (attempt {attempt+1}/{max_retries}), "
                          f"retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  API returned empty after {max_retries} attempts, giving up")
                    return ""
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  API error (attempt {attempt+1}/{max_retries}), "
                          f"retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"DeepSeek API error after {max_retries} retries: {e}"
                    ) from e

        return ""

    def chat_json(self, messages: List[Dict[str, str]]) -> dict:
        """Gửi chat request với response_format="json_object", parse thành dict.

        Ép DeepSeek API trả về JSON hợp lệ thay vì text tự do.
        Nếu parse thất bại, trả về raw text wrapped trong dict.
        """
        text = self.chat(messages, response_format={"type": "json_object"})
        if not text:
            return {}
        parsed = _parse_json(text)
        if parsed:
            return parsed
        # Fallback: nếu text bắt đầu bằng {, thử fixed JSON
        text_stripped = text.strip()
        if text_stripped.startswith("{") or text_stripped.startswith("["):
            # Thử đủ cách fix
            for fix_fn in [
                lambda t: t,
                lambda t: re.sub(r",\s*}", "}", t),
                lambda t: re.sub(r"([{,])\s*(\w+)\s*:", r'\1"\2":', t),
                lambda t: re.sub(r"'", '"', t),
            ]:
                try:
                    return json.loads(fix_fn(text_stripped))
                except (json.JSONDecodeError, TypeError):
                    continue
            # Cuối cùng: lưu raw text vào _raw field
            return {"_raw": text_stripped[:500], "_parse_error": True}
        return {}


def _parse_json(text: str) -> dict:
    """Parse JSON từ LLM response, có fallback + fix lỗi phổ biến."""
    if not text or not text.strip():
        return {}
    
    # Loại bỏ markdown code fences
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Tìm JSON object trong text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Fix lỗi phổ biến: trailing comma sau field cuối
        try:
            fixed = re.sub(r",\s*}", "}", candidate)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        # Fix lỗi: thiếu quote key
        try:
            fixed = re.sub(r"([{,])\s*(\w+)\s*:", r'\1"\2":', candidate)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
    
    # Trả về dict chứa field đầu tiên nếu có (dù không hoàn chỉnh)
    key_match = re.findall(r'"(\w+)":\s*(\[[^\]]*\]|"[^"]*"|\d+)', text)
    if key_match:
        result = {}
        for key, val in key_match:
            try:
                result[key] = json.loads(val)
            except:
                result[key] = val
        return result
    
    return {}


def test_connection():
    """Test nhanh kết nối DeepSeek API."""
    client = DeepSeekClient(
        temperature=0.1,
        max_tokens=100,
        reasoning_effort="low",
    )
    text = client.chat([
        {"role": "user", "content": "Nói 'OK' nếu bạn hoạt động."}
    ])
    print(f"Response: {text}")
    return text


if __name__ == "__main__":
    test_connection()
