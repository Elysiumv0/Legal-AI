"""
Shared JSON parsing and context building utilities.
Centralized to avoid duplication across agents/nodes and rag modules.
"""

import json
import re


def parse_json(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown code fences.
    
    Falls back to regex extraction if standard parsing fails.
    """
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract first {...} block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


def build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks for LLM prompts."""
    return "\n\n---\n\n".join(
        f"[{i}] {c['full_id']}\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
