from dataclasses import dataclass, field
from datetime import datetime, timezone

VALID_ROLES = {"user", "assistant", "system"}
DEFAULT_MAX_CONTENT_CHARS = 1200


@dataclass
class Turn:
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    articles: list[str] = field(default_factory=list)

    def to_message(self, max_chars: int = DEFAULT_MAX_CONTENT_CHARS) -> dict:
        content = self.content.strip()
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        return {"role": self.role, "content": content}


class ShortTermMemory:
    """
    Session-level memory cho hội thoại pháp lý.

    Mục tiêu:
    - Giữ ngữ cảnh vài lượt gần nhất để hiểu câu hỏi tiếp nối.
    - Ghi lại các điều luật đã dùng trong session để ưu tiên continuity.
    - Không lưu vĩnh viễn, tránh stale legal advice.
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ):
        self.max_turns = max_turns
        self.max_content_chars = max_content_chars
        self.turns: list[Turn] = []
        self.session_articles: set[str] = set()

    def add_turn(self, role: str, content: str, articles: list[str] | None = None):
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Expected one of {sorted(VALID_ROLES)}")
        clean_content = content.strip()
        if not clean_content:
            return
        turn = Turn(role=role, content=clean_content, articles=articles or [])
        self.turns.append(turn)
        self.session_articles.update(articles or [])
        self._trim()

    def add_user(self, content: str):
        self.add_turn("user", content)

    def add_assistant(self, content: str, articles: list[str] | None = None):
        self.add_turn("assistant", content, articles=articles)

    def _trim(self):
        max_items = self.max_turns * 2
        if len(self.turns) > max_items:
            self.turns = self.turns[-max_items:]

    def get_messages(self) -> list[dict]:
        return [t.to_message(self.max_content_chars) for t in self.turns]

    def recent_summary(self, last_n: int = 4) -> str:
        if not self.turns:
            return ""
        recent = self.turns[-last_n:]
        lines = []
        for turn in recent:
            content = turn.content.strip().replace("\n", " ")
            if len(content) > 300:
                content = content[:300].rstrip() + "..."
            articles = f" | articles={', '.join(turn.articles[:5])}" if turn.articles else ""
            lines.append(f"{turn.role}: {content}{articles}")
        return "\n".join(lines)

    def condense_query(self, new_query: str) -> str:
        """
        Tạo query có bối cảnh cho retrieval khi user hỏi nối tiếp.
        Không dùng để trả lời trực tiếp; chỉ dùng làm query expansion/context.
        """
        if not self.turns:
            return new_query.strip()

        history = self.recent_summary(last_n=4)
        articles = sorted(self.session_articles)
        article_context = ""
        if articles:
            article_context = "\nCác điều luật đã nhắc trong phiên: " + "; ".join(articles[:20])

        return (
            "Bối cảnh hội thoại gần đây:\n"
            f"{history}"
            f"{article_context}\n\n"
            f"Câu hỏi mới: {new_query.strip()}"
        )

    def clear(self):
        self.turns = []
        self.session_articles = set()

    @property
    def is_empty(self) -> bool:
        return len(self.turns) == 0
