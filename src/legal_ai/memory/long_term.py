import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

DEFAULT_CACHE_PATH = "data/cache/long_term.json"
DEFAULT_TTL_DAYS = 30
DEFAULT_THRESHOLD = 0.92
DEFAULT_MAX_ENTRIES = 500


class LongTermMemory:
    """
    Semantic cache cho kết quả RAG pháp lý.

    Lưu ý pháp lý:
    - Cache phải gắn corpus_version/index_version để tránh dùng câu trả lời cũ khi luật/index đổi.
    - Cache có TTL vì văn bản pháp luật có thể thay đổi.
    - Chỉ nên cache kết quả có citation và confidence đủ cao.
    """

    def __init__(
        self,
        cache_path: str = DEFAULT_CACHE_PATH,
        threshold: float = DEFAULT_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_days: int = DEFAULT_TTL_DAYS,
        corpus_version: str = "unknown",
        index_version: str = "unknown",
    ):
        self.cache_path = Path(cache_path)
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl = timedelta(days=ttl_days)
        self.corpus_version = corpus_version
        self.index_version = index_version
        self.cache: list[dict] = []
        self._load()

    def _load(self):
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.cache = data.get("entries", [])
            elif isinstance(data, list):
                # Backward compatibility với format cũ.
                self.cache = data
            else:
                self.cache = []
        except json.JSONDecodeError:
            backup_path = self.cache_path.with_suffix(".corrupt.json")
            self.cache_path.replace(backup_path)
            self.cache = []
            print(f"Long-term cache bị lỗi JSON, đã backup sang {backup_path}")

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corpus_version": self.corpus_version,
            "index_version": self.index_version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entries": self.cache,
        }
        tmp_path = self.cache_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.cache_path)

    def _similarity(self, v1: list[float], v2: list[float]) -> float:
        if len(v1) != len(v2):
            return 0.0
        a, b = np.array(v1), np.array(v2)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _is_expired(self, entry: dict) -> bool:
        created_at = entry.get("created_at")
        if not created_at:
            return True
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError:
            return True
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - created > self.ttl

    def _is_version_compatible(self, entry: dict) -> bool:
        return (
            entry.get("corpus_version") == self.corpus_version
            and entry.get("index_version") == self.index_version
        )

    def _is_cacheable_result(self, result: dict) -> bool:
        citations = result.get("relevant_articles") or result.get("citations") or []
        confidence = result.get("confidence", "high")
        if confidence == "low":
            return False
        return bool(citations)

    def lookup(
        self,
        query: str,
        query_vector: list[float],
    ) -> dict | None:
        best_score = 0.0
        best_entry = None

        fresh_entries = []
        for entry in self.cache:
            if self._is_expired(entry):
                continue
            if not self._is_version_compatible(entry):
                continue
            fresh_entries.append(entry)
            score = self._similarity(query_vector, entry.get("vector", []))
            if score > best_score:
                best_score = score
                best_entry = entry

        # Opportunistic cleanup stale entries.
        if len(fresh_entries) != len(self.cache):
            self.cache = fresh_entries
            self._save()

        if best_entry and best_score >= self.threshold:
            best_entry["hit_count"] = best_entry.get("hit_count", 0) + 1
            best_entry["last_hit_at"] = datetime.now(timezone.utc).isoformat()
            self._save()
            result = dict(best_entry["result"])
            result["cache_hit"] = True
            result["cache_similarity"] = best_score
            return result
        return None

    def store(
        self,
        query: str,
        query_vector: list[float],
        result: dict,
    ):
        if not self._is_cacheable_result(result):
            return

        for entry in self.cache:
            if not self._is_version_compatible(entry):
                continue
            if self._similarity(query_vector, entry.get("vector", [])) >= self.threshold:
                return

        self.cache.append({
            "question": query.strip(),
            "vector": query_vector,
            "result": result,
            "corpus_version": self.corpus_version,
            "index_version": self.index_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_hit_at": None,
            "hit_count": 0,
        })

        if len(self.cache) > self.max_entries:
            self.cache.sort(key=lambda x: x.get("hit_count", 0), reverse=True)
            self.cache = self.cache[:self.max_entries]
        self._save()

    def clear(self):
        self.cache = []
        self._save()

    @property
    def size(self) -> int:
        return len(self.cache)
