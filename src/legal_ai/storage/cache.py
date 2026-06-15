"""
Cache — Semantic cache + Exact cache cho RAG pipeline
Giảm LLM/Reranker calls cho câu hỏi lặp lại
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timezone


class ExactCache:
    """
    Exact match cache — hash(query) → result
    Dùng cho câu hỏi giống hệt nhau 100%
    """

    def __init__(self, path: str = "data/cache/exact.json"):
        self.path  = Path(path)
        self.store: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.store = json.load(f)

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, ensure_ascii=False, indent=2)

    def _key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> dict | None:
        entry = self.store.get(self._key(query))
        if entry:
            entry["hit_count"] += 1
            self._save()
            return entry["result"]
        return None

    def set(self, query: str, result: dict):
        self.store[self._key(query)] = {
            "query":      query,
            "result":     result,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hit_count":  0,
        }
        self._save()


class SemanticCache:
    """
    Semantic cache — embedding similarity → result
    Dùng cho câu hỏi tương tự nhau về nghĩa
    threshold mặc định 0.92 từ settings.yaml
    """

    def __init__(
        self,
        path:      str   = "data/cache/semantic.json",
        threshold: float = 0.92,
        max_size:  int   = 500,
    ):
        self.path      = Path(path)
        self.threshold = threshold
        self.max_size  = max_size
        self.entries:  list[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.entries = json.load(f)
            print(f"✅ Semantic cache loaded: {len(self.entries)} entries")

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def _sim(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

    def get(self, query_vector: list[float]) -> dict | None:
        best_score, best_entry = 0.0, None

        for entry in self.entries:
            score = self._sim(query_vector, entry["vector"])
            if score > best_score:
                best_score, best_entry = score, entry

        if best_score >= self.threshold and best_entry:
            print(f"💾 Semantic cache hit (score={best_score:.3f})")
            best_entry["hit_count"] += 1
            self._save()
            return best_entry["result"]

        return None

    def set(self, query: str, query_vector: list[float], result: dict):
        # Tránh duplicate
        for entry in self.entries:
            if self._sim(query_vector, entry["vector"]) >= self.threshold:
                return

        self.entries.append({
            "query":      query,
            "vector":     query_vector,
            "result":     result,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hit_count":  0,
        })

        # Evict ít dùng nhất khi đầy
        if len(self.entries) > self.max_size:
            self.entries.sort(key=lambda x: x["hit_count"], reverse=True)
            self.entries = self.entries[:self.max_size]

        self._save()

    def clear(self):
        self.entries = []
        self._save()


class CacheManager:
    """
    Unified cache — thử exact trước, semantic sau
    """

    def __init__(
        self,
        exact_path:    str   = "data/cache/exact.json",
        semantic_path: str   = "data/cache/semantic.json",
        threshold:     float = 0.92,
    ):
        self.exact    = ExactCache(exact_path)
        self.semantic = SemanticCache(semantic_path, threshold=threshold)

    def get(
        self,
        query:        str,
        query_vector: list[float] | None = None,
    ) -> dict | None:
        # 1. Exact match trước — nhanh nhất
        result = self.exact.get(query)
        if result:
            return result

        # 2. Semantic match
        if query_vector is not None:
            result = self.semantic.get(query_vector)
            if result:
                return result

        return None

    def set(
        self,
        query:        str,
        result:       dict,
        query_vector: list[float] | None = None,
    ):
        self.exact.set(query, result)
        if query_vector is not None:
            self.semantic.set(query, query_vector, result)