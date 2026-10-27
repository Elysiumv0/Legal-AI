"""
Cache — Semantic cache + Exact cache cho RAG pipeline
Giảm LLM/Reranker calls cho câu hỏi lặp lại
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone


class ExactCache:
    """
    Exact match cache — hash(query) → result
    Dùng cho câu hỏi giống hệt nhau 100%
    """

    def __init__(
        self,
        path: str = "data/cache/exact.json",
        ttl_days: int | None = None,
        corpus_version: str | None = None,
    ):
        self.path  = Path(path)
        self.store: dict[str, dict] = {}
        self.ttl = timedelta(days=ttl_days) if ttl_days else None
        self.corpus_version = corpus_version
        self._load()
        self._evict_stale()

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

    def _is_stale(self, entry: dict) -> bool:
        """Entry hết hạn hoặc thuộc corpus/index khác phải bị loại.

        Không có guard này thì sau khi rebuild index hoặc đổi embedding model,
        cache vẫn trả answer sinh ra trên dữ liệu cũ — vốn là rủi ro pháp lý.
        """
        if self.corpus_version and entry.get("corpus_version") not in (
            None, self.corpus_version
        ):
            return True
        if self.ttl:
            created = entry.get("created_at")
            if not created:
                return True
            try:
                ts = datetime.fromisoformat(created)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return datetime.now(timezone.utc) - ts > self.ttl
            except ValueError:
                return True
        return False

    def _evict_stale(self) -> None:
        if not self.ttl and not self.corpus_version:
            return
        stale = [k for k, e in self.store.items() if self._is_stale(e)]
        for k in stale:
            del self.store[k]
        if stale:
            self._save()

    def get(self, query: str) -> dict | None:
        entry = self.store.get(self._key(query))
        if entry and not self._is_stale(entry):
            entry["hit_count"] += 1
            self._save()
            return entry["result"]
        if entry:
            del self.store[self._key(query)]
            self._save()
        return None

    def set(self, query: str, result: dict):
        self.store[self._key(query)] = {
            "query":      query,
            "result":     result,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_version": self.corpus_version,
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
        # Khác chiều (đổi embedding model) phải trả 0.0, không ném ValueError.
        # Trước đây np.dot(va, vb) với len khác nhau raise, và vì entries đọc
        # từ đĩa nên chỉ cần đổi model là lần get() đầu crash cả pipeline.
        if len(a) != len(b):
            return 0.0
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / (denom + 1e-9))

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

        new_entry = {
            "query":      query,
            "vector":     query_vector,
            "result":     result,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hit_count":  0,
        }
        self.entries.append(new_entry)

        # Evict ít dùng nhất khi đầy — nhưng KHÔNG cho entry vừa thêm vào vòng.
        # Entry mới hit_count=0 nên luôn đứng cuối khi sort; cache đầy + entry
        # cũ có hit sẽ loại ngay entry mới, cache đóng băng ở bộ khởi tạo.
        if len(self.entries) > self.max_size:
            others = [e for e in self.entries if e is not new_entry]
            others.sort(key=lambda x: x.get("hit_count", 0), reverse=True)
            self.entries = others[: self.max_size - 1] + [new_entry]

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