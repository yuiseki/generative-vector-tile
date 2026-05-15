"""In-memory LRU cache for (dataset_id, q) → SQL fragment.

Per-pod cache: gets warm during the lifetime of a Knative pod, drops on
scale-to-zero. Phase 3+ may layer a persistent cache (SQLite or Redis) on
top, but the per-pod layer is the cheapest hit and amortises the LLM cost
for any user who pans around the same map.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Final

from cachetools import LRUCache

DEFAULT_MAX_SIZE: Final = 5000


def _cache_key(dataset_id: str, q: str) -> str:
    normalised = q.strip()
    raw = f"{dataset_id}\x00{normalised}".encode()
    return hashlib.sha256(raw).hexdigest()


class FilterCache:
    def __init__(self, maxsize: int | None = None) -> None:
        size = maxsize if maxsize is not None else int(
            os.environ.get("FILTER_CACHE_MAX", DEFAULT_MAX_SIZE)
        )
        self._cache: LRUCache[str, str] = LRUCache(maxsize=size)
        self._lock = threading.Lock()

    def get(self, dataset_id: str, q: str) -> str | None:
        key = _cache_key(dataset_id, q)
        with self._lock:
            return self._cache.get(key)

    def put(self, dataset_id: str, q: str, sql_fragment: str) -> None:
        key = _cache_key(dataset_id, q)
        with self._lock:
            self._cache[key] = sql_fragment

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_DEFAULT_CACHE: FilterCache | None = None


def get_default_cache() -> FilterCache:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = FilterCache()
    return _DEFAULT_CACHE
