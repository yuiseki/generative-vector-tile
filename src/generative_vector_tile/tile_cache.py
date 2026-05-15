"""Tile-level result cache plus request coalescing.

After a tile has been computed once, subsequent identical requests should
not pay DuckDB-over-S3 cost again. After a tile request is in flight,
other identical concurrent requests should share the result instead of
each running their own DuckDB query. Both are per-pod, in-memory.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Final

from cachetools import LRUCache

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE: Final = 2000


def _key(dataset_id: str, z: int, x: int, y: int, filter_sql: str | None) -> str:
    return f"{dataset_id}\x00{z}\x00{x}\x00{y}\x00{filter_sql or ''}"


class TileCache:
    """Thread-safe LRU cache mapping tile key → raw MVT bytes."""

    def __init__(self, maxsize: int = DEFAULT_MAX_SIZE) -> None:
        self._cache: LRUCache[str, bytes] = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()

    def get(
        self, dataset_id: str, z: int, x: int, y: int, filter_sql: str | None
    ) -> bytes | None:
        with self._lock:
            return self._cache.get(_key(dataset_id, z, x, y, filter_sql))

    def put(
        self,
        dataset_id: str,
        z: int,
        x: int,
        y: int,
        filter_sql: str | None,
        mvt_bytes: bytes,
    ) -> None:
        with self._lock:
            self._cache[_key(dataset_id, z, x, y, filter_sql)] = mvt_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


class RequestCoalescer:
    """Deduplicates concurrent computations for the same key.

    When N threads ask for the same key simultaneously and the cache is
    cold, only one of them actually runs `compute_fn`; the others wait on
    an Event and pick up the cached value once the leader writes it.

    If the leader raises, followers re-raise the same exception so a flaky
    compute doesn't silently return a stale or empty result.
    """

    def __init__(self) -> None:
        self._inflight: dict[str, threading.Event] = {}
        self._errors: dict[str, BaseException] = {}
        self._lock = threading.Lock()

    def get_or_compute(
        self,
        dataset_id: str,
        z: int,
        x: int,
        y: int,
        filter_sql: str | None,
        cache: TileCache,
        compute_fn: Callable[[], bytes],
    ) -> bytes:
        cached = cache.get(dataset_id, z, x, y, filter_sql)
        if cached is not None:
            return cached

        key = _key(dataset_id, z, x, y, filter_sql)
        wait_event: threading.Event | None = None
        leader = False
        with self._lock:
            if key in self._inflight:
                wait_event = self._inflight[key]
            else:
                wait_event = threading.Event()
                self._inflight[key] = wait_event
                leader = True

        if not leader:
            assert wait_event is not None
            wait_event.wait()
            err = self._errors.get(key)
            if err is not None:
                raise err
            cached = cache.get(dataset_id, z, x, y, filter_sql)
            if cached is not None:
                return cached
            # Leader finished but cache lookup missed (e.g. leader put bytes
            # but cache evicted them before this follower woke up). Fall
            # through and compute again rather than return empty.
            logger.warning(
                "coalesce: leader done but cache miss for %s/%d/%d/%d, recomputing",
                dataset_id, z, x, y,
            )
            return compute_fn()

        try:
            result = compute_fn()
            cache.put(dataset_id, z, x, y, filter_sql, result)
            return result
        except BaseException as e:
            with self._lock:
                self._errors[key] = e
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                # Hold onto _errors until the event is set so followers can
                # observe the failure; clear afterwards in a separate step.
            wait_event.set()
            with self._lock:
                self._errors.pop(key, None)


_DEFAULT_TILE_CACHE: TileCache | None = None
_DEFAULT_COALESCER: RequestCoalescer | None = None


def get_default_tile_cache() -> TileCache:
    global _DEFAULT_TILE_CACHE
    if _DEFAULT_TILE_CACHE is None:
        _DEFAULT_TILE_CACHE = TileCache()
    return _DEFAULT_TILE_CACHE


def get_default_coalescer() -> RequestCoalescer:
    global _DEFAULT_COALESCER
    if _DEFAULT_COALESCER is None:
        _DEFAULT_COALESCER = RequestCoalescer()
    return _DEFAULT_COALESCER
