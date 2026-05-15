"""Tile-level result cache plus request coalescing.

After a tile has been computed once, subsequent identical requests should
not pay DuckDB-over-S3 cost again. After a tile request is in flight,
other identical concurrent requests should share the result instead of
each running their own DuckDB query.

The cache is two-layered: an in-memory LRU for hot tiles and a disk
write-through so that a server restart -- or a different gvt process
hitting the same machine -- doesn't have to redo a 160s DuckDB-over-S3
query. The disk layout is one file per tile under

    <base>/<dataset>/<z>/<x>/<y>/<filter_hash>.mvt

so individual entries can be inspected, deleted by zoom level, or shipped
to another host with `cp -r`. Empty filters (q=None) use the literal
filename `_nofilter.mvt`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Final

from cachetools import LRUCache

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE: Final = 2000


def _key(dataset_id: str, z: int, x: int, y: int, filter_sql: str | None) -> str:
    return f"{dataset_id}\x00{z}\x00{x}\x00{y}\x00{filter_sql or ''}"


def _default_disk_dir() -> Path:
    base = os.environ.get("TILE_CACHE_DIR")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "generative-vector-tile" / "tiles"


def _filter_filename(filter_sql: str | None) -> str:
    if not filter_sql:
        return "_nofilter.mvt"
    # 16 hex chars (64 bits) is more than enough for a per-(z,x,y) namespace.
    h = hashlib.sha256(filter_sql.encode("utf-8")).hexdigest()[:16]
    return f"{h}.mvt"


class TileCache:
    """Two-layer cache (in-memory LRU + on-disk write-through) mapping tile
    key → raw MVT bytes.

    Disk writes are best-effort: any I/O exception is logged but never
    propagated to the request path, so a full disk / permission error
    degrades us to in-memory-only rather than failing the tile request.
    """

    def __init__(
        self,
        maxsize: int = DEFAULT_MAX_SIZE,
        disk_dir: Path | None = None,
    ) -> None:
        self._cache: LRUCache[str, bytes] = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()
        self._disk_dir = disk_dir if disk_dir is not None else _default_disk_dir()
        try:
            self._disk_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("tile-cache disk dir unavailable (%s); memory-only", e)
            self._disk_dir = None  # type: ignore[assignment]

    def _disk_path(
        self, dataset_id: str, z: int, x: int, y: int, filter_sql: str | None
    ) -> Path | None:
        if self._disk_dir is None:
            return None
        return (
            self._disk_dir
            / dataset_id
            / str(z)
            / str(x)
            / str(y)
            / _filter_filename(filter_sql)
        )

    def get(
        self, dataset_id: str, z: int, x: int, y: int, filter_sql: str | None
    ) -> bytes | None:
        k = _key(dataset_id, z, x, y, filter_sql)
        with self._lock:
            hit = self._cache.get(k)
        if hit is not None:
            return hit
        # Memory miss → disk lookup. Read outside the lock so concurrent disk
        # hits don't serialize, then warm the memory cache for next time.
        path = self._disk_path(dataset_id, z, x, y, filter_sql)
        if path is None or not path.exists():
            return None
        try:
            data = path.read_bytes()
        except OSError as e:
            logger.warning("tile-cache disk read failed for %s: %s", path, e)
            return None
        with self._lock:
            self._cache[k] = data
        logger.info(
            "tile cache HIT (disk) %s/%d/%d/%d filter=%s",
            dataset_id, z, x, y, _filter_filename(filter_sql),
        )
        return data

    def put(
        self,
        dataset_id: str,
        z: int,
        x: int,
        y: int,
        filter_sql: str | None,
        mvt_bytes: bytes,
    ) -> None:
        k = _key(dataset_id, z, x, y, filter_sql)
        with self._lock:
            self._cache[k] = mvt_bytes
        path = self._disk_path(dataset_id, z, x, y, filter_sql)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(mvt_bytes)
            tmp.replace(path)
        except OSError as e:
            logger.warning("tile-cache disk write failed for %s: %s", path, e)

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
