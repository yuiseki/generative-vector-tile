"""Compile a natural-language `q` into a DuckDB SQL WHERE-clause fragment.

The pipeline is intentionally minimal -- LLM + LRU cache -- because all
safety (against SSRF, resource abuse, kernel escape) lives in the infra
layer per ADR-0002. No regex fast-path, no AST allowlist.

Public surface:
    compile_filter(dataset, q) -> str | None     # None when q is empty
    CompileFilterError
"""

from __future__ import annotations

import logging
import threading

from generative_vector_tile.cache import FilterCache, get_default_cache
from generative_vector_tile.datasets.base import Dataset
from generative_vector_tile.llm import (
    LlmTranslationError,
    LlmUnavailable,
    translate_q,
)

logger = logging.getLogger(__name__)


class CompileFilterError(RuntimeError):
    """Raised when a non-empty `q` cannot be translated.

    Wraps the underlying cause so callers can distinguish "LLM not configured"
    from "LLM produced bad output" via the exception chain.
    """


# In-flight coalescing for LLM translation calls. When the browser pans, 20+
# tile requests can arrive simultaneously with the same q. Without this, each
# request makes its own LLM call (wasted compute + slower queue). With this,
# only the leader calls; followers wait for the leader's event and read from
# the cache.
_inflight: dict[tuple[str, str], threading.Event] = {}
_inflight_errors: dict[tuple[str, str], BaseException] = {}
_inflight_lock = threading.Lock()


def compile_filter(
    dataset: Dataset,
    q: str | None,
    *,
    cache: FilterCache | None = None,
) -> str | None:
    if q is None:
        return None
    text = q.strip()
    if not text:
        return None

    # `cache or get_default_cache()` would silently swap in the default cache
    # whenever the passed instance happens to be empty, because __len__
    # makes empty FilterCache instances falsy. Compare to None instead.
    if cache is None:
        cache = get_default_cache()
    cached = cache.get(dataset.id, text)
    if cached is not None:
        logger.debug("filter cache hit dataset=%s q=%r", dataset.id, text)
        return cached

    key = (dataset.id, text)
    leader = False
    wait_event: threading.Event | None = None
    with _inflight_lock:
        if key in _inflight:
            wait_event = _inflight[key]
        else:
            wait_event = threading.Event()
            _inflight[key] = wait_event
            leader = True

    if not leader:
        assert wait_event is not None
        wait_event.wait()
        err = _inflight_errors.get(key)
        if err is not None:
            if isinstance(err, LlmUnavailable):
                raise CompileFilterError(f"LLM not configured: {err}") from err
            raise CompileFilterError(f"LLM translation failed: {err}") from err
        cached = cache.get(dataset.id, text)
        if cached is not None:
            return cached
        # Cache evicted between leader writing and follower waking. Fall
        # through to compute again rather than serve a wrong value.

    try:
        sql = translate_q(dataset, text)
        cache.put(dataset.id, text, sql)
        logger.info("filter cache write dataset=%s q=%r sql=%r", dataset.id, text, sql)
        return sql
    except LlmUnavailable as e:
        with _inflight_lock:
            _inflight_errors[key] = e
        raise CompileFilterError(f"LLM not configured: {e}") from e
    except LlmTranslationError as e:
        with _inflight_lock:
            _inflight_errors[key] = e
        raise CompileFilterError(f"LLM translation failed: {e}") from e
    finally:
        if leader:
            with _inflight_lock:
                _inflight.pop(key, None)
            wait_event.set()
            with _inflight_lock:
                _inflight_errors.pop(key, None)


__all__ = ["compile_filter", "CompileFilterError"]
