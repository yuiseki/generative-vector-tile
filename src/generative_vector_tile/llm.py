"""LLM call that translates a natural-language `q` into a DuckDB SQL
WHERE-clause fragment.

Uses the `chat.completions.parse` API (with response_format = JSON Schema)
rather than the newer Responses API, because llama.cpp's llama-server speaks
chat.completions out of the box but not Responses. The OpenAI Python SDK
treats both backends interchangeably as long as base_url + api_key are set.

The LLM emits a raw SQL string. Validation lives in the infrastructure layer
(gVisor + Knative NetworkPolicy + DuckDB resource caps), not here -- see
ADR-0002 for the reasoning. This module's job is just to be a thin, retryable
caller with a tight timeout.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel, Field

from generative_vector_tile.datasets.base import Column, Dataset

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.1"
# Aggressive default: a 2-4B local model on Apple Silicon should respond in
# 1-3s. Anything beyond 6s is almost certainly a model-load stall or a
# thinking-loop, neither of which gets faster by waiting longer. Override
# with LLM_TIMEOUT_S for slower backends.
DEFAULT_TIMEOUT_S = 6.0


class LlmUnavailable(RuntimeError):
    """Raised when no LLM backend is configured."""


class LlmTranslationError(RuntimeError):
    """Raised when the LLM call fails (timeout, API error, malformed output)."""


class FilterTranslation(BaseModel):
    """Structured output from the LLM: a single SQL WHERE-clause fragment.

    The schema is intentionally trivial -- a single string field -- because
    structured outputs guarantee the model emits exactly this shape, avoiding
    the markdown-fence / prose-prefix noise of plain text mode.
    """

    where_clause: str = Field(
        ...,
        description=(
            "A DuckDB SQL boolean expression suitable for use as a WHERE clause. "
            "Do NOT include the WHERE keyword or trailing semicolons. "
            "Examples: \"height >= 100\", \"class IN ('commercial', 'office') AND height >= 50\""
        ),
    )


@lru_cache(maxsize=1)
def _clients() -> tuple[OpenAI, ...]:
    """Build the OpenAI-compatible client pool once.

    Three intended deployments:
      - OpenAI cloud: OPENAI_API_KEY only. OPENAI_BASE_URL / OPENAI_BASE_URLS
        unset; the SDK uses api.openai.com.
      - Single OpenAI-compatible server: OPENAI_BASE_URL=http://host:port/v1
        and OPENAI_API_KEY=anything (server ignores it).
      - llama-server pool (round-robin across N instances):
        OPENAI_BASE_URLS=http://host:p1/v1,http://host:p2/v1,...

    OPENAI_BASE_URLS takes precedence over OPENAI_BASE_URL when both set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    urls_csv = os.environ.get("OPENAI_BASE_URLS")
    single_url = os.environ.get("OPENAI_BASE_URL")

    urls: list[str] = []
    if urls_csv:
        urls = [u.strip() for u in urls_csv.split(",") if u.strip()]
    elif single_url:
        urls = [single_url]

    if not urls and not api_key:
        raise LlmUnavailable(
            "None of OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_BASE_URLS is set; "
            "q-based filters cannot be translated"
        )

    effective_key = api_key or "dummy"
    if urls:
        clients = tuple(OpenAI(api_key=effective_key, base_url=u) for u in urls)
        logger.info("LLM client pool: %d endpoints", len(clients))
        return clients
    return (OpenAI(api_key=effective_key),)


_rr_counter = itertools.count()
_rr_lock = threading.Lock()


def _next_client() -> OpenAI:
    """Round-robin pick across the configured client pool."""
    pool = _clients()
    if len(pool) == 1:
        return pool[0]
    with _rr_lock:
        i = next(_rr_counter)
    return pool[i % len(pool)]


# Backwards-compatible accessor used by tests; returns the first client.
def _client() -> OpenAI:
    return _clients()[0]


_client.cache_clear = _clients.cache_clear  # type: ignore[attr-defined]


def _model_name() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def _timeout_s() -> float:
    return float(os.environ.get("LLM_TIMEOUT_S", DEFAULT_TIMEOUT_S))


def _format_column(c: Column) -> str:
    base = f"- {c.name} ({c.type}) sql_expr={c.sql_expr}"
    if c.enum_values:
        # Listing the legal enum values keeps the model from inventing
        # plausible-but-fictional category strings. The full list goes
        # inline because Qwen3 / gpt-5.1 attend to it reliably even at
        # 30-50 items, and the cost is amortised by automatic prompt
        # caching once the prompt has been seen.
        joined = ", ".join(c.enum_values)
        return f"{base}\n  allowed values: {joined}"
    return base


def build_system_prompt(dataset: Dataset) -> str:
    """Stable system prompt that describes the dataset's filterable columns.

    The dataset-dependent part lives in the system message so providers'
    automatic prompt caching can amortise its cost across repeated tiles.
    Only the user message (the q string) varies per request.
    """
    columns_block = "\n".join(_format_column(c) for c in dataset.filterable_columns)
    aliases_block = (
        "\n".join(f"- {jp} → {col}" for jp, col in dataset.filter_aliases.items())
        if dataset.filter_aliases
        else "(none)"
    )
    return f"""You translate a natural-language filter description into a DuckDB SQL boolean expression suitable for the WHERE clause of a vector tile query.

Dataset: {dataset.id} ({dataset.description})

Filterable columns (only these may appear in the expression):
{columns_block}

Japanese aliases:
{aliases_block}

Rules:
- Output ONLY the boolean expression body. Do NOT include the WHERE keyword, ORDER BY, LIMIT, semicolons, or any prose.
- Reference columns by their column name as listed above. Do not use aliases in SQL output -- map them yourself.
- String literals must be single-quoted, e.g. 'commercial'. Inside the literal, escape single quotes by doubling them.
- The user's filter is often Japanese, but column values are ALWAYS the English enum strings listed under "allowed values". Translate Japanese category names ("学校", "病院", "ホテル", ...) to the corresponding English enum value yourself; never compare a column against a Japanese string literal.
- Combine predicates with AND / OR / NOT. Use parentheses for grouping where precedence matters.
- Allowed comparison: = != < > <= >= IN BETWEEN LIKE ILIKE
- Allowed scalar functions: LOWER, UPPER, ABS, COALESCE
- Do NOT use subqueries, JOINs, CTEs, window functions, aggregate functions, COPY, INSTALL, LOAD, or any function not in the allowed list above.
- If the description is ambiguous, prefer the most permissive interpretation (the bbox already narrows results).
- The expression will be inserted into: SELECT geom, ... FROM read_parquet([...]) WHERE ST_Intersects(...) AND ( YOUR_EXPRESSION ) LIMIT N
"""


def translate_q(dataset: Dataset, q: str) -> str:
    """Translate a natural-language `q` into a DuckDB WHERE-clause fragment.

    Raises:
      LlmUnavailable: no LLM backend is configured.
      LlmTranslationError: LLM timed out, errored, or produced empty output.
    """
    try:
        client = _next_client()
    except LlmUnavailable:
        raise

    system_prompt = build_system_prompt(dataset)

    # Ollama-served thinking models (qwen3.5 etc.) burn tokens on a "reasoning"
    # field before emitting the JSON we asked for. Ollama lets the caller turn
    # this off via options.think=false on its native API; via the OpenAI-compat
    # path the same parameter rides along in extra_body. OpenAI cloud silently
    # ignores unknown body fields, so it's safe to send unconditionally.
    extra_body: dict = {"options": {"think": False}}

    try:
        response = client.chat.completions.parse(
            model=_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q.strip()},
            ],
            response_format=FilterTranslation,
            timeout=_timeout_s(),
            extra_body=extra_body,
        )
    except Exception as e:
        logger.warning("LLM call failed for q=%r on %s: %s", q, dataset.id, e)
        raise LlmTranslationError(str(e)) from e

    choice = response.choices[0] if response.choices else None
    if choice is None:
        raise LlmTranslationError("LLM returned no choices")
    if getattr(choice.message, "refusal", None):
        raise LlmTranslationError(f"LLM refused: {choice.message.refusal}")
    parsed = choice.message.parsed
    if parsed is None or not parsed.where_clause or not parsed.where_clause.strip():
        raise LlmTranslationError("LLM returned empty where_clause")
    return parsed.where_clause.strip()
