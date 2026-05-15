from __future__ import annotations

import logging
import os
import threading

import duckdb

from generative_vector_tile.datasets.base import Dataset
from generative_vector_tile.stac_index import StacBBox, get_stac_index

logger = logging.getLogger(__name__)

# Concurrent queries against a single duckdb connection over `spatial` +
# `httpfs` SIGSEGV inside libgeos/libcurl. The buildings-tile study landed on
# a process-level lock for PoC scale; horizontal scaling lives in Knative
# autoscaling instead of in-process concurrency.
_query_lock = threading.Lock()
_connection: duckdb.DuckDBPyConnection | None = None

DEFAULT_MEMORY_LIMIT = "1GB"
DEFAULT_THREADS = "4"
DEFAULT_QUERY_TIMEOUT_S = 15.0


def _query_timeout_s() -> float:
    return float(os.environ.get("DUCKDB_QUERY_TIMEOUT_S", DEFAULT_QUERY_TIMEOUT_S))


def get_connection() -> duckdb.DuckDBPyConnection:
    global _connection
    if _connection is not None:
        return _connection
    with _query_lock:
        if _connection is not None:
            return _connection
        con = duckdb.connect(database=":memory:")
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("INSTALL httpfs; LOAD httpfs;")

        # Per-query resource caps. memory_limit bounds a single query so a
        # pathological filter cannot push the pod into OOMKilled. threads is
        # capped so concurrent tile requests don't oversubscribe the pod's
        # CPU allocation. See ADR-0002 "安全性の設計".
        mem = os.environ.get("DUCKDB_MEMORY_LIMIT", DEFAULT_MEMORY_LIMIT)
        threads = os.environ.get("DUCKDB_THREADS", DEFAULT_THREADS)
        con.execute(f"SET memory_limit='{mem}';")
        con.execute(f"SET threads={threads};")

        _connection = con
        return con


def query_features(
    dataset: Dataset,
    bbox: StacBBox,
    filter_sql: str | None,
    limit: int,
) -> list[dict]:
    """Run the dataset query within bbox, returning row dicts including a __wkb blob.

    `filter_sql` is a raw DuckDB boolean expression string (LLM output) or
    None for bbox-only. It is interpolated directly into the WHERE clause;
    per the threat model in ADR-0002, the data is public, the connection is
    read-only, and SSRF is blocked at the network layer, so the absence of
    a parser-level allowlist is acceptable. DuckDB's own parser fails closed
    on syntactically invalid expressions, which the caller surfaces as 400.
    """
    index = get_stac_index(
        dataset.overture_release, dataset.overture_theme, dataset.overture_type
    )
    files = index.files_for_bbox(bbox)
    if not files:
        return []

    select_columns = ", ".join(
        f"{c.sql_expr} AS {c.name}" for c in dataset.projected_columns
    )
    geom_expr = dataset.geometry_column.sql_expr
    files_literal = ", ".join(f"'{f}'" for f in files)

    where_clauses = [
        f"ST_Intersects({geom_expr}, ST_MakeEnvelope({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}))"
    ]
    if filter_sql:
        where_clauses.append(f"({filter_sql})")
    where = " AND ".join(where_clauses)

    sql = (
        f"SELECT {select_columns}, ST_AsWKB({geom_expr}) AS __wkb "
        f"FROM read_parquet([{files_literal}]) "
        f"WHERE {where} "
        f"LIMIT {int(limit)}"
    )

    con = get_connection()
    timeout = _query_timeout_s()
    # Watchdog cancels the in-flight query via DuckDB's interrupt(). Without
    # this, a pathological filter could hold the connection lock past Knative's
    # request timeout and stall the pod.
    timer = threading.Timer(timeout, con.interrupt)
    with _query_lock:
        cur = con.cursor()
        try:
            timer.start()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description]
        finally:
            timer.cancel()
            cur.close()

    return [dict(zip(columns, row, strict=False)) for row in rows]
