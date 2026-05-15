from __future__ import annotations

import logging
import os
import queue
import threading

import duckdb

from generative_vector_tile.datasets.base import Dataset
from generative_vector_tile.stac_index import StacBBox, get_stac_index

logger = logging.getLogger(__name__)

# spatial + httpfs SIGSEGV under concurrent queries on the SAME connection
# (libgeos / libcurl global state). Independent connections are fine: the
# buildings-tile study rates `duckdb.connect() per thread` as both
# thread-safe and high throughput. A small fixed pool lets tiles run truly
# in parallel instead of queuing behind a single _query_lock; this is the
# difference between "first tile 20s + linear queue growth" and "first
# tile 20s + 4-way parallel after that" for a 6+ tile pan burst.
DEFAULT_POOL_SIZE = 4


def _pool_size() -> int:
    return max(1, int(os.environ.get("DUCKDB_POOL_SIZE", DEFAULT_POOL_SIZE)))


# Aggressive default: a single tile against a bbox-pruned Parquet file
# should land in well under 5s once the connection is warm. Anything longer
# is either S3 weather or a hostile query, neither of which is improved by
# letting it run further. Override via DUCKDB_QUERY_TIMEOUT_S.
DEFAULT_QUERY_TIMEOUT_S = 8.0


def _query_timeout_s() -> float:
    return float(os.environ.get("DUCKDB_QUERY_TIMEOUT_S", DEFAULT_QUERY_TIMEOUT_S))


_pool: queue.Queue[duckdb.DuckDBPyConnection] | None = None
_pool_lock = threading.Lock()


def _build_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # Skip DuckDB's region auto-discovery (which adds a us-east-1 round-trip
    # per session). Overture's public bucket lives in us-west-2.
    con.execute("SET s3_region='us-west-2';")
    # Resource caps default to DuckDB's own (memory ~80% of RAM, threads =
    # num_cores). Hard-coded values of memory_limit=1GB / threads=4 crippled
    # local dev: the external_file_cache had nowhere to live and parquet
    # reads couldn't use the available cores. Production deployment should
    # set DUCKDB_MEMORY_LIMIT / DUCKDB_THREADS to match the pod's allocation.
    mem = os.environ.get("DUCKDB_MEMORY_LIMIT")
    threads = os.environ.get("DUCKDB_THREADS")
    if mem:
        con.execute(f"SET memory_limit='{mem}';")
    if threads:
        con.execute(f"SET threads={threads};")
    return con


def _get_pool() -> queue.Queue[duckdb.DuckDBPyConnection]:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        size = _pool_size()
        pool: queue.Queue[duckdb.DuckDBPyConnection] = queue.Queue(maxsize=size)
        for _ in range(size):
            pool.put(_build_connection())
        _pool = pool
        logger.info("duckdb pool initialised: size=%d", size)
        return pool


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return one connection for warmup / one-shot use.

    Pool is initialised on first call; the returned connection is one of
    the pool members. Callers must not run concurrent queries on the
    returned object -- use `query_features` for that.
    """
    pool = _get_pool()
    con = pool.get()
    pool.put(con)
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
    west, south, east, north = bbox

    # Filter on the GeoParquet `bbox` struct rather than ST_Intersects(geom,
    # envelope). The bbox columns are native parquet stats that DuckDB pushes
    # down for row-group pruning -- the actual S3 byte fetch becomes
    # proportional to features in the tile, not file size.
    sql_parts = [
        f"SELECT {select_columns}, ST_AsWKB({geom_expr}) AS __wkb",
        f"FROM read_parquet([{files_literal}])",
        "WHERE bbox.xmin <= ?",
        "  AND bbox.xmax >= ?",
        "  AND bbox.ymin <= ?",
        "  AND bbox.ymax >= ?",
    ]
    params: list[object] = [east, west, north, south]
    if filter_sql:
        sql_parts.append(f"  AND ({filter_sql})")
    sql_parts.append(f"LIMIT {int(limit)}")
    sql = "\n".join(sql_parts)

    pool = _get_pool()
    con = pool.get()
    timeout = _query_timeout_s()
    # Watchdog cancels the in-flight query via DuckDB's interrupt(). Without
    # this, a pathological filter could hold the connection past Knative's
    # request timeout and stall the pod.
    timer = threading.Timer(timeout, con.interrupt)
    try:
        timer.start()
        cur = con.cursor()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description]
        finally:
            cur.close()
    finally:
        timer.cancel()
        pool.put(con)

    logger.info(
        "duckdb: dataset=%s files=%d rows=%d filter=%r",
        dataset.id, len(files), len(rows), filter_sql,
    )
    return [dict(zip(columns, row, strict=False)) for row in rows]
