from __future__ import annotations

import logging
import os
import threading

import duckdb

from generative_vector_tile.datasets.base import Dataset
from generative_vector_tile.stac_index import StacBBox, get_stac_index

logger = logging.getLogger(__name__)

# One DuckDB instance, one cursor per in-flight query.
#
# spatial + httpfs SIGSEGV under concurrent queries on the SAME connection
# object (libgeos / libcurl global state), so every query still needs its
# own context -- but a `cursor()` is exactly that, and unlike a separate
# `duckdb.connect()` it shares the instance's external file cache.
#
# That cache is what makes remote Parquet bearable. Measured on one z16
# divisions tile (3 files, 31 rows): 65.2s cold, then 1.5s and 1.5s. The
# previous pool of N independent connections meant N independent caches, so
# at most one request in N saw a warm one and the rest paid the full S3 read
# again. Cursors are also cheap to throw away, which matters for the
# watchdog below: killing a cursor leaves the instance (and its warm cache)
# intact.
#
# Concurrency is bounded by a semaphore rather than by the number of
# connections, preserving the old DUCKDB_POOL_SIZE knob's meaning: how many
# tiles may hit DuckDB at once.
DEFAULT_MAX_CONCURRENT_QUERIES = 4


def _max_concurrent_queries() -> int:
    return max(1, int(os.environ.get("DUCKDB_POOL_SIZE", DEFAULT_MAX_CONCURRENT_QUERIES)))


# Aggressive default: a single tile against a bbox-pruned Parquet file
# should land in well under 5s once the connection is warm. Anything longer
# is either S3 weather or a hostile query, neither of which is improved by
# letting it run further. Override via DUCKDB_QUERY_TIMEOUT_S.
DEFAULT_QUERY_TIMEOUT_S = 8.0


class QueryTimeout(Exception):
    """The watchdog interrupted the query before it produced rows."""


def _query_timeout_s() -> float:
    return float(os.environ.get("DUCKDB_QUERY_TIMEOUT_S", DEFAULT_QUERY_TIMEOUT_S))


_instance: duckdb.DuckDBPyConnection | None = None
_instance_lock = threading.Lock()
_query_slots: threading.BoundedSemaphore | None = None


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


def _get_instance() -> duckdb.DuckDBPyConnection:
    global _instance, _query_slots
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            slots = _max_concurrent_queries()
            _query_slots = threading.BoundedSemaphore(slots)
            _instance = _build_connection()
            logger.info(
                "duckdb instance initialised: max_concurrent_queries=%d, "
                "external_file_cache=%s, memory_limit=%s, threads=%s",
                slots,
                _setting(_instance, "enable_external_file_cache"),
                _setting(_instance, "memory_limit"),
                _setting(_instance, "threads"),
            )
        return _instance


def _setting(con: duckdb.DuckDBPyConnection, name: str) -> str:
    try:
        return str(con.execute(f"SELECT current_setting('{name}')").fetchone()[0])
    except duckdb.Error:
        return "n/a"


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the shared instance for warmup / one-shot use.

    Callers must not run concurrent queries directly on it -- use
    `query_features`, which takes a cursor per query.
    """
    return _get_instance()


def query_features(
    dataset: Dataset,
    bbox: StacBBox,
    filter_sql: str | None,
    limit: int,
    timeout_s: float | None = None,
) -> list[dict]:
    """Run the dataset query within bbox, returning row dicts including a __wkb blob.

    `filter_sql` is a raw DuckDB boolean expression string (LLM output) or
    None for bbox-only. It is interpolated directly into the WHERE clause;
    per the threat model in ADR-0002, the data is public, the connection is
    read-only, and SSRF is blocked at the network layer, so the absence of
    a parser-level allowlist is acceptable. DuckDB's own parser fails closed
    on syntactically invalid expressions, which the caller surfaces as 400.

    Two source modes:
    - Overture (default): STAC bbox index → filtered S3 URLs → bbox struct WHERE
    - Direct URL (dataset.parquet_urls set): fixed URLs → ST_Intersects WHERE
    """
    if dataset.parquet_urls is not None:
        files = list(dataset.parquet_urls)
    else:
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

    if dataset.use_st_intersects:
        # Direct-URL datasets (e.g. Natural Earth) lack the GeoParquet bbox
        # struct. Use ST_Intersects with a tile envelope instead.
        sql_parts = [
            f"SELECT {select_columns}, ST_AsWKB({geom_expr}) AS __wkb",
            f"FROM read_parquet([{files_literal}])",
            f"WHERE ST_Intersects({geom_expr}, ST_MakeEnvelope(?, ?, ?, ?))",
        ]
        params: list[object] = [west, south, east, north]
    else:
        # Overture GeoParquet 1.1 bbox struct: DuckDB pushes the filter down
        # to row-group level, so only the relevant S3 byte ranges are fetched.
        sql_parts = [
            f"SELECT {select_columns}, ST_AsWKB({geom_expr}) AS __wkb",
            f"FROM read_parquet([{files_literal}])",
            "WHERE bbox.xmin <= ?",
            "  AND bbox.xmax >= ?",
            "  AND bbox.ymin <= ?",
            "  AND bbox.ymax >= ?",
        ]
        params = [east, west, north, south]

    if filter_sql:
        sql_parts.append(f"  AND ({filter_sql})")
    sql_parts.append(f"LIMIT {int(limit)}")
    sql = "\n".join(sql_parts)

    con = _get_instance()
    assert _query_slots is not None
    # Startup warmup passes its own, much larger budget: the whole point of
    # that query is to pay the cold read once, and the per-request cap would
    # kill it right before it finished populating the cache.
    timeout = _query_timeout_s() if timeout_s is None else timeout_s
    _query_slots.acquire()
    cur = con.cursor()
    # Watchdog cancels the in-flight query so a pathological filter cannot
    # hold the connection past Knative's request timeout and stall the pod.
    #
    # interrupt() must be called on the object that is running the query --
    # the cursor. Calling it on the parent connection (which is what this
    # used to do) either does nothing at all or corrupts the in-flight
    # parquet read into "TProtocolException: Invalid data"; both were
    # observed, and neither stopped the query. That is why divisions tiles
    # ran for 125s and then returned an empty tile labelled as a rejected
    # filter.
    fired = threading.Event()

    def _interrupt() -> None:
        fired.set()
        cur.interrupt()

    timer = threading.Timer(timeout, _interrupt)
    try:
        timer.start()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description]
        except duckdb.Error as e:
            if fired.is_set():
                raise QueryTimeout(
                    f"query exceeded {timeout:.1f}s "
                    f"(dataset={dataset.id}, files={len(files)})"
                ) from e
            raise
        finally:
            cur.close()
    finally:
        timer.cancel()
        # Only the cursor is discarded. Verified that interrupting one cursor
        # leaves sibling cursors and the instance untouched, so the warm file
        # cache survives a timeout instead of being thrown away with the
        # connection.
        _query_slots.release()

    logger.info(
        "duckdb: dataset=%s files=%d rows=%d filter=%r",
        dataset.id, len(files), len(rows), filter_sql,
    )
    return [dict(zip(columns, row, strict=False)) for row in rows]
