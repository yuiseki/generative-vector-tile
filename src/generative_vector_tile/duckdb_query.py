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

# Aggressive default: a single tile against a bbox-pruned Parquet file
# should land in well under 5s once the connection is warm. Anything longer
# is either S3 weather or a hostile query, neither of which is improved by
# letting it run further. Override via DUCKDB_QUERY_TIMEOUT_S.
DEFAULT_QUERY_TIMEOUT_S = 8.0


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
        # Skip DuckDB's region auto-discovery (which makes an extra
        # us-east-1 round-trip per session). Overture's public bucket lives
        # in us-west-2.
        con.execute("SET s3_region='us-west-2';")

        # Lean on every layer of DuckDB's built-in caching:
        # - enable_external_file_cache: chunk-level in-memory cache for
        #   httpfs reads (default on in 1.5.x). Adjacent tiles reading
        #   overlapping row groups in the same parquet file hit RAM.
        # - parquet_metadata_cache: caches parquet footer + row group stats
        #   so re-querying the same file skips the metadata refetch.
        # - enable_http_metadata_cache: caches HEAD responses so re-reading
        #   the same URL skips a network round-trip.
        # - validate_external_file_cache=NO_VALIDATION: Overture releases
        #   are immutable, so the cache never needs revalidation. With the
        #   default VALIDATE_ALL we'd re-HEAD remote files on every lookup.
        con.execute("SET enable_external_file_cache=true;")
        con.execute("SET parquet_metadata_cache=true;")
        con.execute("SET enable_http_metadata_cache=true;")
        con.execute("SET validate_external_file_cache='NO_VALIDATION';")

        # Resource caps default to DuckDB's own (memory ~80% of system RAM,
        # threads = num_cores). Earlier hard-coded values of memory_limit=1GB
        # / threads=4 were defensive for a Knative pod with 2Gi limit but
        # crippled local dev: the external_file_cache had nowhere to live and
        # parquet reads couldn't use the available cores. The Mac PoC sees a
        # ~100x speedup just from leaving these alone vs the previous limits.
        # Production deployment should set DUCKDB_MEMORY_LIMIT / DUCKDB_THREADS
        # explicitly to match the pod's allocation.
        mem = os.environ.get("DUCKDB_MEMORY_LIMIT")
        threads = os.environ.get("DUCKDB_THREADS")
        if mem:
            con.execute(f"SET memory_limit='{mem}';")
        if threads:
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
    west, south, east, north = bbox

    # Filter on the GeoParquet `bbox` struct rather than ST_Intersects(geom,
    # envelope). The bbox columns are native parquet stats that DuckDB pushes
    # down for row-group pruning -- the actual S3 byte fetch becomes
    # proportional to features in the tile, not file size. Buildings-tile
    # study measured this as the single biggest perf win for Overture.
    # Geometries whose bbox overlaps the tile may sometimes not actually
    # intersect, but the MVT clipper takes care of trimming polygons to tile
    # bounds at encode time, so the overshoot is harmless.
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
            cur.execute(sql, params)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description]
        finally:
            timer.cancel()
            cur.close()

    logger.info(
        "duckdb: dataset=%s files=%d rows=%d filter=%r",
        dataset.id, len(files), len(rows), filter_sql,
    )
    return [dict(zip(columns, row, strict=False)) for row in rows]
