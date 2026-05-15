"""generative-vector-tile: natural-language parametrized dynamic vector tile FaaS.

Endpoints:
    GET /health                          → {"ok": true}
    GET /datasets                        → list registered datasets
    GET /tile/{dataset}/{z}/{x}/{y}.mvt  → MVT bytes
        Query parameters:
            q       natural-language filter (optional)
            limit   max features per tile (default 5000)

Run: `uv run python -m generative_vector_tile.server`
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import duckdb
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from generative_vector_tile.cache import get_default_cache
from generative_vector_tile.datasets import get_dataset, list_datasets
from generative_vector_tile.duckdb_query import get_connection, query_features
from generative_vector_tile.filters import CompileFilterError, compile_filter
from generative_vector_tile.llm import LlmUnavailable
from generative_vector_tile.mvt import encode_mvt, tile_to_bbox
from generative_vector_tile.stac_index import get_stac_index
from generative_vector_tile.tile_cache import (
    get_default_coalescer,
    get_default_tile_cache,
)

logger = logging.getLogger("generative-vector-tile")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MVT_MEDIA_TYPE = "application/vnd.mapbox-vector-tile"
ZOOM_MIN = 0
ZOOM_MAX = 22
LIMIT_MAX = 50000


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm DuckDB extensions and STAC indexes for registered datasets so the
    # first request lands on a hot worker.
    try:
        get_connection()
        for ds in list_datasets():
            get_stac_index(
                ds.overture_release, ds.overture_theme, ds.overture_type
            ).build()
    except Exception:
        logger.exception("startup warmup failed")
    yield


app = FastAPI(
    title="generative-vector-tile",
    version="0.1.0",
    description="Generative dynamic vector tile FaaS: natural-language filter parameters over Overture GeoParquet via DuckDB Spatial.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/datasets")
def datasets() -> dict:
    return {
        "datasets": [
            {
                "id": d.id,
                "description": d.description,
                "overture_release": d.overture_release,
                "overture_theme": d.overture_theme,
                "overture_type": d.overture_type,
                "filterable_columns": [c.name for c in d.filterable_columns],
            }
            for d in list_datasets()
        ]
    }


@app.get("/tile/{dataset_id}/{z}/{x}/{y}.mvt")
def tile(
    dataset_id: str,
    z: int,
    x: int,
    y: int,
    q: str | None = Query(default=None, max_length=256),
    limit: int = Query(default=5000, ge=1, le=LIMIT_MAX),
) -> Response:
    try:
        dataset = get_dataset(dataset_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if not (ZOOM_MIN <= z <= ZOOM_MAX):
        raise HTTPException(status_code=400, detail=f"zoom {z} out of range")

    try:
        filter_sql = compile_filter(dataset, q)
    except CompileFilterError as e:
        # Distinguish "service not configured" from "translation failed".
        # LlmUnavailable is a config / operations issue and should surface
        # loudly (503) so the operator notices.
        if isinstance(e.__cause__, LlmUnavailable):
            raise HTTPException(status_code=503, detail=str(e)) from e
        # LlmTranslationError (timeout, refusal, empty output) is per-request.
        # Don't 500 the browser -- return an empty MVT so MapLibre's tile
        # loading doesn't hang. The browser sees a successful empty layer
        # and the user can retry with a different phrasing.
        logger.warning(
            "tile LLM error %s/%d/%d/%d q=%r: %s",
            dataset_id, z, x, y, q, e,
        )
        return Response(
            content=b"",
            media_type=MVT_MEDIA_TYPE,
            headers={
                "Cache-Control": "no-store",
                "X-Tile-Cache": "error",
                "X-Tile-Error": "llm-translation-failed",
            },
        )

    bbox = tile_to_bbox(z, x, y)
    cache = get_default_tile_cache()
    coalescer = get_default_coalescer()

    # Check cache before announcing the work; cheap path for repeat tiles.
    cached = cache.get(dataset.id, z, x, y, filter_sql)
    if cached is not None:
        logger.info(
            "tile cache HIT %s/%d/%d/%d q=%r", dataset.id, z, x, y, q
        )
        return Response(
            content=cached,
            media_type=MVT_MEDIA_TYPE,
            headers={"Cache-Control": "public, max-age=300", "X-Tile-Cache": "hit"},
        )

    def _compute() -> bytes:
        logger.info(
            "tile compute %s/%d/%d/%d q=%r filter_sql=%r",
            dataset.id, z, x, y, q, filter_sql,
        )
        rows = query_features(dataset, bbox, filter_sql, limit)
        return encode_mvt(dataset, z, x, y, rows)

    # Coalesce concurrent identical misses so 20+ tile fetches for the same
    # (tile, filter) only pay one DuckDB+S3 round-trip.
    try:
        mvt_bytes = coalescer.get_or_compute(
            dataset.id, z, x, y, filter_sql, cache, _compute
        )
    except duckdb.Error as e:
        # The LLM produced SQL that DuckDB rejects (parse error, unknown
        # function, etc.). Don't 500 the browser -- it'd leave blank tiles
        # forever. Instead: drop the bad filter from the cache so the next
        # request gets a fresh LLM call, and return an empty MVT so MapLibre
        # finishes loading the tile gracefully.
        logger.warning(
            "tile DuckDB error %s/%d/%d/%d q=%r filter_sql=%r: %s",
            dataset.id, z, x, y, q, filter_sql, e,
        )
        if q:
            get_default_cache().invalidate(dataset.id, q.strip())
        return Response(
            content=b"",
            media_type=MVT_MEDIA_TYPE,
            headers={
                "Cache-Control": "no-store",
                "X-Tile-Cache": "error",
                "X-Tile-Error": "duckdb-rejected-filter",
            },
        )

    return Response(
        content=mvt_bytes,
        media_type=MVT_MEDIA_TYPE,
        headers={"Cache-Control": "public, max-age=300", "X-Tile-Cache": "miss"},
    )


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
