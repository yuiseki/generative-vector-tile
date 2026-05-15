from __future__ import annotations

from typing import Any

import mapbox_vector_tile
import mercantile
from shapely import wkb

from generative_vector_tile.datasets.base import Dataset


def tile_to_bbox(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    b = mercantile.bounds(x, y, z)
    return (b.west, b.south, b.east, b.north)


def encode_mvt(
    dataset: Dataset,
    z: int,
    x: int,
    y: int,
    rows: list[dict[str, Any]],
) -> bytes:
    """Encode DuckDB result rows into MVT bytes for this tile.

    Geometries arrive as WKB blobs in the `__wkb` column (set by the SQL
    template in duckdb_query). Non-geometry projected columns become MVT
    feature properties.
    """
    features = []
    for row in rows:
        wkb_bytes = row.get("__wkb")
        if wkb_bytes is None:
            continue
        geom = wkb.loads(bytes(wkb_bytes))
        props = {
            k: v
            for k, v in row.items()
            if k != "__wkb" and v is not None
        }
        features.append({"geometry": geom.__geo_interface__, "properties": props})

    tile_bounds = mercantile.bounds(x, y, z)
    quantize_bounds = (
        tile_bounds.west,
        tile_bounds.south,
        tile_bounds.east,
        tile_bounds.north,
    )

    return mapbox_vector_tile.encode(
        [{"name": dataset.mvt_layer_name, "features": features}],
        quantize_bounds=quantize_bounds,
    )
