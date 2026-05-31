"""Download Natural Earth shapefiles and convert to GeoParquet.

Usage:
    uv run python scripts/build_natural_earth_parquet.py [target ...]

Targets (default: all):
    countries_110m   - 110m admin_0 countries (population / economics datasets)
    countries_50m    - 50m  admin_0 countries (higher quality polygons)
    admin1_10m       - 10m  admin_1 states & provinces

Output directory: /data/www/html/static/natural-earth/
Accessible at:    https://z.yuiseki.net/static/natural-earth/<filename>
"""

from __future__ import annotations

import os
import sys
import urllib.request

OUT_DIR = "/data/www/html/static/natural-earth"

TARGETS: dict[str, dict] = {
    "countries_110m": {
        "url": "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
        "out": "ne_110m_admin_0_countries.parquet",
        "keep_cols": [
            "NAME", "NAME_EN", "NAME_JA", "ADMIN", "ISO_A3", "ISO_A2",
            "CONTINENT", "REGION_UN", "SUBREGION", "REGION_WB",
            "POP_EST", "POP_RANK", "POP_YEAR",
            "GDP_MD", "GDP_YEAR", "ECONOMY", "INCOME_GRP",
            "SOVEREIGNT", "TYPE", "FORMAL_EN",
        ],
    },
    "countries_50m": {
        "url": "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip",
        "out": "ne_50m_admin_0_countries.parquet",
        "keep_cols": [
            "NAME", "NAME_EN", "NAME_JA", "ADMIN", "ISO_A3", "ISO_A2",
            "CONTINENT", "REGION_UN", "SUBREGION", "REGION_WB",
            "POP_EST", "POP_RANK", "POP_YEAR",
            "GDP_MD", "GDP_YEAR", "ECONOMY", "INCOME_GRP",
            "SOVEREIGNT", "TYPE", "FORMAL_EN",
        ],
    },
    "admin1_10m": {
        "url": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
        "out": "ne_10m_admin_1_states_provinces.parquet",
        "keep_cols": [
            "name", "name_en", "name_ja", "name_alt", "name_local",
            "admin", "iso_a2", "iso_3166_2", "adm0_a3",
            "type", "type_en", "region", "region_sub",
            "latitude", "longitude", "area_sqkm",
        ],
    },
}


def build_target(name: str) -> None:
    import geopandas as gpd

    cfg = TARGETS[name]
    url = cfg["url"]
    out_path = os.path.join(OUT_DIR, cfg["out"])
    keep_cols: list[str] = cfg["keep_cols"]

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "ne.zip")
        print(f"[{name}] Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[{name}]   -> {os.path.getsize(zip_path):,} bytes")
        gdf = gpd.read_file(zip_path)
        print(f"[{name}] Loaded: {len(gdf)} features, {len(gdf.columns)} columns")

    # Filter to columns that exist in this file (schema may vary slightly)
    available = [c for c in keep_cols if c in gdf.columns or c.upper() in gdf.columns]
    # Normalise case: prefer exact match, then upper
    cols_to_use: list[str] = []
    for c in keep_cols:
        if c in gdf.columns:
            cols_to_use.append(c)
        elif c.upper() in gdf.columns:
            cols_to_use.append(c.upper())
    cols_to_use.append("geometry")

    gdf_slim = gdf[cols_to_use].copy()
    gdf_slim.columns = [c.lower() for c in gdf_slim.columns]

    os.makedirs(OUT_DIR, exist_ok=True)
    gdf_slim.to_parquet(out_path, index=False)
    print(f"[{name}] Written: {out_path} ({os.path.getsize(out_path):,} bytes)")
    print(f"[{name}] URL: https://z.yuiseki.net/static/natural-earth/{cfg['out']}")


def main() -> None:
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(TARGETS.keys())
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        print(f"Unknown targets: {unknown}. Available: {list(TARGETS.keys())}")
        sys.exit(1)
    for t in targets:
        build_target(t)


if __name__ == "__main__":
    main()
