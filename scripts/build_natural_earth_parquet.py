"""Download Natural Earth 110m countries shapefile and convert to GeoParquet.

Usage:
    uv run python scripts/build_natural_earth_parquet.py

Output:
    /data/www/html/static/natural-earth/ne_110m_admin_0_countries.parquet

The file is then accessible at:
    https://z.yuiseki.net/static/natural-earth/ne_110m_admin_0_countries.parquet

Columns retained (lowercased):
    geometry                   - country polygon (EPSG:4326)
    name, name_en, name_ja     - country names
    admin, iso_a3, iso_a2      - identifiers
    continent, region_un, subregion, region_wb  - geographic regions
    pop_est, pop_rank, pop_year                 - population statistics
    gdp_md, gdp_year, economy, income_grp       - economic statistics
    sovereignt, type, formal_en                 - political status
"""

import os
import urllib.request
import zipfile
import tempfile

SHP_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
OUT_PATH = "/data/www/html/static/natural-earth/ne_110m_admin_0_countries.parquet"

KEEP_COLS = [
    "geometry",
    "NAME", "NAME_EN", "NAME_JA", "ADMIN", "ISO_A3", "ISO_A2",
    "CONTINENT", "REGION_UN", "SUBREGION", "REGION_WB",
    "POP_EST", "POP_RANK", "POP_YEAR",
    "GDP_MD", "GDP_YEAR", "ECONOMY", "INCOME_GRP",
    "SOVEREIGNT", "TYPE", "FORMAL_EN",
]


def main() -> None:
    import geopandas as gpd

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "ne_110m.zip")
        print(f"Downloading {SHP_URL} ...")
        urllib.request.urlretrieve(SHP_URL, zip_path)
        print(f"  -> {os.path.getsize(zip_path):,} bytes")

        gdf = gpd.read_file(zip_path)
        print(f"Loaded: {len(gdf)} features, {len(gdf.columns)} columns")

    gdf_slim = gdf[KEEP_COLS].copy()
    gdf_slim.columns = [c.lower() for c in gdf_slim.columns]
    print(f"Columns after slim: {list(gdf_slim.columns)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    gdf_slim.to_parquet(OUT_PATH, index=False)
    print(f"Written: {OUT_PATH} ({os.path.getsize(OUT_PATH):,} bytes)")
    print(f"Accessible at: https://z.yuiseki.net/static/natural-earth/ne_110m_admin_0_countries.parquet")


if __name__ == "__main__":
    main()
