"""STAC catalog → in-memory spatial index of Overture Parquet file URLs.

Overture partitions by ID, so the S3 path tells you nothing about geography.
DuckDB's wildcard `read_parquet('s3://.../*')` would fetch every file's
footer (~1-2 min cold start). Overture publishes a STAC catalog that lists
each Parquet file's bbox, so we fetch the catalog once at boot and resolve
`(theme, type, bbox)` → `list[s3_url]` in-memory per request.

Phase 1 keeps this minimal: STAC fetch + linear bbox scan. The buildings-tile
study showed linear scan over ~512 items is fast enough for sub-millisecond
lookups; an actual R-tree adds complexity without measurable gain at this size.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

STAC_ROOT = "https://stac.overturemaps.org"
StacBBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class StacItem:
    href: str
    bbox: StacBBox


class StacIndex:
    """Lazy STAC-backed file index for a single Overture (theme, type)."""

    def __init__(self, release: str, theme: str, type_: str) -> None:
        self._release = release
        self._theme = theme
        self._type = type_
        self._items: list[StacItem] | None = None
        self._lock = threading.Lock()

    def build(self) -> None:
        with self._lock:
            if self._items is not None:
                return
            collection_url = (
                f"{STAC_ROOT}/{self._release}/{self._theme}/{self._type}/collection.json"
            )
            logger.info("STAC fetch: %s", collection_url)
            with httpx.Client(timeout=30.0) as client:
                col = client.get(collection_url).raise_for_status().json()
                hrefs = [link["href"] for link in col.get("links", []) if link.get("rel") == "item"]
                items: list[StacItem] = []
                for href in hrefs:
                    # Item hrefs are relative like "./00000/00000.json"; resolve
                    # against the collection URL.
                    url = href if href.startswith("http") else urljoin(collection_url, href)
                    item = client.get(url).raise_for_status().json()
                    bbox = tuple(item["bbox"])
                    # Overture's STAC items publish the GeoParquet asset under the
                    # `aws` key (HTTPS URL to the public S3 bucket). Earlier code
                    # tried `data` / `<type>` keys based on an older convention.
                    assets = item.get("assets", {})
                    data_href = (
                        assets.get("aws", {}).get("href")
                        or assets.get("data", {}).get("href")
                        or assets.get(self._type, {}).get("href")
                    )
                    if data_href is None:
                        continue
                    items.append(StacItem(href=data_href, bbox=bbox))
            logger.info("STAC built: %d items for %s/%s", len(items), self._theme, self._type)
            self._items = items

    def files_for_bbox(self, bbox: StacBBox) -> list[str]:
        if self._items is None:
            self.build()
        assert self._items is not None
        west, south, east, north = bbox
        out: list[str] = []
        for it in self._items:
            iw, is_, ie, in_ = it.bbox
            if ie < west or iw > east or in_ < south or is_ > north:
                continue
            out.append(it.href)
        return out


_INDEXES: dict[tuple[str, str, str], StacIndex] = {}
_INDEXES_LOCK = threading.Lock()


def get_stac_index(release: str, theme: str, type_: str) -> StacIndex:
    key = (release, theme, type_)
    with _INDEXES_LOCK:
        idx = _INDEXES.get(key)
        if idx is None:
            idx = StacIndex(release, theme, type_)
            _INDEXES[key] = idx
        return idx
