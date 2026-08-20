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

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

STAC_ROOT = "https://stac.overturemaps.org"
StacBBox = tuple[float, float, float, float]

# Datasets register this sentinel instead of a pinned release id. Overture
# deletes old releases from the STAC catalog after a few months (the pinned
# 2026-04-15.0 vanished and every tile request started 500ing), so the release
# is resolved from the catalog's `"latest": true` child link at first use.
LATEST_RELEASE = "latest"

_RELEASE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$")

_resolved_latest: str | None = None
_resolve_lock = threading.Lock()


def _release_from_href(href: str) -> str | None:
    """Extract the release id from a catalog child href.

    Hrefs look like `https://stac.overturemaps.org/2026-08-19.0/catalog.json`
    (or the relative `./2026-08-19.0/catalog.json`); the release is the last
    path segment before the filename.
    """
    parts = [p for p in urlparse(href).path.split("/") if p and p != "."]
    for part in reversed(parts):
        if _RELEASE_RE.match(part):
            return part
    return None


def _fetch_latest_release() -> str:
    url = f"{STAC_ROOT}/catalog.json"
    logger.info("STAC release resolve: %s", url)
    with httpx.Client(timeout=30.0) as client:
        catalog = client.get(url).raise_for_status().json()
    children = [
        link for link in catalog.get("links", []) if link.get("rel") == "child"
    ]
    for link in children:
        if link.get("latest"):
            release = _release_from_href(link.get("href", ""))
            if release:
                return release
    # No `latest` flag (or an href we could not parse): fall back to the
    # highest-sorting release id. The ids are zero-padded dates, so a plain
    # lexicographic sort is chronological.
    candidates = sorted(
        r for r in (_release_from_href(link.get("href", "")) for link in children) if r
    )
    if candidates:
        logger.warning(
            "STAC catalog has no `latest` child; falling back to %s", candidates[-1]
        )
        return candidates[-1]
    raise RuntimeError(f"no Overture release found in STAC catalog {url}")


def resolve_release(release: str) -> str:
    """Turn the `latest` sentinel into a concrete Overture release id.

    Resolution is cached for the process lifetime so the index and the disk
    cache stay pinned to one release while a worker is alive. Knative scales
    these pods to zero, so a fresh pod picks up a new release on its own.
    Set `OVERTURE_RELEASE` to pin a specific release without a code change.
    """
    if release != LATEST_RELEASE:
        return release
    override = os.environ.get("OVERTURE_RELEASE")
    if override and override != LATEST_RELEASE:
        return override
    global _resolved_latest
    with _resolve_lock:
        if _resolved_latest is None:
            _resolved_latest = _fetch_latest_release()
            logger.info("STAC latest release resolved: %s", _resolved_latest)
        return _resolved_latest


def _disk_cache_dir() -> Path:
    base = os.environ.get("STAC_CACHE_DIR")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "generative-vector-tile" / "stac"


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

    def _cache_path(self) -> Path:
        return _disk_cache_dir() / f"{self._release}_{self._theme}_{self._type}.json"

    def _load_from_disk(self) -> list[StacItem] | None:
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            items = [StacItem(href=r["href"], bbox=tuple(r["bbox"])) for r in raw]
            logger.info(
                "STAC disk-cache hit: %d items from %s", len(items), path
            )
            return items
        except Exception as e:
            logger.warning("STAC disk-cache load failed (%s); refetching", e)
            return None

    def _save_to_disk(self, items: list[StacItem]) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(
                [{"href": it.href, "bbox": list(it.bbox)} for it in items], f
            )
        tmp.replace(path)
        logger.info("STAC disk-cache wrote %d items to %s", len(items), path)

    def build(self) -> None:
        with self._lock:
            if self._items is not None:
                return
            cached = self._load_from_disk()
            if cached is not None:
                self._items = cached
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
            try:
                self._save_to_disk(items)
            except Exception as e:
                logger.warning("STAC disk-cache save failed (%s); continuing", e)

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
    release = resolve_release(release)
    key = (release, theme, type_)
    with _INDEXES_LOCK:
        idx = _INDEXES.get(key)
        if idx is None:
            idx = StacIndex(release, theme, type_)
            _INDEXES[key] = idx
        return idx
