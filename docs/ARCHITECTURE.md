# generative-vector-tile architecture

## Request lifecycle

```
GET /tile/{dataset}/{z}/{x}/{y}.mvt?q=<nl>
        │
        ▼
  server.tile()
        │
        ├── get_dataset(dataset_id) ─────────────► Dataset (registered at import)
        │
        ├── parse_filter(dataset, q) ────────────► FilterExpr | None
        │     (Phase 1: regex patterns; Phase 2: LLM + AST allowlist)
        │
        ├── tile_to_bbox(z, x, y) ───────────────► (west, south, east, north)
        │
        ├── query_features(dataset, bbox, filter_expr, limit)
        │     │
        │     ├── get_stac_index(...).files_for_bbox(bbox) ──► list[s3_url]
        │     │
        │     └── DuckDB Spatial + httpfs:
        │           SELECT projected_cols, ST_AsWKB(geom) AS __wkb
        │           FROM read_parquet([files])
        │           WHERE ST_Intersects(geom, ST_MakeEnvelope(bbox))
        │             AND (<filter_expr.sql>)
        │           LIMIT <limit>
        │
        └── encode_mvt(...) ─────────────────────► MVT bytes
```

## Security boundary

The two security boundaries are:

1. **`Dataset.filterable_columns`** - only listed columns can be referenced in `q`. Adding a filterable column is an explicit code change reviewable in PR.
2. **`parse_filter` output is parameterised** - the SQL fragment uses `?` placeholders and bound `params`. No user-supplied value is interpolated into the SQL string.

DuckDB extension loading runs once at boot (`spatial`, `httpfs`). Phase 2 will add `enable_external_access=false` + a URL allowlist enforced by Knative network policy. Until then, the deployment relies on the cluster-level egress to limit damage if the parser is bypassed.

## STAC index

Overture's S3 path is keyed by ID, not geography. To avoid `read_parquet('s3://.../*')` having to fetch 512 file footers, `stac_index.StacIndex` fetches `https://stac.overturemaps.org/{release}/theme={theme}/type={type}/collection.json` plus all linked items at boot. Lookup is a linear bbox scan over a few hundred items.

## Concurrency

DuckDB-Spatial + httpfs is not thread-safe in our usage pattern (this was the buildings-tile study's painful finding). The connection is wrapped in `threading.Lock`; horizontal concurrency lives in Knative autoscaling (multiple pods) rather than in-process threads.

## Datasets registry

`src/generative_vector_tile/datasets/__init__.py` builds a static `REGISTRY` dict at import time. Each dataset is a frozen dataclass declaring columns, MVT layer name, Overture coordinates, and the filterable subset. Adding a dataset is a single new module + registry entry.

## Phase 2 roadmap

1. LLM-driven `q` → filter expression (cached by `(dataset_id, q)` hash → SQL fragment)
2. AST-level allowlist validation (not just regex pattern match)
3. DuckDB hardening: `enable_external_access=false` after httpfs is bootstrapped, function allowlist, URL allowlist enforced at process level
4. Knative network policy: egress only to Overture S3 and STAC

## Phase 3+ roadmap

- Cloudflare Tunnel for `generative-vector-tile.yuiseki.com`
- More datasets: `transportation`, `base.land`, ...
- Cache layer for `(z, x, y, q, dataset)` → MVT bytes
- Consumer-side integrations: any MVT-capable client (MapLibre / Mapbox GL / native renderers). The first known consumer is the TRIDENT smart maps assistant, which routes user input to a new vector source URL through a chat ability.
