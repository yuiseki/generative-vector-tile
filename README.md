# generative-vector-tile

> **Generative dynamic vector tile FaaS.** Natural-language filter parameters compile to safe DuckDB Spatial queries over Overture GeoParquet, encoded as MVT bytes on the fly. Standard MVT URL, no client-side glue.

| | |
| --- | --- |
| **viewer** | <https://yuiseki.github.io/generative-vector-tile/> (requires `?server=...` pointing at a running tile server) |
| **endpoint** | not yet hosted publicly; run the server locally and pass it via `?server=http://127.0.0.1:8080` |
| **architecture** | FastAPI + DuckDB Spatial + httpfs + STAC index + OpenAI-compatible LLM (cloud or llama.cpp) |
| **sibling study** | [study-cng-overture-buildings-tile](https://github.com/yuiseki/study-cng-overture-buildings-tile) (this repo's structural template) |

## What it is

A vector tile server where the **filter is written in natural language** as a query parameter:

```
GET /tile/buildings/14/14552/6451.mvt?q=高さ100m以上のビル
GET /tile/places/15/29089/12947.mvt?q=カテゴリがcafe
GET /tile/transportation/13/7274/3225.mvt?q=高速道路
```

The server translates `q` into a safe DuckDB filter expression, fetches just the Overture Parquet files whose bbox intersects the tile (via a pre-built STAC spatial index), runs the filter through DuckDB Spatial, and emits standard MVT. MapLibre GL JS consumes the result like any other vector tile source.

## Why this is interesting

- **Standard MVT URL** drop-in source for MapLibre or any MVT consumer
- **`q` as natural language** no SQL knowledge required to use it, and no SQL injection target to defend (see the safety model)
- **`(z, x, y)` bounds the query** cost is per-tile, not per-dataset
- **STAC index keeps cold-start sane** only the Parquet files that intersect the tile bbox are read
- **2-layer tile cache** in-memory LRU plus write-through disk cache under `~/.cache/generative-vector-tile/tiles/` so repeat requests skip DuckDB entirely
- **No pre-built tiles** the underlying Overture release updates and the next request reflects it

## Safety model

`q` is **never converted to full SQL**. The compiler produces a single **WHERE-clause expression** that goes into a fixed server-controlled template:

```sql
SELECT <projected_columns>, ST_AsWKB(geom) AS __wkb
FROM read_parquet([<stac_filtered_files>])
WHERE bbox.xmin <= ?       -- GeoParquet 1.1 bbox struct, used for row-group pruning
  AND bbox.xmax >= ?
  AND bbox.ymin <= ?
  AND bbox.ymax >= ?
  AND (<validated_filter_expr>)
LIMIT <limit>
```

The threat model is intentionally narrow: data is public, DuckDB httpfs is read-only, and the connections are pooled `:memory:`, so SQL injection has no integrity or confidentiality target. The remaining risks (resource abuse, SSRF) are bounded by **infrastructure**, not parser checks:

| Layer | What it bounds | Where it lives |
|---|---|---|
| **DuckDB resource cap** | per-query memory, CPU threads, wall-clock timeout | `SET memory_limit`, `SET threads`, watchdog `con.interrupt()` (this repo) |
| **Knative NetworkPolicy** | egress allowed only to Overture S3, STAC, OpenAI, DNS | `k8s/networkpolicy.yaml` (planned, when deployed) |
| **gVisor sandbox** | syscall-level pod isolation | `k8s/ksvc.yaml` `runtimeClassName: gvisor` (planned, when deployed) |

The `q` compiler itself is LLM-only (no AST allowlist, no regex fast-path). The reasoning behind that converging on this minimal design lives in [ADR-0002](https://github.com/yuiseki/TRIDENT/blob/main/docs/ADR/0002-generative-vector-tile-faas.md).

## Endpoint reference

```
GET /tile/{dataset}/{z}/{x}/{y}.mvt
       ?q=<natural-language filter>       (optional)
       &limit=<int, default 1000>

GET /datasets                              list registered datasets and filterable columns
GET /health                                liveness probe
```

## Datasets

| id | source | filterable columns | example `q` |
|---|---|---|---|
| `places` | Overture Places GeoParquet | `name, category, confidence` | `q=カテゴリがcafe` / `q=信頼度0.7以上` |
| `buildings` | Overture Buildings GeoParquet | `height, num_floors, class, subtype` | `q=高さ100m以上` / `q=学校` |
| `transportation` | Overture Transportation segments | `subtype, class, name` | `q=高速道路` / `q=線路` |

Adding a dataset is one new module in `src/generative_vector_tile/datasets/` plus an entry in the registry. Each `Dataset` declares its `filterable_columns`, which is the column-level security boundary.

## Run locally

```bash
uv sync
cp .env.example .env
# edit .env to point at an LLM backend (see "Inference backend" below)
export $(cat .env | xargs)
uv run python -m generative_vector_tile.server
# -> http://127.0.0.1:8080/health
```

In another terminal, serve the static viewer:

```bash
python -m http.server --directory docs 8000
# -> http://localhost:8000/?server=http://127.0.0.1:8080
```

The viewer is a single `docs/index.html` that talks to the server via the configured `?server=...` URL parameter.

## Inference backend

`q` translation goes through any OpenAI-compatible chat completions endpoint with `response_format: json_schema` support. Two modes are supported.

### Mode A: OpenAI cloud

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.1
```

`OPENAI_BASE_URL` stays unset (SDK default).

### Mode B: llama.cpp llama-server (local pool)

The default tile-fetch burst is 6+ concurrent requests, so the recommended mode is to run an N-instance llama-server pool that the application client round-robins across. The repo ships two scripts; the pool is the default for any non-trivial use:

```bash
# Launches 4 instances on ports 18099-18102 with Qwen3.5-2B-Q4_K_M.
./scripts/llama.cpp/start-gvt-llm-pool-mac.sh
```

Then in `.env`:

```env
OPENAI_BASE_URLS=http://127.0.0.1:18099/v1,http://127.0.0.1:18100/v1,http://127.0.0.1:18101/v1,http://127.0.0.1:18102/v1
OPENAI_API_KEY=dummy
OPENAI_MODEL=gvt-llm
LLM_TIMEOUT_S=30
```

`OPENAI_BASE_URLS` is a CSV of OpenAI-compatible base URLs. The application picks one per request via atomic round-robin counter. Setting just `OPENAI_BASE_URL` (singular) still works for single-instance setups.

The launch script defaults to `LLAMA_CPP_DIR=$HOME/llama.cpp`. Override `LLAMA_CPP_DIR` / `HF_REPO` / `HF_QUANT` / `POOL_SIZE` env vars if your layout or hardware budget differs. There is also `start-gvt-llm-mac.sh` for a single-instance setup if you only care about smoke-testing.

## DuckDB tuning

Concurrent tiles run through a small connection pool (default 4) rather than a single locked connection: `spatial + httpfs` SIGSEGV when the same connection runs concurrent queries, so the pool is the way to get parallelism. Override with:

```env
DUCKDB_POOL_SIZE=4
DUCKDB_MEMORY_LIMIT=4GB         # leave unset to let DuckDB pick (~80% of RAM)
DUCKDB_THREADS=8                # leave unset to use num_cores
DUCKDB_QUERY_TIMEOUT_S=15
```

The pool is initialised lazily on first request and pre-warmed by the `lifespan` hook on startup.

## Viewer

GitHub Pages serves `docs/` directly, so the viewer is live at <https://yuiseki.github.io/generative-vector-tile/>. **It needs `?server=...` to point at a running server**; without it the tile fetches fail.

Viewer URL parameters:

| param | default | description |
|---|---|---|
| `server` | none | server base URL; e.g. `http://127.0.0.1:8080` for local dev |
| `dataset` | `buildings` | initial dataset id; can be changed in the UI |
| `q` | (empty) | initial natural-language filter |
| `#<lng>/<lat>/<z>` | (none) | MapLibre's hash for view state |

## Build and deploy (Knative, planned)

The repository ships a `docker/Dockerfile` and a `k8s/ksvc.yaml` for Knative, but the service is not yet running on a public endpoint. The deploy plan and TODOs live in [`TODO.md`](./TODO.md).

```bash
docker build -t generative-vector-tile:0.1.0 -f docker/Dockerfile .
docker save generative-vector-tile:0.1.0 | ctr -n=k8s.io images import -
kubectl apply -f k8s/ksvc.yaml
kubectl get ksvc generative-vector-tile -n knative-pool
```

## Tests and lint

```bash
uv run pytest -v
uv run ruff check
```

## License

[MIT](./LICENSE.md). Overture Maps data usage is bound by [Overture's data license](https://docs.overturemaps.org/attribution/).
