# generative-vector-tile

> **Generative dynamic vector tile FaaS.** Natural-language filter parameters compile to safe DuckDB Spatial queries over Overture GeoParquet, encoded as MVT bytes on the fly. Standard MVT URL, no client-side glue.

| | |
| --- | --- |
| **viewer (planned)** | `https://yuiseki.github.io/generative-vector-tile/` |
| **endpoint (planned)** | `https://generative-vector-tile.yuiseki.com/tile/{dataset}/{z}/{x}/{y}.mvt?q=<natural-language>` |
| **architecture** | FastAPI on Knative ksvc, DuckDB Spatial + httpfs + STAC index |
| **sibling study** | [study-cng-overture-buildings-tile](https://github.com/yuiseki/study-cng-overture-buildings-tile) (this repo's structural template) |

## What it is

A vector tile server where the **filter is written in natural language** as a query parameter:

```
GET /tile/buildings/14/14552/6451.mvt?q=高さ100m以上のビル
GET /tile/places/15/29089/12947.mvt?q=カテゴリがcafe
```

The server translates `q` into a safe DuckDB filter expression, fetches just the Overture Parquet files whose bbox intersects the tile (via a pre-built STAC spatial index), runs the filter through DuckDB Spatial, and emits standard MVT. MapLibre GL JS consumes the result like any other vector tile source.

## Why this is interesting

- **Standard MVT URL** → drop-in source for MapLibre / any MVT consumer
- **`q` as natural language** → no SQL knowledge required to use it, and no SQL injection surface to defend against (see the safety model)
- **`(z, x, y)` bounds the query** → cost is per-tile, not per-dataset
- **STAC index keeps cold-start sane** → only the Parquet files that intersect the tile bbox are read
- **Knative scale-to-zero** → ksvc costs nothing when idle, autoscales horizontally under load
- **No pre-built tiles** → the underlying Overture release updates and the next request reflects it

## Safety model

`q` is **never converted to full SQL**. The compiler produces a single **WHERE-clause expression** that goes into a fixed server-controlled template:

```sql
SELECT geom, <projected_columns>
FROM read_parquet([<stac_filtered_files>])
WHERE ST_Intersects(geom, ST_MakeEnvelope(<bbox>))
  AND (<validated_filter_expr>)
LIMIT <limit>
```

The threat model is intentionally narrow: data is public, DuckDB httpfs is read-only, and the connection is per-request `:memory:`, so SQL injection has no integrity or confidentiality target. The remaining risks (resource abuse, SSRF) are bounded by **infrastructure**, not parser checks:

| Layer | What it bounds | Where it lives |
|---|---|---|
| **DuckDB resource cap** | per-query memory, CPU threads, wall-clock timeout | `SET memory_limit`, `SET threads`, watchdog `con.interrupt()` (this repo) |
| **Knative NetworkPolicy** | egress allowed only to Overture S3, STAC, OpenAI, DNS | `k8s/networkpolicy.yaml` (deploy time) |
| **gVisor sandbox** | syscall-level pod isolation, blast radius limited even on unknown RCE | `k8s/ksvc.yaml` `runtimeClassName: gvisor` (deploy time) |

The `q` compiler itself is LLM-only (no AST allowlist, no regex fast-path). See [ADR-0002 §意思決定の経緯](https://github.com/yuiseki/TRIDENT/blob/main/docs/ADR/0002-generative-vector-tile-faas.md) for why we converged on this minimal design.

## Endpoint reference

```
GET /tile/{dataset}/{z}/{x}/{y}.mvt
       ?q=<natural-language filter>       (optional)
       &limit=<int, default 5000>

GET /datasets                              list registered datasets and filterable columns
GET /health                                liveness probe
```

## Datasets (Phase 1)

| id | source | filterable columns | example `q` |
|---|---|---|---|
| `places` | Overture Places GeoParquet | `name, category, confidence` | `q=カテゴリ=cafe` / `q=信頼度0.7以上` |
| `buildings` | Overture Buildings GeoParquet | `name, height, num_floors, class` | `q=高さ100m以上` / `q=クラス=commercial` |

Adding a dataset is one new module in `src/generative_vector_tile/datasets/` plus an entry in the registry. Each `Dataset` declares its `filterable_columns`, which is the column-level security boundary.

## Run locally

```bash
uv sync
cp .env.example .env
# edit .env to point at an LLM backend (see "Inference backend" below)
export $(cat .env | xargs)
uv run python -m generative_vector_tile.server
# → http://127.0.0.1:8080/health
```

In another terminal, serve the static viewer:

```bash
python -m http.server --directory docs 8000
# → http://localhost:8000/?server=http://127.0.0.1:8080
```

The viewer is a single `docs/index.html` that talks to the server via the configured `?server=...` URL parameter. It works as a static page on GitHub Pages too.

## Inference backend

`q` translation goes through any OpenAI-compatible chat completions endpoint. Two modes are supported out of the box.

### Mode A: OpenAI cloud

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.1
```

`OPENAI_BASE_URL` stays unset (SDK default).

### Mode B: llama.cpp llama-server (local)

The repo ships a script that launches llama-server with [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) for testing on Apple Silicon:

```bash
./scripts/llama.cpp/start-gvt-llm-mac.sh
# → http://127.0.0.1:18099/v1   (alias: gvt-llm)
```

Then in `.env`:

```env
OPENAI_BASE_URL=http://127.0.0.1:18099/v1
OPENAI_API_KEY=dummy
OPENAI_MODEL=gvt-llm
LLM_TIMEOUT_S=30
```

llama-server speaks the same `/v1/chat/completions` API with `response_format: json_schema` (used by `chat.completions.parse` under the hood). gvt's call path is identical for both backends.

The launch script defaults to `LLAMA_CPP_DIR=$HOME/llama.cpp` and the model at the cached HF snapshot path. Override `LLAMA_CPP_DIR` / `MODEL_PATH` env vars if your layout differs.

## Viewer (GitHub Pages)

The `docs/` directory is self-contained:

- `docs/index.html` — MapLibre GL JS UI with dataset selector and `q` input
- `docs/style.json` — Esri World Imagery raster base style
- both are vanilla, no build step

To enable GitHub Pages: repository **Settings → Pages** → Source `Deploy from a branch`, Branch `main` / Folder `/docs`. The viewer then publishes to `https://<owner>.github.io/generative-vector-tile/`.

Viewer URL parameters:

| param | default | description |
|---|---|---|
| `server` | `https://generative-vector-tile.yuiseki.com` | server base URL; override with `http://127.0.0.1:8080` for local dev |
| `dataset` | `buildings` | initial dataset id; can be changed in the UI |
| `q` | (empty) | initial natural-language filter |
| `#<lng>/<lat>/<z>` | (none) | MapLibre's hash for view state |

## Build and deploy (Knative)

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
