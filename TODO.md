# TODO

PoC として「Overture GeoParquet を自然言語フィルタで動的ベクタタイル化する」一連は動いている。タイル表示まで届くようにはなったが、まだ十分速いとは言えない。残課題を優先度順に並べる。

## 1. LLM の per-tile 重複呼び出しを coalesce する (最優先)

現状: フロントエンドが新しい `q` でタイル群を一斉に投げると、同じ `(dataset, q)` に対する LLM 翻訳が 6 〜 12 並列で発火する。先頭の 1 個が `filter cache` を書く前に、他のタイルもそれぞれ llama-server を叩いてしまう。round-robin プールで分散されるので落ちはしないが、初動の遅延が「タイルが完了する時間」ではなく「LLM プールが捌き切る時間」になる。

対応: `tile_cache.RequestCoalescer` と同じ発想を LLM 層にも入れる。

- キー: `(dataset_id, normalized_q)`
- 先頭の 1 リクエストだけが LLM を呼び、後続は `threading.Event` で待つ
- リーダーが失敗したら例外を全員に伝搬 (今の coalescer と同じ挙動)
- フォロワーは `filter_cache` から結果を取り直す

これで初動の LLM 呼び出しが 6-12 並列 → 1 + (5-11)待ち に圧縮される。

## 2. クライアント側の q 入力 debounce

現状: フォーム `q` を 1 文字打つたびにタイル fetch が走り、次の打鍵で全部 abort される。サーバログに `tile abandoned-before-compute` が大量に出る (1.5 秒前の q の LLM 翻訳がまだ終わってない間に新しい q が来てキャンセル)。LLM 計算がドブに捨てられている。

対応: `docs/index.html` の q input の `input` ハンドラに 300ms 程度の debounce を入れる。Enter / blur では即座に確定。

## 3. DuckDB の HTTP range fetch が連続タイルで再走査される

現状: pool=4 で並列に DuckDB を回せるようになったが、それでも隣接タイル間で 5-18 秒の cold fetch が発生する。同一 Parquet ファイルの別 row group に毎回 HTTP range request が飛んでいる。`enable_external_file_cache` のデフォルトはファイルメタデータ寄りで、row group 本体のキャッシュが効きにくい。

対応案 (どれか / 組み合わせ):

- bbox から特定したファイル一覧について、最初のリクエスト直後に背景で `SELECT count(*) FROM read_parquet([files]) WHERE bbox.xmin <= ?...` を投げて row group 本体をキャッシュに乗せる
- 隣接タイル prefetch を frontend 側で発行する (現在のビューポートの外周 1 周分のタイルを低優先度 fetch)
- DuckDB ではなく `pyarrow.parquet.ParquetFile` で row group を明示的に読み、`pyarrow.fs.S3FileSystem` のリトライ込みで生バイトを LRU しておく (これは大改造)

`enable_external_file_cache=true` / `parquet_metadata_cache=true` の設定は試したが体感差なし。row group 本体の HTTP range キャッシュは httpfs が持っていないので、上のような上位レイヤでの対応が必要。

## 4. Knative デプロイ

ローカル PoC は十分回ったので、`generative-vector-tile.yuiseki.com` で公開する Knative service 化に進む。

- `Dockerfile` (uv + python:3.14-slim)
- Knative `Service` YAML: cpu=2, memory=4Gi, autoscale=0-3, request-concurrency=4
- Cloudflare Tunnel の DNS と Knative gateway の TLS をどう繋ぐか確認

LLM 側 (llama-server) は GPU 必要なので別 Knative service にし、`OPENAI_BASE_URL` で接続する。`llm.py` の pool は CSV (`OPENAI_BASE_URLS`) で渡せるので Knative で 2 〜 3 replica にしておけば良い。

## 5. 既知の改善余地 (緊急性は低い)

- マルチデータセット質問 (例: 「高速道路と高層ビル」) は LLM プロンプトで `dataset` を選ばせる構成が必要。今は 1 タイル = 1 dataset 固定で諦めている
- フィルタ式の制約強化: 現在は LLM 出力をそのまま DuckDB に通している (ADR-0002 の脅威モデル上は許容範囲)。将来的に外部公開する場合は predicate JSON への移行を検討する
- `lifespan` の warmup で 1 つだけ実タイルを取得しておけば初回ユーザの体感がさらに良くなる
- `place` データセットの追加属性 (categories, websites など) を MVT に乗せて popup に出すと PoC として説得力が増す
