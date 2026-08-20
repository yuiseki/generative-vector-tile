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

## 3. DuckDB の HTTP range fetch が連続タイルで再走査される (解決済み)

**ここに書いてあった結論が間違っていた。** 元の記述はこうだった:

> `enable_external_file_cache=true` / `parquet_metadata_cache=true` の設定は試したが体感差なし。row group 本体の HTTP range キャッシュは httpfs が持っていないので、上のような上位レイヤでの対応が必要。

`enable_external_file_cache` は DuckDB 1.5.2 では**デフォルトで有効**で、row group 本体もちゃんとキャッシュする。効果が観測できなかった原因は設定ではなく `duckdb_query` 側の構造だった。当時は `duckdb.connect()` を pool=4 で持っていたが、external file cache は**データベースインスタンス単位**なので、キャッシュも4つに分裂していた。つまりウォームなキャッシュに当たるのは最大4回に1回で、残りは毎回 S3 を読み直していた。「体感差なし」はこれを見ていた。

1インスタンス + クエリごとに `cursor()` へ変更したところ (cursor はインスタンスのキャッシュを共有する)、z16 の divisions タイルで:

```
cold        31 rows in 80.6s
same tile   31 rows in  1.5s
neighbour   32 rows in  1.5s     ← 隣接タイルも同じ row group を共有
4-tile concurrent burst: 3.1s wall for all four
```

上に挙げていた「pyarrow で row group を明示的に読んで自前 LRU」の大改造は不要になった。「最初のリクエスト直後に背景で投げて温める」案も、`WARMUP_TILES` として起動時に実行する形で入っている (Cloudflare が100秒で切るので、cold read はユーザーのリクエスト上で起こしてはいけない)。

教訓: キャッシュ設定を評価するときは、その設定のスコープ (ここではインスタンス単位) と、こちらの接続の持ち方が噛み合っているかを先に確認する。

残っている案:

- 隣接タイル prefetch を frontend 側で発行する (現在のビューポートの外周 1 周分のタイルを低優先度 fetch)。キャッシュが効くようになったので、以前より筋が良くなった
- `max-scale: 3` のままなので、バーストで2つ目の Pod に載ったタイルは cold キャッシュを引く。キャッシュ局所性を取るなら `max-scale: 1`、スループットを取るなら現状維持

## 4. Knative デプロイ

ローカル PoC は十分回ったので、`generative-vector-tile.yuiseki.com` で公開する Knative service 化に進む。

- `Dockerfile` (uv + python:3.14-slim)
- Knative `Service` YAML: cpu=2, memory=4Gi, autoscale=1-3, request-concurrency=5 (`min-scale=0` は不可 — §3 の通りキャッシュを捨ててしまう)
- Cloudflare Tunnel の DNS と Knative gateway の TLS をどう繋ぐか確認

LLM 側 (llama-server) は GPU 必要なので別 Knative service にし、`OPENAI_BASE_URL` で接続する。`llm.py` の pool は CSV (`OPENAI_BASE_URLS`) で渡せるので Knative で 2 〜 3 replica にしておけば良い。

## 5. 既知の改善余地 (緊急性は低い)

- マルチデータセット質問 (例: 「高速道路と高層ビル」) は LLM プロンプトで `dataset` を選ばせる構成が必要。今は 1 タイル = 1 dataset 固定で諦めている
- フィルタ式の制約強化: 現在は LLM 出力をそのまま DuckDB に通している (ADR-0002 の脅威モデル上は許容範囲)。将来的に外部公開する場合は predicate JSON への移行を検討する
- ~~`lifespan` の warmup で 1 つだけ実タイルを取得しておけば初回ユーザの体感がさらに良くなる~~ → `WARMUP_TILES` として実装済み。Ready になる前に実タイルを1枚引く (クラスタ内で167秒、1回だけ)
- `place` データセットの追加属性 (categories, websites など) を MVT に乗せて popup に出すと PoC として説得力が増す
