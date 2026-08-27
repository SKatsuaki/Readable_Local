# Readable Local MVP

英語論文PDFまたはTeXソースを受け取り、日本語PDFを作るローカルアプリです。

Readableのように、元論文のレイアウトを見ながらローカル環境だけで日本語PDFを作ることを目標にしています。外部APIへPDFやTeXソースを送らず、翻訳専用NMTまたはOllamaのローカルLLMを使えます。

## できること

- PDFをアップロードして翻訳済みPDFを出力
- `.tex`、`.zip`、`.tar`、`.tar.gz`、`.tgz` のTeXソースをアップロードして翻訳済みPDFを出力
- TeX入力ではOllamaでソース内の本文、見出し、キャプションを翻訳し、LuaLaTeXでPDFを生成
- TeX入力ではLaTeXコマンド、数式、引用、参照、ラベル、ファイル名をできるだけ保持
- 翻訳中の進捗をWeb画面に表示
- 生成後にWeb画面内のPDFビューアーで確認
- 高速本文訳、元レイアウト寄せ、左右並びのPDFを選べる
- 数式っぽい領域は翻訳で上書きせず、元PDFの表示を残す
- 文中に混ざる数式変数の文字化けを減らすため、数学用Unicodeを通常文字へ正規化
- 日本語フォントをPDFへ埋め込み、ビューアーや環境による日本語欠落を減らす
- 高速本文訳では翻訳専用NMTをGPUバッチ実行し、短時間で本文を翻訳
- Ollama LLMでは原則1ページごとに1回だけ呼び出し、長いページだけ安全に分割
- 元レイアウト寄せではページ内の複数領域をまとめて翻訳し、精密さを優先
- タイムアウト時は長いページを自動分割して再試行
- 翻訳エンジンは差し替え式
- NMTとOllamaのどちらもローカルで実行できる

## 起動

```bash
./scripts/start_app.sh
```

ブラウザで以下を開きます。

```text
http://127.0.0.1:8765/translate
```

別ポートで起動したい場合:

```bash
PORT=8766 ./scripts/start_app.sh
```

```text
http://127.0.0.1:8766/translate
```

## 環境構築

Codex内では同梱Pythonを優先して使います。別のMac、Linux、NVIDIA GPUマシンへ持っていく場合は、Python依存とPopplerを入れてください。

Python依存:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PDFの画像化にPopplerを使います。

macOS:

```bash
brew install poppler
```

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils python3-venv
```

Ollamaは別途インストールしてください。NVIDIA GPUで使う場合は、先にNVIDIAドライバが入り、`nvidia-smi` が動く状態にしておきます。

## Dockerで動かす場合

DGX SPARKなどのLinux/NVIDIA環境では、Docker Composeでアプリ、Ollama、翻訳専用NMTを同じDockerプロジェクト内に立ち上げます。ホストにはPython依存やTeX環境を入れません。アプリコンテナ内にPoppler、TeX Live、LuaLaTeX、XeLaTeX、latexmk、日本語フォントを入れます。

```bash
docker compose up -d --build
```

推奨モデルを入れる場合:

```bash
docker compose exec ollama ollama pull qwen2.5:3b-instruct
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull gemma3:4b
docker compose exec ollama ollama pull qwen3:4b
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull deepseek-r1:7b
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull gemma2:9b
docker compose exec ollama ollama pull gpt-oss:20b
docker compose exec ollama ollama pull gemma3:27b
docker compose exec ollama ollama pull qwen3:30b
docker compose exec ollama ollama pull deepseek-r1:32b
```

複数モデルをまとめて入れる場合は、ネットワークとストレージ負荷を見ながら3本程度の並列取得にすると安定しやすいです。

状態確認:

```bash
docker compose ps
docker compose logs --tail=80 app
docker compose logs --tail=80 nmt
docker compose exec ollama ollama ps
```

ブラウザ:

```text
http://<DGX_SPARKのLAN内IP>:8766/translate
```

Docker運用時は、アップロードPDF、作業ファイル、生成PDFはコンテナ内のtmpfsに置きます。ホストのプロジェクトディレクトリへ論文PDFを保存しません。生成PDFはWeb画面で確認・ダウンロードできますが、既定では30分後にジョブ履歴から削除され、コンテナ再起動でも消えます。NMTモデルだけはDockerボリュームに保持し、論文データとは分離します。

TeXソースをアップロードした場合も、展開したソース、翻訳後ソース、ビルド中間ファイルはコンテナ内の一時領域だけに置きます。ホスト側の `/home/katsuaki/codex/readable_loacal` には論文データを保存しません。

ジョブ制限:

```bash
READABLE_MAX_CONCURRENT_JOBS=4 READABLE_MAX_QUEUED_JOBS=2 docker compose up -d
```

NVIDIA環境では既定でGPUメモリ使用量を見て、同時実行数を動的に制限します。SPARK向けの既定値は、最大同時実行4本、メモリ予算100GB、最低空き8GB、標準ジョブ予約20GBです。NMTはモデルを共有するため、既定で同時1本に制限しています。`READABLE_MEMORY_BUDGET_GB`、`READABLE_MIN_FREE_MEMORY_GB`、`READABLE_JOB_MEMORY_RESERVE_GB`、`READABLE_JOB_MEMORY_RESERVE_NMT_GB`、`READABLE_NMT_MAX_CONCURRENT_JOBS`、`READABLE_DYNAMIC_JOB_LIMITS` で調整できます。GB10のように `nvidia-smi` がメモリ容量を返さない場合は、ユニファイドメモリとしてシステムメモリ空き量を使います。

状態API:

```text
http://<DGX_SPARKのLAN内IP>:8766/api/status
http://<DGX_SPARKのLAN内IP>:8766/api/models
```

## 翻訳専用NMTで翻訳する場合

画面の標準エンジンは `高速NMT（NLLB 600M）` です。NLLB-200 Distilled 600MをPyTorch CUDAでNVIDIA GPU上に読み込みます。初回起動だけモデルの取得に時間がかかります。画面右上または`/api/status`の`nmt`が`準備完了`になった後は、モデルを再利用します。

NMTは本文の内容を素早く読む用途に向きます。図中ラベル、表セル、専門用語の言い回しを細かく整えたい場合は、Ollama LLMの元レイアウト寄せを使います。

NLLBはTransformerのScaled Dot-Product Attentionを使う翻訳専用モデルです。実行はPyTorch CUDAが行い、SDPAの最適化カーネルとGPUバッチ処理を使います。

## Ollamaで翻訳する場合

このフォルダには、プロジェクト内だけで動くOllamaを置けます。

```bash
./scripts/start_ollama.sh
```

別のターミナルでアプリを起動します。

```bash
./scripts/start_app.sh
```

画面の翻訳エンジンで `Ollama LLM` を選び、モデルを選びます。レイアウトは `高速本文訳（読み比べ）` が標準です。高速本文訳では原則1ページあたり1回だけLLMを呼び、6,000文字を超えるページだけ分割します。元レイアウト寄せでもページ内の文字領域をまとめて投げ、長すぎるページやタイムアウトしたページだけ小さく分割して再試行します。2から3分程度で内容確認したい場合は、まずNMTを使い、表現を整える用途で3Bから8B級LLMを使ってください。

TeXソースをアップロードした場合は、画面の翻訳エンジン選択にかかわらずOllamaを使います。`.tex` 単体だけでなく、複数ファイルを含む `.zip` や `.tar.gz` も受け付けます。`\\documentclass` と `\\begin{document}` を持つメインTeXを自動で探し、プロジェクト内の `.tex` ファイルを翻訳してから `latexmk -lualatex` でPDFを生成します。必要な場合は `READABLE_TEX_ENGINE=xelatex` でXeLaTeXへ切り替えられます。

TeX入力で翻訳する主な対象:

- 本文段落
- `\\title`、`\\section`、`\\caption` などの自然言語引数
- `\\item` の本文
- `tabular`、`tabular*`、`array` のセル内にある通常テキスト

TeX入力で保持する主な対象:

- 数式、引用、参照、ラベル
- LaTeXコマンド、環境、ファイル名
- `equation`、`align`、`tikzpicture` など構文崩れの影響が大きい環境

図そのものが画像PDF/PNG/JPEGとして埋め込まれている場合、その画像内の英語はTeXソース翻訳では変更できません。TikZや本文として書かれたキャプション、見出し、段落は翻訳対象になります。

現在のOllama実行は、MLX形式ではなく `GGUF` モデルをMetal GPUで動かす方式です。Apple Silicon上でGPUは使えますが、MLXランタイムそのものではありません。MLX形式にする場合は、別途 `mlx-lm` とMLX用モデルを使う翻訳エンジンを追加する必要があります。

現在のMacでの構成:

- モデル形式: `GGUF`
- 実行: Ollama / llama.cpp系ランナー
- Apple Silicon GPU: Metal経由で利用
- MLX: 未使用

NVIDIA GPUのLinux環境でも、OllamaがCUDAでモデルを動かせる状態なら、このアプリはそのまま使えます。アプリ側はOllama APIへ翻訳を投げるだけなので、GPUの種類はOllama側が吸収します。Docker運用では既定で `OLLAMA_KEEP_ALIVE=30m`、`OLLAMA_MAX_LOADED_MODELS=1`、`OLLAMA_NUM_PARALLEL=1` を設定し、翻訳中に同じモデルを保持しやすくしています。

選べるモデル:

- `qwen2.5:3b-instruct`: 軽量確認用
- `qwen2.5:3b`: 軽量確認用
- `gemma3:4b`: 軽量、多言語向け
- `qwen3:4b`: 軽量、新しめ
- `qwen2.5:7b-instruct`: 中量、翻訳向け
- `qwen2.5:7b`: 中量
- `qwen3:8b`: 7から10B帯、新しめ
- `deepseek-r1:7b`: 7から10B帯、推論重視
- `llama3.1:8b`: 中量、汎用
- `gemma2:9b`: 中量、多言語向け
- `gpt-oss:20b`: 高品質寄り、遅め
- `qwen3:30b`: 翻訳向け、遅め
- `gemma3:27b`: 多言語向け、比較的安定
- `deepseek-r1:32b`: 推論重視、かなり遅め

速度重視なら `高速本文訳（読み比べ）` を使います。これは元ページ画像を左、翻訳本文を右に置きます。NMTは全ページをGPUバッチへまとめ、Ollama LLMはページ単位にまとめて呼び出します。元レイアウト寄せは図表・キャプション・小さな領域まで拾うため、品質優先ですが時間がかかります。

モデルを入れ直す場合:

```bash
./scripts/pull_model.sh qwen2.5:3b-instruct
./scripts/pull_model.sh qwen2.5:3b
./scripts/pull_model.sh gemma3:4b
./scripts/pull_model.sh qwen3:4b
./scripts/pull_model.sh qwen2.5:7b-instruct
./scripts/pull_model.sh qwen2.5:7b
./scripts/pull_model.sh qwen3:8b
./scripts/pull_model.sh deepseek-r1:7b
./scripts/pull_model.sh llama3.1:8b
./scripts/pull_model.sh gemma2:9b
```

30B前後のモデルを使う場合は、必要なものだけ事前に入れてください。

```bash
./scripts/pull_model.sh gpt-oss:20b
./scripts/pull_model.sh qwen3:30b
./scripts/pull_model.sh gemma3:27b
./scripts/pull_model.sh deepseek-r1:32b
```

## レイアウトと翻訳方式

### 高速本文訳

左に元PDFページ画像、右に翻訳文を流し込む形式です。NMTは論文全体をGPUバッチ処理し、Ollama LLMは原則ページ数と同程度の呼び出し回数にします。数分で内容を把握したい場合の標準モードです。図中の小さな英語や表セル単位の上書き翻訳は行いません。

### 元レイアウト寄せ

元PDFページを画像として背景に置き、PDFから抽出できた文字領域だけを日本語訳で上書きします。式や数式っぽい短い領域は上書きせず、元PDFの表示を残します。Ollama LLMではページ内の領域を小さな束に分けて翻訳し、Qwen 2.5 3Bのような軽量モデルでもJSON応答が崩れてページ全体が未翻訳になる問題を避けます。

### 左右並び

左に元PDFページ画像、右に翻訳文を流し込む形式です。元レイアウト再現より読みやすさを優先したい場合に使います。

### バッチ翻訳

高速本文訳ではNMTのトークン数ベースGPUバッチ、またはLLMのページ単位呼び出しを使います。元レイアウト寄せではPDF内の小さな文字領域をページ内バッチとして翻訳します。既定ではOllamaの元レイアウト寄せを最大4領域・約2,200文字ずつに分け、長すぎるページや時間切れになったページだけさらに自動分割します。`READABLE_OLLAMA_OVERLAY_BATCH_ITEMS` と `READABLE_OLLAMA_OVERLAY_BATCH_CHARS` で調整できます。完全な元レイアウト寄せは翻訳対象が多くなるため、2から3分を狙う場合は高速本文訳を使ってください。

### 数式と変数名

式そのものはできるだけ元PDF側を残します。本文中に出る `𝑥`, `𝜃`, `𝐿`, `𝓛`, `𝛼`, `ᵢ` のような数学用Unicodeは、PDF描画前に `x`, `θ`, `L`, `α`, `i` などへ正規化します。これにより、文中の変数名が四角や欠落になる問題を減らします。

### Apple SiliconのGPU利用

Apple SiliconではOllamaがMetal経由でGPUを使えます。`start_ollama.sh` はCPU固定をしない設定にしてあります。

GPUが見えているか確認する場合:

```bash
./scripts/check_ollama_gpu.sh
```

詳細ログを見ながら起動する場合:

```bash
OLLAMA_DEBUG=1 ./scripts/start_ollama.sh
```

もし古い設定でOllamaが起動済みの場合は、一度Ollamaを止めてから起動し直してください。

```bash
pkill ollama
./scripts/start_ollama.sh
```

CPUで動かしたい場合だけ、次のように起動します。

```bash
READABLE_FORCE_CPU=1 ./scripts/start_ollama.sh
```

現在ロードされているモデルがGPUを使っているか見る場合:

```bash
./.local/ollama/ollama ps
```

`PROCESSOR` が `100% GPU` のようになっていれば、Metal GPUで推論しています。

### NVIDIA GPUの利用

LinuxやWSL2でNVIDIA GPUを使う場合は、先にNVIDIAドライバとOllamaを入れてください。`nvidia-smi` が動き、Ollamaが起動できる状態であれば、このアプリから同じOllamaを使えます。

確認:

```bash
nvidia-smi
ollama --version
ollama serve
```

別ターミナルでモデルを入れます。

```bash
ollama pull qwen2.5:3b-instruct
```

このプロジェクトから起動する場合、`scripts/start_ollama.sh` は以下の順でOllamaを探します。

- `OLLAMA_BIN` で指定されたOllama
- `./.local/ollama/ollama`
- `PATH` 上の `ollama`

Linux/NVIDIAでは、通常はシステムに入れた `ollama` が使われます。

```bash
./scripts/start_ollama.sh
./scripts/check_ollama_gpu.sh
```

`ollama ps` または `./scripts/check_ollama_gpu.sh` の `Loaded models` で `PROCESSOR` がGPUになっていれば、NVIDIA GPU側で推論しています。

CPUで動かしたい場合だけ、次のように起動します。

```bash
READABLE_FORCE_CPU=1 ./scripts/start_ollama.sh
```

## 動作確認だけする場合

画面の翻訳エンジンで `Copy only` を選ぶと、翻訳APIなしでPDF生成だけ試せます。

## コマンドで使う場合

```bash
python3 readable_pdf.py input.pdf output/pdf/translated.pdf --engine ollama --model qwen2.5:3b-instruct
```

高速本文訳を明示する場合:

```bash
python3 readable_pdf.py input.pdf output/pdf/translated.pdf --engine ollama --model qwen2.5:3b-instruct --layout fast
```

元レイアウト寄せで作る場合:

```bash
python3 readable_pdf.py input.pdf output/pdf/translated.pdf --engine ollama --model qwen2.5:3b-instruct --layout overlay
```

PDF生成だけ確認する場合:

```bash
python3 readable_pdf.py input.pdf output/pdf/check.pdf --engine copy
```

## 出力先

生成されたPDFは以下に保存されます。

```text
output/pdf/
```

一時ファイルは以下に作られます。

```text
tmp/pdfs/
```

Webアプリを再起動すると画面上の処理履歴は消えますが、`output/pdf/` にPDFが残っていれば、ジョブID付きのURLから生成済みPDFを復元表示できます。

## よくある問題

### `address already in use`

OllamaやWebアプリが既に起動しています。まずブラウザで開けるか確認してください。別ポートでWebアプリを起動する場合は次のようにします。

```bash
PORT=8766 ./scripts/start_app.sh
```

### 翻訳が遅い

2から3分で内容確認したい場合は、まず`高速NMT（NLLB 600M）`と`高速本文訳（読み比べ）`を使います。NMTでは物足りないページだけ、`qwen2.5:3b-instruct`、`qwen2.5:3b`、`gemma3:4b`、`qwen3:4b`、`qwen2.5:7b`などで確認します。元レイアウト寄せは図表・表・キャプションなどの小領域も翻訳するため、3Bモデルでも長い論文ではかなり時間がかかります。20Bから30B級モデルは仕上げ確認向けです。

### Ollamaが何度も復帰・killされる

Ollamaのランナーが短時間で起動・終了を繰り返す場合は、モデルの保持時間が短い、GPUメモリの都合で退避している、または小さな翻訳リクエストが多すぎることが原因になりがちです。Docker運用ではモデルを30分保持する設定にし、元レイアウト寄せもページ単位の呼び出しへ寄せています。処理完了後すぐVRAMから外したい場合は、`.env` などで `OLLAMA_KEEP_ALIVE=0` に変更できますが、連続翻訳は遅くなります。

### MLXになっているか確認したい

現在はMLXではありません。Ollamaのモデル一覧で `details.format` が `gguf` ならGGUF形式です。

```bash
curl -s http://127.0.0.1:11434/api/tags
```

### 数式変数が文字化けする

本文中に混ざる数学用Unicodeは正規化します。ただし、画像として埋め込まれた式や図中の文字はOCRしていないため、完全な置換対象ではありません。

## 制限

- スキャン画像だけのPDFは、まだ本文抽出できません。
- 段組みの抽出順はPDFによって乱れることがあります。
- 元レイアウト寄せは、元PDFページ画像の上に翻訳テキストを重ねる方式です。
- 図中ラベルもPDFから文字として抽出できる場合は置換します。画像に埋め込まれた文字のOCR置換はまだ未対応です。
- 複雑な表、脚注、式番号、複数段組みの細かな揃えはPDFによって崩れることがあります。
- 完全なMLXランタイム対応は未実装です。
