# YASAKANI — Linux x86_64 ビルド済み配布物（AMD/HIPフォーク）

YASAKANIエンジンでQwen3.8-27BをAMD Radeon GPU 1枚でローカル実行：OpenAI互換APIサーバー・
ターミナルチャット・公式ブラウザチャット。**GUIはなし** — ブラウザチャットは本家と同じ
`chat.html` ページを127.0.0.1で配信するもの（Windows版と同一）。

これは [yasakani-dev/yasakani](https://github.com/yasakani-dev/yasakani) のAMD/HIPフォークです。
HIPバックエンドは**追加**であり置換ではありません：CUDA/NVIDIA経路・Python CLI・API・すべての
デフォルトは本家から不変。`bin/` のエンジンライブラリは `gfx1100` 向けHIPビルドです。

## はじめに（3ステップ）

1. GitHub Releasesから `yasakani-hip-<version>-linux-x86_64.tar.gz` を**ダウンロード**して**解凍**。
2. 前提を一度だけインストール：**ROCm 7.1ランタイム**（下のaptコマンド1つ）と
   **Python依存**（解凍フォルダ内で `python3 -m pip install ./python`）。
3. **`./start_amd.sh`** を実行 — 環境をチェック（不足時は対処を表示）、モデルフォルダと
   モード（ブラウザチャット / ターミナルチャット / APIサーバー）を選択、モデルが無い場合は
   ダウンロードコマンドを案内します。

## 検証済み環境

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XTX（`gfx1100`・24 GB）— 唯一の検証済みGPU |
| OS | Ubuntu 26.04.1 LTS・x86-64・kernel 7.0.0（in-tree `amdgpu` ドライバ） |
| ROCm | 7.1（Ubuntuパッケージ・例 `libamdhip64-7` 7.1.0） |
| Python | 仮想環境内3.14（3.10以上が必要） |
| Engine | `bin/libyskn.so`・35,397,400 bytes・md5 `1099125190aa8cfd50c868ed577cbf98`・sha256 `041d1c69b189d9bb242adc99f485285f1d2a55767d689fbba71a2fac45508d73` |
| 検査 | `yasakani selfcheck`: 公開RTX 5090参照に対しtop-1 98.9%・sure-changed 0・self-consistency合格・テキスト/画像生成エンドツーエンド検証済み |

他のGPUは**未検証**。ライブラリは `gfx1100` のマシンコードのみ。他のRDNA3カード
（`gfx1101`/`gfx1102`）は再ビルド（`YSKN_HIP_ARCH`・下記）が必要で未検証のまま。

## 必要条件

* ROCm 7.1がサポートするAMD GPUとin-tree `amdgpu` カーネルドライバを備えたLinux x86_64。
  ユーザーは `render` と `video` グループに所属（`sudo usermod -aG render,video $USER` 後に再ログイン）。
* **ROCm 7.1ランタイム — 自身でインストール。非バンドル**（約1.3 GB・カーネルドライバと整合必須）。
  Ubuntu 26.04ではアーカイブにランタイムあり：

  ```sh
  sudo apt install libamdhip64-7 libhipblas3 librocblas5 librocsolver0 libhsa-runtime64-1 libhsakmt1 libdrm-amdgpu1 libroctx64-4
  ```

  （残りの `ldd` 依存 — `libdrm2 libfmt10 libnuma1 libelf1t64 zlib1g libzstd1 libstdc++6
  libgcc-s1 libgomp1 libc6` — は基本システムライブラリ）。他ディストリはAMD公式
  ROCm 7.1インストーラを使用：<https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>。
* Python 3.10+ と `numpy>=1.23`・`tokenizers>=0.15`・`jinja2>=3.1`（画像入力には `Pillow>=10`）。
  一度だけ、システムPythonか任意の仮想環境へ：
  `python3 -m pip install ./python`。`start_amd.sh` は隣の
  `.venv`/`venv` か `YASAKANI_PYTHON` を再利用。モジュール不足時は停止してインストールコマンドを
  表示（自身ではインストールしない）。依存があれば同梱パッケージはインストール無しでも動作
  （`start_amd.sh` が `PYTHONPATH` を設定）。

  **[uv](https://docs.astral.sh/uv/) を使う場合**（推奨・システムPythonを汚さない）：

  ```sh
  uv venv .venv                          # start_amd.sh 隣の私有環境
  uv pip install --python .venv/bin/python ./python   # パッケージ + numpy/tokenizers/jinja2 (+Pillow)
  ./start_amd.sh                         # .venv を自動検出
  ```

  `uv pip install --python .venv/bin/python ./python` は `python/pyproject.toml` を読み、
  yasakaniパッケージと依存を `.venv` へインストール。画像入力には `Pillow` を追加
  （`uv pip install --python .venv/bin/python Pillow`）。別のインタプリタを使う場合は
  `YASAKANI_PYTHON` で指定（例 `YASAKANI_PYTHON=.venv/bin/python ./start_amd.sh`）。
* モデルフォルダ（非バンドル・約10–12 GB・下記）。

`start_amd.sh` は起動前にROCmランタイム（`libamdhip64`）・`/dev/kfd` アクセス・エンジンライブラリ・
Pythonモジュールをチェックし、すべてのエラーに原因と対処を表示。
特殊環境では `YASAKANI_SKIP_ROCM_CHECK=1` でランタイムチェックをスキップ。

## 構成

```
yasakani-hip-0.1.5-linux-x86_64/
├── start_amd.sh          ランチャー: browser / terminal / api（ここから起動）
├── amd_web_chat.py       ブラウザチャットヘルパー（本家quickstartプロキシを再利用）
├── bin/libyskn.so        HIPエンジン（gfx1100）
├── python/               yasakaniパッケージ + pyproject.toml（本家CLI不変）
├── quickstart/           本家 quickstart.py + chat.html（Apache-2.0・無改変再利用）
├── llama-b10488/         画像入力用 llama.cpp b10488 CPU共有ライブラリ（MIT）
├── README.md             英語版
├── README.ja.md          このファイル（日本語版）
├── LICENSE / NOTICE / THIRD_PARTY_NOTICES.md
```

## クイックスタート

```sh
./start_amd.sh                 # 対話式: モデル（スクリプト隣のmodels/・または質問）とモードを選択
```

モデルフォルダが無い場合、対話起動はHugging Faceダウンロードコマンドを表示し、
既存フォルダのパスを尋ねます。

モード（非対話式）：

```sh
./start_amd.sh --mode browser  --model /path/to/model   # 公式Webチャット・ブラウザ起動・URL表示
./start_amd.sh --mode terminal --model /path/to/model   # このターミナルでチャット（/exitで終了）
./start_amd.sh --mode api      --model /path/to/model   # APIサーバーのみ・URL表示・Ctrl+Cで停止
```

`--model` はモデル質問をスキップ。他のオプションは既存CLIへパススルー（例
`--port 8765 --max-ctx 32768 --dll /path/libyskn.so --no-think --api-key KEY`）。
ブラウザチャットは常に127.0.0.1でlistenしAPIキーを送れない — `--host`/`--api-key` は
`--mode api` を使用。`--no-browser` はブラウザを開かずURLのみ表示。

APIモードは `yasakani serve` そのもの：`listening on http://127.0.0.1:8765` と
`ready: OpenAI-compatible API at http://127.0.0.1:8765/v1` を表示、フォアグラウンド実行
（systemd対応）、Ctrl+C / SIGTERMでクリーン停止。リクエスト例：

```sh
curl http://127.0.0.1:8765/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": false}}'
```

### ランチャー無し（素のCLI・不変）

```sh
export PYTHONPATH=$PWD/python
export YASAKANI_DLL=$PWD/bin/libyskn.so
python3 -m yasakani info  --model-dir /path/to/model
python3 -m yasakani serve --model-dir /path/to/model
python3 -m yasakani chat  --model-dir /path/to/model
```

## モデル（別途ダウンロード）

<https://huggingface.co/yasakani-dev> から（Apache-2.0・Qwen3.8-27B派生）：

* `yasakani-dev/Qwen3.8-27B-YASAKANI-speed` — 最速・重み約10.3 GB・16 GB以上GPUに適合。
* `yasakani-dev/Qwen3.8-27B-YASAKANI-quality` — 原モデルに最も近い・重み約12.4 GB・24 GB必要。

```sh
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-speed --local-dir models/qwen38-speed
```

`yasakani.json` を持つフォルダがモデルフォルダ。`start_amd.sh` 隣の `models/` に置くと
自動発見、または `--model` で指定。ダウンロードは現在手動（`hf download` かHF Webページ）。
ランチャー内ガイド付きダウンロードは将来候補。コンテキスト長は起動毎の空きGPUメモリから
選択（`--max-ctx` で上書き）。

### 画像入力（任意）

CPUエンコーダライブラリは同梱（`llama-b10488/`）。2つのGGUFが必要（非バンドル）：

* `--mmproj`: `ggml-org/Qwen3.8-27B-GGUF` の `mmproj-Qwen3.8-27B-BF16.gguf`。
* `--vision-model`: 任意のQwen3.8-27B GGUF（例 `bartowski/Qwen3.8-27B-GGUF` の
  `Qwen3.8-27B-Q4_K_M.gguf`・設定と語彙のみ読み重みは読まない）。

```sh
./start_amd.sh --mode api --model /path/to/model \
  --mmproj /path/mmproj-Qwen3.8-27B-BF16.gguf \
  --llama-dir "$PWD/llama-b10488" \
  --vision-model /path/Qwen3.8-27B-Q4_K_M.gguf \
  --vision-allow-local-files /path/to/images
```

その後 `image_url` 部を持つOpenAI形式リクエストを送信（`data:image/...;base64,...` URIか
`--vision-allow-local-files` 内のローカルファイル）。`Pillow` 必須。ターミナル `chat` は
本家同様テキストのみ。

## Kill switch（実行時・再ビルド不要）

すべてのAMDネイティブ高速パスは無効化可能。その場合エンジンは変換済み本家経路を使用。
CUDAデフォルトは不変。

| Env | 効果 |
|---|---|
| `YSKN_HIP_PREFILL=0` | prefillがPFA/lnord経路へフォールバック |
| `YSKN_ATX_NATIVE=0` | trellis/pk3 attentionが既存ランチャーへフォールバック |
| `YSKN_ATXROW_NATIVE=0` | INT8-row attentionが既存ランチャーへフォールバック |

## 自分のマシンでの検証

```sh
export PYTHONPATH=$PWD/python YASAKANI_DLL=$PWD/bin/libyskn.so
python3 -m yasakani info --model-dir /path/to/model        # GPU・ライブラリchecksum・モデルファイル
python3 -m yasakani verify --model-dir /path/to/model      # モデルchecksum（yasakani.json）
python3 -m yasakani selfcheck --model-dir /path/to/model   # 固定greedyプロンプト vs 参照
```

RX 7900 XTXでは `selfcheck` は**3**で終了：生成テキストがRTX 5090参照と途中から異なる
（GPU演算差）が、参照テキスト上の予測は許容誤差内で一致しself-consistencyも合格 —
これが期待される検証済み結果。exit 1は実際の問題：`info` と `selfcheck` の出力を添えて報告。

## ソースからビルド

ソースツリーの `packaging/build-hip.sh` が検証済み構成でライブラリをビルド
（`hipcc` 由来ROCm `clang`・Release・`gfx1100`。コンパイラパッケージ：
`sudo apt install hipcc libamdhip64-dev libhipblas-dev cmake ninja-build`）：

```sh
./packaging/build-hip.sh                 # -> build-hip/bin/libyskn.so (gfx1100)
YSKN_HIP_ARCH=gfx1101 ./packaging/build-hip.sh   # 他RDNA3: ビルドはできるが未検証
./packaging/make-dist.sh build-hip       # そのビルドで配布物を再組立て
```

ビルドにGPUは不要。**実行と検証には必要**。CIジョブはGPU無しランナーでライブラリを
ビルドできるが、正確性検査（`selfcheck`・生成）は実gfx1100ハードウェアが必要。

## ライセンス

YASAKANI: Apache-2.0（`LICENSE`・`NOTICE`）。`quickstart/` 資産は本家Apache-2.0ファイルを
無改変再利用。`llama-b10488/`: llama.cpp b10488（commit `9d77fa172`）・MIT（`llama-b10488/LICENSE`）。
モデルライセンスと追加帰属：`THIRD_PARTY_NOTICES.md`。

## 非含（設計上）

* GUIと `gui/` ディレクトリなし — ブラウザチャット（`chat.html`）がグラフィカル表面。
* AppImageなし — 将来候補。現在はtar.gzが配布形式。
* Python環境自動ブートストラップとランチャー内モデルダウンロードなし — 将来候補。
  現在は前提を一度インストール（上記）・モデルは `hf download` でダウンロード。
* ROCmランタイム・モデル重みなし（上記）・daemon/モデル管理機能なし。
