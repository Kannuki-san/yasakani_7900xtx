# YASAKANI — Linux x86_64 prebuilt (AMD/HIP fork)

Run Qwen3.8-27B locally on one AMD Radeon GPU with the YASAKANI engine: an OpenAI-compatible API
server, a terminal chat and the official browser chat. **No GUI** — the browser chat is the upstream
`chat.html` page served on 127.0.0.1, exactly as on Windows.

This is the AMD/HIP fork of [yasakani-dev/yasakani](https://github.com/yasakani-dev/yasakani).
The HIP backend is **added**, not substituted: the CUDA/NVIDIA path, the Python CLI, the API and all
defaults are unchanged from upstream. The engine library in `bin/` is a HIP build for `gfx1100`.

## Get started (3 steps)

1. **Download** `yasakani-hip-<version>-linux-x86_64.tar.gz` from GitHub Releases and **extract** it.
2. Install the prerequisites once: the **ROCm 7.1 runtime** (one `apt` command below) and the
   **Python dependencies** (`python3 -m pip install ./python` inside the extracted folder).
3. Run **`./start_amd.sh`** — it checks the setup (and shows the fix when something is missing),
   lets you pick a model folder and a mode (browser chat / terminal chat / API server), and prints
   the model download commands when no model is found.

## Verified environment

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XTX (`gfx1100`, 24 GB) — the only verified GPU |
| OS | Ubuntu 26.04.1 LTS, x86-64, kernel 7.0.0 (in-tree `amdgpu` driver) |
| ROCm | 7.1 (Ubuntu packages, e.g. `libamdhip64-7` 7.1.0) |
| Python | 3.14 in a virtual environment (3.10+ required) |
| Engine | `bin/libyskn.so`, 35,397,400 bytes, md5 `1099125190aa8cfd50c868ed577cbf98`, sha256 `041d1c69b189d9bb242adc99f485285f1d2a55767d689fbba71a2fac45508d73` |
| Checks | `yasakani selfcheck`: top-1 98.9% vs the public RTX 5090 reference, sure-changed 0; self-consistency passed; text and image generation verified end-to-end |

Other GPUs are **untested**. The library carries machine code for `gfx1100` only; other RDNA3 cards
(`gfx1101`/`gfx1102`) need a rebuild (`YSKN_HIP_ARCH`, see below) and remain unverified.

## Requirements

* Linux x86_64 with an AMD GPU supported by ROCm 7.1 and the in-tree `amdgpu` kernel driver.
  Your user must be in the `render` and `video` groups (`sudo usermod -aG render,video $USER`, then log in again).
* **ROCm 7.1 runtime — install it yourself; it is not bundled** (~1.3 GB, and it must match the kernel
  driver). On Ubuntu 26.04 the runtime is in the archive:

  ```sh
  sudo apt install libamdhip64-7 libhipblas3 librocblas5 librocsolver0 libhsa-runtime64-1 libhsakmt1 libdrm-amdgpu1 libroctx64-4
  ```

  (the remaining `ldd` dependencies — `libdrm2 libfmt10 libnuma1 libelf1t64 zlib1g libzstd1 libstdc++6
  libgcc-s1 libgomp1 libc6` — are base-system libraries). On other distributions use AMD's official
  ROCm 7.1 installer: <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>.
* Python 3.10+ with `numpy>=1.23`, `tokenizers>=0.15`, `jinja2>=3.1` (and `Pillow>=10` for image input).
  Install once, into the system Python or a virtual environment of your choice:
  `python3 -m pip install ./python`. `start_amd.sh` reuses an existing
  `.venv`/`venv` next to it, or `YASAKANI_PYTHON`; when the modules are missing it stops and shows
  the install command (it does not install anything by itself). With the dependencies present, the
  bundled package also works without installation (`start_amd.sh` sets `PYTHONPATH` itself).

  **With [uv](https://docs.astral.sh/uv/)** (recommended; keeps the system Python untouched):

  ```sh
  uv venv .venv                          # private environment next to start_amd.sh
  uv pip install --python .venv/bin/python ./python   # package + numpy/tokenizers/jinja2 (+Pillow)
  ./start_amd.sh                         # auto-detects .venv
  ```

  `uv pip install --python .venv/bin/python ./python` reads `python/pyproject.toml` and installs the
  yasakani package plus its dependencies into `.venv`. Add `Pillow` for image input
  (`uv pip install --python .venv/bin/python Pillow`). To use a different interpreter, point
  `YASAKANI_PYTHON` at it (e.g. `YASAKANI_PYTHON=.venv/bin/python ./start_amd.sh`).
* A model folder (not bundled, ~10–12 GB; see below).

`start_amd.sh` checks the ROCm runtime (`libamdhip64`), `/dev/kfd` access, the engine library and the
Python modules before starting, and every error names the cause and the fix.
`YASAKANI_SKIP_ROCM_CHECK=1` skips the runtime check for unusual setups.

## Layout

```
yasakani-hip-0.1.5-linux-x86_64/
├── start_amd.sh          launcher: browser / terminal / api (start here)
├── amd_web_chat.py       browser-chat helper (reuses the upstream quickstart proxy)
├── bin/libyskn.so        the HIP engine (gfx1100)
├── python/               the yasakani package + pyproject.toml (unchanged upstream CLI)
├── quickstart/           upstream quickstart.py + chat.html (Apache-2.0, reused unchanged)
├── llama-b10488/         llama.cpp b10488 CPU shared libraries for image input (MIT)
├── README.md             this file
├── LICENSE / NOTICE / THIRD_PARTY_NOTICES.md
```

## Quick start

```sh
./start_amd.sh                 # interactive: picks the model (models/ next to the script, or asks) and the mode
```

With no model folder around, the interactive start shows the Hugging Face download commands and asks
for the path of an existing folder.

Modes (non-interactive):

```sh
./start_amd.sh --mode browser  --model /path/to/model   # official web chat; opens the browser, prints the URL
./start_amd.sh --mode terminal --model /path/to/model   # chat in this terminal (/exit quits)
./start_amd.sh --mode api      --model /path/to/model   # API server only; prints the URL; Ctrl+C stops
```

`--model` skips the model question. Every other option passes through to the existing CLI, e.g.
`--port 8765 --max-ctx 32768 --dll /path/libyskn.so --no-think --api-key KEY`.
The browser chat always listens on 127.0.0.1 and cannot send an API key — use `--mode api` for
`--host`/`--api-key`. `--no-browser` prints the URL without opening a browser.

The API mode is `yasakani serve` itself: it prints `listening on http://127.0.0.1:8765` and
`ready: OpenAI-compatible API at http://127.0.0.1:8765/v1`, runs in the foreground (systemd-friendly)
and shuts down cleanly on Ctrl+C / SIGTERM. Example request:

```sh
curl http://127.0.0.1:8765/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": false}}'
```

### Without the launcher (plain CLI, unchanged)

```sh
export PYTHONPATH=$PWD/python
export YASAKANI_DLL=$PWD/bin/libyskn.so
python3 -m yasakani info  --model-dir /path/to/model
python3 -m yasakani serve --model-dir /path/to/model
python3 -m yasakani chat  --model-dir /path/to/model
```

## Models (download separately)

From <https://huggingface.co/yasakani-dev> (Apache-2.0, derived from Qwen3.8-27B):

* `yasakani-dev/Qwen3.8-27B-YASAKANI-speed` — fastest, ~10.3 GB weights, fits 16 GB+ GPUs.
* `yasakani-dev/Qwen3.8-27B-YASAKANI-quality` — closest to the original, ~12.4 GB weights, needs 24 GB.

```sh
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-speed --local-dir models/qwen38-speed
```

A folder with `yasakani.json` is a model folder; put it in `models/` next to `start_amd.sh` for
automatic discovery, or pass `--model`. Downloading is manual for now (`hf download`, or the
Hugging Face web pages); a guided in-launcher download is a future candidate. The context length is
chosen from the free GPU memory at each start (override with `--max-ctx`).

### Image input (optional)

The CPU encoder libraries are bundled (`llama-b10488/`). Two GGUF files are needed (not bundled):

* `--mmproj`: `mmproj-Qwen3.8-27B-BF16.gguf` from `ggml-org/Qwen3.8-27B-GGUF`.
* `--vision-model`: any Qwen3.8-27B GGUF, e.g. `Qwen3.8-27B-Q4_K_M.gguf` from
  `bartowski/Qwen3.8-27B-GGUF` (only settings and vocabulary are read, never the weights).

```sh
./start_amd.sh --mode api --model /path/to/model \
  --mmproj /path/mmproj-Qwen3.8-27B-BF16.gguf \
  --llama-dir "$PWD/llama-b10488" \
  --vision-model /path/Qwen3.8-27B-Q4_K_M.gguf \
  --vision-allow-local-files /path/to/images
```

Then send an OpenAI-style request with an `image_url` part (a `data:image/...;base64,...` URI, or a
local file inside `--vision-allow-local-files`). `Pillow` must be installed. The terminal `chat`
subcommand is text-only, as upstream.

## Kill switches (runtime, no rebuild)

Every AMD-native fast path can be disabled; the engine then uses the converted upstream path.
CUDA defaults are untouched.

| Env | Effect |
|---|---|
| `YSKN_HIP_PREFILL=0` | prefill falls back to the PFA/lnord path |
| `YSKN_ATX_NATIVE=0` | trellis/pk3 attention falls back to the existing launcher |
| `YSKN_ATXROW_NATIVE=0` | INT8-row attention falls back to the existing launcher |

## Verification on your machine

```sh
export PYTHONPATH=$PWD/python YASAKANI_DLL=$PWD/bin/libyskn.so
python3 -m yasakani info --model-dir /path/to/model        # GPU, library checksum, model files
python3 -m yasakani verify --model-dir /path/to/model      # model checksums (yasakani.json)
python3 -m yasakani selfcheck --model-dir /path/to/model   # fixed greedy prompts vs the reference
```

On the RX 7900 XTX `selfcheck` exits **3**: the generated text differs part-way from the RTX 5090
reference (different GPU arithmetic), but the predictions on the reference text agree within the
tolerances and the self-consistency check passes — this is the expected, verified result. Exit 1
means a real problem: report it with the output of `info` and `selfcheck`.

## Building from source

`packaging/build-hip.sh` in the source tree builds the library with the verified configuration
(ROCm's `clang` from `hipcc`, Release, `gfx1100`; compiler packages:
`sudo apt install hipcc libamdhip64-dev libhipblas-dev cmake ninja-build`):

```sh
./packaging/build-hip.sh                 # -> build-hip/bin/libyskn.so (gfx1100)
YSKN_HIP_ARCH=gfx1101 ./packaging/build-hip.sh   # other RDNA3: builds, but untested
./packaging/make-dist.sh build-hip       # reassemble the distribution with that build
```

Building needs no GPU; **running and verifying do**. A CI job can build the library on a GPU-less
runner, but correctness checks (`selfcheck`, generation) require real gfx1100 hardware.

## Licenses

YASAKANI: Apache-2.0 (`LICENSE`, `NOTICE`). The `quickstart/` assets are the upstream Apache-2.0
files, reused unchanged. `llama-b10488/`: llama.cpp b10488 (commit `9d77fa172`), MIT (`llama-b10488/LICENSE`).
Model licenses and further attributions: `THIRD_PARTY_NOTICES.md`.

## Not included (by design)

* No GUI and no `gui/` directory — the browser chat (`chat.html`) is the graphical surface.
* No AppImage — a future candidate; the tar.gz is the distribution format for now.
* No automatic Python-environment bootstrap and no in-launcher model download — future candidates;
  for now the prerequisites are installed once (above) and models are downloaded with `hf download`.
* No ROCm runtime, no model weights (see above), no daemon or model-management features.
