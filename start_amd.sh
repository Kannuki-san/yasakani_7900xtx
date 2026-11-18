#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
# YASAKANI AMD/HIP launcher for Linux x86_64.  No GUI: it reuses the existing Python CLI.
#   browser  -> amd_web_chat.py (the upstream quickstart web proxy + chat.html) + yasakani serve
#   terminal -> yasakani chat
#   api      -> yasakani serve   (prints the URL, handles Ctrl+C/SIGTERM itself, foreground)
# Checks the ROCm runtime and the Python dependencies up front and shows the fix when something is
# missing.  Every option not listed below is passed through to the underlying command unchanged.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

usage() {
  cat <<'EOF'
YASAKANI AMD/HIP launcher (Linux x86_64, ROCm 7.1 runtime required)

Usage: ./start_amd.sh                          interactive: pick a model folder and a mode
       ./start_amd.sh --mode MODE [options]    non-interactive

Modes:
  --mode browser    official web chat in your browser (starts yasakani serve + the chat page)
  --mode terminal   chat in this terminal (yasakani chat)
  --mode api        OpenAI-compatible API server only (yasakani serve; prints the URL, no browser)

Launcher options:
  --model DIR       model folder with yasakani.json (skips the model question; alias: --model-dir)
  --no-browser      with --mode browser: do not open a browser window (the URL is printed)
  --help            this text

Everything else is passed through to yasakani serve / chat, e.g.:
  --host --port --api-key --max-ctx --device --dll --mmproj --llama-dir --vision-model
  --vision-allow-local-files --no-warmup --no-think --max-tokens ...
  (browser chat always listens on 127.0.0.1 and cannot send an API key: use --mode api for those)

Prerequisites (checked at start, the error shows the fix):
  * ROCm 7.1 runtime + /dev/kfd access (see README.md for the apt command).
  * Python 3.10+ with the dependencies: python3 -m pip install ./python
    (an existing venv next to this script, or YASAKANI_PYTHON, is reused).
  * A model folder: --model DIR, or pick one interactively.
    Download: https://huggingface.co/yasakani-dev (speed ~10.3 GB / quality ~12.4 GB).

Environment:
  YASAKANI_PYTHON     Python interpreter to use (default: .venv/venv next to this script, else python3)
  YASAKANI_DLL        path of libyskn.so (default: bin/ or build-publish/bin/ next to this script)
  YASAKANI_MODEL_DIR  default model folder
  YASAKANI_MMPROJ / YASAKANI_LLAMA_DIR / YASAKANI_VISION_MODEL   image input (see README.md)
  YASAKANI_SKIP_ROCM_CHECK=1   skip the ROCm runtime check (unusual setups only)

Examples:
  ./start_amd.sh --mode api --model ~/models/qwen38-speed --port 8765
  ./start_amd.sh --mode browser --model ~/models/qwen38-speed
  ./start_amd.sh --mode terminal --model ~/models/qwen38-speed --no-think
EOF
}

# ---- arguments: launcher flags are consumed, the rest passes through --------------------------------
MODE="" MODEL="" NO_BROWSER=0 PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)    [ $# -ge 2 ] || die "--mode needs a value (browser|terminal|api)"; MODE="$2"; shift 2 ;;
    --model|--model-dir) [ $# -ge 2 ] || die "$1 needs a folder path"; MODEL="$2"; shift 2 ;;
    --no-browser) NO_BROWSER=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) PASS+=("$1"); shift ;;
  esac
done
if [ -n "$MODE" ]; then
  case "$MODE" in browser|terminal|api) ;; *) die "unknown --mode '$MODE' (use browser, terminal or api)" ;; esac
fi
DLL_ARG=0
case " ${PASS[*]-} " in *" --dll "*|*" --dll="*) DLL_ARG=1 ;; esac

# ---- Python: an existing environment with the dependencies (venv next to the script, or python3) ----
if [ -f "$ROOT/python/yasakani/__init__.py" ]; then
  export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
fi
modules_ok() {  # find_spec only: nothing is imported, no GPU is touched
  "$1" - <<'EOF' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in ("yasakani", "numpy", "tokenizers", "jinja2")) else 1)
EOF
}
BASE_PY="$(command -v python3 || true)"
PY=""
if [ -n "${YASAKANI_PYTHON:-}" ]; then
  # an explicit interpreter is honored, never silently replaced
  [ -x "$YASAKANI_PYTHON" ] || die "YASAKANI_PYTHON ($YASAKANI_PYTHON) is not an executable Python interpreter."
  modules_ok "$YASAKANI_PYTHON" \
    || die "YASAKANI_PYTHON ($YASAKANI_PYTHON) lacks the Python modules (yasakani numpy tokenizers jinja2). Install them into it: $YASAKANI_PYTHON -m pip install numpy tokenizers jinja2"
  PY="$YASAKANI_PYTHON"
else
  for c in "$ROOT/.venv/bin/python" "$ROOT/venv/bin/python" "$ROOT/zluda-experiment/venv/bin/python" ${BASE_PY:+"$BASE_PY"}; do
    if [ -x "$c" ] && modules_ok "$c"; then PY="$c"; break; fi
  done
  if [ -z "$PY" ]; then
    [ -n "$BASE_PY" ] || die "no Python interpreter found. Install Python 3.10+ (Ubuntu: sudo apt install python3 python3-venv), or set YASAKANI_PYTHON."
    MISSING="$("$BASE_PY" - <<'EOF' 2>/dev/null
import importlib.util
print(" ".join(m for m in ("yasakani", "numpy", "tokenizers", "jinja2") if importlib.util.find_spec(m) is None))
EOF
)"
    die "Python modules missing: ${MISSING:-unknown}. Install the bundled package and its dependencies: $BASE_PY -m pip install $ROOT/python (or: $BASE_PY -m pip install numpy tokenizers jinja2; Pillow too for image input). An existing environment can be selected with YASAKANI_PYTHON."
  fi
fi

# ---- ROCm runtime + GPU access (cause + fix, before anything loads) ----------------------------------
if [ "${YASAKANI_SKIP_ROCM_CHECK:-0}" != 1 ]; then
  if ! ldconfig -p 2>/dev/null | grep -q 'libamdhip64'; then
    die "ROCm runtime not found (libamdhip64). Install ROCm 7.1 - Ubuntu 26.04: sudo apt install libamdhip64-7 libhipblas3 librocblas5 librocsolver0 libhsa-runtime64-1 libhsakmt1 libdrm-amdgpu1 libroctx64-4 ; other distros: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/ . Unusual setup? YASAKANI_SKIP_ROCM_CHECK=1 skips this check."
  fi
  if [ ! -e /dev/kfd ]; then
    die "/dev/kfd not found: the amdgpu kernel driver is not loaded or no AMD GPU is visible. Check: lsmod | grep amdgpu ; ROCm installation: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
  fi
  if [ ! -r /dev/kfd ] || [ ! -w /dev/kfd ]; then
    die "no read/write access to /dev/kfd. Add your user to the render and video groups: sudo usermod -aG render,video \$USER  - then log out and back in."
  fi
fi

# ---- engine library ----------------------------------------------------------------------------------
if [ -z "${YASAKANI_DLL:-}" ] && [ "$DLL_ARG" = 0 ]; then
  for c in "$ROOT/bin/libyskn.so" "$ROOT/build-publish/bin/libyskn.so" "$ROOT/build-integ/bin/libyskn.so" "$ROOT/build/bin/libyskn.so"; do
    if [ -f "$c" ]; then export YASAKANI_DLL="$c"; break; fi
  done
fi
if [ "$DLL_ARG" = 0 ]; then
  [ -n "${YASAKANI_DLL:-}" ] && [ -f "${YASAKANI_DLL}" ] \
    || die "engine library libyskn.so not found. Set YASAKANI_DLL=/path/to/libyskn.so, pass --dll, or build it (packaging/build-hip.sh). See README.md."
fi

# ---- model folder: --model, env, discovery, or a guided first-time pick/download ---------------------
[ -n "$MODEL" ] || MODEL="${YASAKANI_MODEL_DIR:-}"
if [ -z "$MODEL" ]; then
  CANDS=()
  for d in "$ROOT/models" "$ROOT/zluda-experiment/models"; do
    [ -d "$d" ] || continue
    for m in "$d"/*/; do
      [ -f "${m}yasakani.json" ] && CANDS+=("${m%/}")
    done
  done
  if [ "${#CANDS[@]}" -eq 1 ]; then
    MODEL="${CANDS[0]}"
    printf 'model: %s\n' "$MODEL"
  elif [ -t 0 ]; then
    if [ "${#CANDS[@]}" -gt 1 ]; then
      printf 'Model folders found:\n'
      for i in "${!CANDS[@]}"; do printf '  %d) %s\n' "$((i + 1))" "${CANDS[$i]}"; done
      read -r -p "Select [1-${#CANDS[@]}]: " n || die "no selection made"
      case "$n" in ''|*[!0-9]*) die "invalid selection '$n'" ;; esac
      [ "$n" -ge 1 ] && [ "$n" -le "${#CANDS[@]}" ] || die "invalid selection '$n'"
      MODEL="${CANDS[$((n - 1))]}"
    else
      printf 'No model folder found in models/ next to this script.\n'
      printf 'Download one from https://huggingface.co/yasakani-dev (not bundled):\n'
      printf '  hf download yasakani-dev/Qwen3.8-27B-YASAKANI-speed   --local-dir models/qwen38-speed    (~10.3 GB, 16 GB+ GPU)\n'
      printf '  hf download yasakani-dev/Qwen3.8-27B-YASAKANI-quality --local-dir models/qwen38-quality  (~12.4 GB, 24 GB+ GPU)\n'
      read -r -p "Path to an existing model folder (with yasakani.json): " MODEL || die "no model folder given"
    fi
  else
    die "no model folder given and stdin is not a terminal. Use --model /path/to/model (or set YASAKANI_MODEL_DIR). Download: hf download yasakani-dev/Qwen3.8-27B-YASAKANI-speed --local-dir models/qwen38-speed (https://huggingface.co/yasakani-dev)"
  fi
fi
[ -n "$MODEL" ] || die "no model folder given. Use --model /path/to/model (or set YASAKANI_MODEL_DIR)."
[ -f "$MODEL/yasakani.json" ] || die "'$MODEL' is not a model folder (yasakani.json missing). Download a model from https://huggingface.co/yasakani-dev and point --model at its folder."
MODEL="$(cd "$MODEL" && pwd)"

# ---- mode --------------------------------------------------------------------------------------------
if [ -z "$MODE" ]; then
  [ -t 0 ] || die "no --mode given and stdin is not a terminal. Use --mode browser|terminal|api."
  printf 'YASAKANI (AMD/HIP) - how do you want to use the model?\n'
  printf '  1) Browser chat   (the official web chat opens in your browser)\n'
  printf '  2) Terminal chat  (chat in this window; /exit quits)\n'
  printf '  3) API server     (OpenAI-compatible; prints the URL, no browser; Ctrl+C stops)\n'
  read -r -p "Select [1-3, default 1]: " choice || die "no selection made"
  case "${choice:-1}" in
    1) MODE=browser ;; 2) MODE=terminal ;; 3) MODE=api ;;
    *) die "invalid selection '$choice' (use 1, 2 or 3)" ;;
  esac
fi

# ---- run: exec keeps signals and exit codes with the real command ------------------------------------
case "$MODE" in
  api)      exec "$PY" -m yasakani serve --model-dir "$MODEL" ${PASS[@]+"${PASS[@]}"} ;;
  terminal) exec "$PY" -m yasakani chat --model-dir "$MODEL" ${PASS[@]+"${PASS[@]}"} ;;
  browser)
    EXTRA=()
    [ "$NO_BROWSER" = 1 ] && EXTRA+=(--no-browser)
    exec "$PY" "$ROOT/amd_web_chat.py" --model-dir "$MODEL" ${EXTRA[@]+"${EXTRA[@]}"} ${PASS[@]+"${PASS[@]}"}
    ;;
esac
