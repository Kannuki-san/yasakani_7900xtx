#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Assemble the Linux x86_64 prebuilt distribution (idempotent: run it again any time).
#   ./packaging/make-dist.sh [BUILD_DIR]
# BUILD_DIR defaults to build-publish, then build-integ; it must contain bin/libyskn.so
# (a GPU-verified build, or one made with packaging/build-hip.sh).
# Output: dist/yasakani-hip-<version>-linux-x86_64/ + .tar.gz + SHA256SUMS (all git-excluded).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-${YSKN_BUILD_DIR:-$ROOT/build-publish}}"
[ -f "$BUILD/bin/libyskn.so" ] || BUILD="$ROOT/build-integ"
LIB="$BUILD/bin/libyskn.so"
[ -f "$LIB" ] || { echo "error: no bin/libyskn.so in the build folder. Build it first: ./packaging/build-hip.sh" >&2; exit 1; }

VER="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$ROOT/python/yasakani/__init__.py")"
[ -n "$VER" ] || { echo "error: cannot read __version__ from python/yasakani/__init__.py" >&2; exit 1; }
NAME="yasakani-hip-$VER-linux-x86_64"
OUT="$ROOT/dist"
STAGE="$OUT/$NAME"
case "$STAGE" in "$OUT/"*) ;; *) echo "error: unexpected stage path $STAGE" >&2; exit 1 ;; esac
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/python" "$STAGE/quickstart" "$STAGE/llama-b10488"

cp "$LIB" "$STAGE/bin/libyskn.so"
cp -r "$ROOT/python/yasakani" "$STAGE/python/"
cp "$ROOT/python/pyproject.toml" "$ROOT/python/README.md" "$STAGE/python/"
cp "$ROOT/start_amd.sh" "$ROOT/amd_web_chat.py" "$STAGE/"
cp "$ROOT/quickstart/quickstart.py" "$ROOT/quickstart/chat.html" "$STAGE/quickstart/"
cp "$ROOT"/llama-b10488/lib*.so* "$STAGE/llama-b10488/"
if [ -f "$ROOT/llama-b10488/LICENSE" ]; then
  cp "$ROOT/llama-b10488/LICENSE" "$STAGE/llama-b10488/LICENSE"
else
  cp "$ROOT/llama-b10488-src/LICENSE" "$STAGE/llama-b10488/LICENSE"   # llama.cpp b10488, MIT
fi
cp "$ROOT/packaging/DISTRIBUTION.md" "$STAGE/README.md"
cp "$ROOT/packaging/DISTRIBUTION.ja.md" "$STAGE/README.ja.md"
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$ROOT/THIRD_PARTY_NOTICES.md" "$STAGE/"

find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
chmod -R a+rX "$STAGE"
chmod a+x "$STAGE/start_amd.sh" "$STAGE/amd_web_chat.py"

# hygiene: no GUI, no model weights, no caches in the archive
if find "$STAGE" \( -name '*.yskn' -o -name '*.gguf' -o -name 'gui' -o -name '__pycache__' -o -name 'local-env.sh' \) | grep -q .; then
  echo "error: forbidden files in the stage tree (see above)" >&2; exit 1
fi
# shared-library sanity: everything must resolve on a host with the ROCm 7.1 runtime installed
MISSING="$(ldd "$STAGE/bin/libyskn.so" "$STAGE"/llama-b10488/*.so* 2>/dev/null | grep 'not found' || true)"
if [ -n "$MISSING" ]; then
  echo "warning: unresolved shared libraries (install the ROCm 7.1 runtime, see README.md):" >&2
  echo "$MISSING" >&2
fi

tar -czf "$OUT/$NAME.tar.gz" --owner=0 --group=0 --numeric-owner -C "$OUT" "$NAME"
(cd "$OUT" && sha256sum "$NAME.tar.gz" > SHA256SUMS && sha256sum -c SHA256SUMS)
echo "staged:  $STAGE"
echo "archive: $OUT/$NAME.tar.gz ($(stat -c%s "$OUT/$NAME.tar.gz") bytes)"
echo "dll:     $(md5sum "$LIB" | cut -d' ' -f1) (md5)  $(sha256sum "$LIB" | cut -d' ' -f1) (sha256)"
