# SPDX-License-Identifier: Apache-2.0
# Build libyskn.so for AMD RDNA3 (gfx1100) with the ROCm HIP compiler.
# Building needs no GPU; running does.  The result is bin/libyskn.so inside the build folder.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-$ROOT/build-hip}"
command -v hipcc >/dev/null 2>&1 || { echo "error: hipcc not found. Install the ROCm 7.1 compiler packages (see DISTRIBUTION.md / README.md)." >&2; exit 1; }
# The verified build uses ROCm's clang (not clang++); take it from hipcc so the path is not hardcoded.
HIP_CLANG="$(hipcc -print-prog-name=clang 2>/dev/null || true)"
COMPILER_ARG=()
[ -n "$HIP_CLANG" ] && [ -x "$HIP_CLANG" ] && COMPILER_ARG=(-DCMAKE_HIP_COMPILER="$HIP_CLANG")
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DYSKN_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES="${YSKN_HIP_ARCH:-gfx1100}" \
  ${COMPILER_ARG[@]+"${COMPILER_ARG[@]}"}
cmake --build "$BUILD" -j"${YSKN_BUILD_JOBS:-$(nproc)}"   # lower YSKN_BUILD_JOBS if the heavy TUs exhaust RAM
echo "built: $BUILD/bin/libyskn.so"
echo "note: gfx1100 (RX 7900 XTX) is the verified target; other architectures build but are untested."
