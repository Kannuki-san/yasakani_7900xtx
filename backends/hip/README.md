# HIP backend (AMD RDNA3)

An independent AMD-native GPU backend that coexists with the upstream CUDA
backend. The CUDA build, its defaults and its runtime behavior are unchanged;
HIP support is added, not substituted. Target: **gfx1100** (Radeon RX 7900 XTX,
wave32). Other HIP architectures are unverified — keep the native options OFF
there (the shared single-source path still compiles, the native kernels are
RDNA3-specific).

## Design: single-source kernels + native additions

The engine keeps one set of kernel sources (`src/*.cu`). They compile as CUDA
with nvcc and as HIP with clang (CMake sets `LANGUAGE HIP` on the same source
list). Two mechanisms make this work without forking the kernels:

1. **`src/gpu_arch.cuh` (`yskn::arch`)** — the only place with NVIDIA PTX
   inline asm. Every architecture primitive (cp.async, mma.sync, ldmatrix,
   barriers, cache hints) has a CUDA branch (the original PTX) and an
   `#if defined(__HIP_PLATFORM_AMD__)` branch (RDNA3 WMMA builtins / LDS reads
   with equivalent semantics). The converted kernels keep NVIDIA fragment
   layouts on both backends; integer paths are bit-exact across backends.
2. **`backends/hip/shim/`** — CUDA-name → HIP-API mapping headers, used only by
   the HIP build (the directory is first on the HIP include path; CUDA builds
   never see it). `-include cuda_runtime.h` reproduces nvcc's implicit runtime
   include. The shim maps only the names the engine actually uses; add a
   mapping when new CUDA API use is introduced. It also carries the
   correctness-critical redirects (IEEE `__fsqrt_rn` via ocml, wave32-safe
   warp-sync masks, software e4m3 fp8 for the KV codec container).

AMD-specific kernels that are *not* translations of CUDA code live here under
`backends/hip/`. They implement the algorithm spec directly on RDNA3 WMMA
layouts (no NVIDIA fragment conversion, no ldmatrix/cp.async):

```
backends/hip/
  shim/        CUDA-name -> HIP-API headers (HIP build only)
  attention/   native trellis/pk3 compressed-KV attention (q35_atxpk_native.hip)
               native INT8-row KV attention (q35_atxrow_native.hip/.hpp)
               CPU geometry self-check (test_atxrow_math.cpp), feature README
  prefill/     hipBLAS int8 per-group prefill GEMM + fused dequant
               (hip_prefill.hip/.hpp, hpf_layout.hpp, test_hpf_math.cpp)
  tests/       standalone probes and CPU references (spike_wmma.hip,
               probe_requant.hip, test_decode_ref.py, test_requant_ref.py,
               analyze_dump.py) — not compiled into the library
```

Shared files touched for HIP (`CMakeLists.txt`, `engine35.cpp`, `lnord.cu`,
`q35_atx.cu`, `q35_atxpk.hpp`, `gpu_arch.cuh`, ...) carry only `#if HIP` guards,
`arch::` substitutions and dispatch hooks; their CUDA view stays equivalent to
upstream (see "CUDA preservation" below).

## Backend selection and build

```bash
# CUDA (default, upstream behavior):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# HIP (RDNA3):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DYSKN_BACKEND=HIP
cmake --build build -j6
```

- `YSKN_BACKEND=CUDA|HIP` (cache, default CUDA). The CUDA branch adds no AMD
  dependency; the HIP branch requires ROCm (clang with HIP support) and links
  `hipblas` only when the prefill GEMM option is ON.
- `CMAKE_HIP_ARCHITECTURES` defaults to `gfx1100`.
- HIP compile flags mirror the CUDA codegen contract: `-O3 -ffp-contract=off`
  (nvcc `--fmad=false` equivalent) and the ocml correctly-rounded-sqrt bitcode.
- HIP-only options (all default ON, GPU-verified on gfx1100):
  - `YSKN_NATIVE_ATXPK` — native trellis/pk3 attention.
  - `YSKN_NATIVE_ATXROW` — native INT8-row KV attention.
  - `YSKN_PREFILL_GEMM` — hipBLAS int8 prefill GEMM.
- `YSKN_KVX_MODE` (KV codec: 0=bf16, 1=fp8, 2/3=rotated INT8, 3=trellis) keeps
  the upstream cache default 3 on both backends.
- HIP-only runtime defaults (CUDA defaults untouched): prefill GEMM and G6
  weight workspace enabled, `YSKN_Q35_LAYER_ROWS=2048`.

## Kill switches and diagnostics (kept intentionally)

Every native fast path can be disabled at runtime without rebuilding; the
engine then uses the existing converted path:

| Env | Effect |
|---|---|
| `YSKN_HIP_PREFILL=0` | prefill falls back to the PFA/lnord path |
| `YSKN_ATX_NATIVE=0` | trellis/pk3 attention falls back to the existing launcher |
| `YSKN_ATXROW_NATIVE=0` | INT8-row attention falls back to the existing launcher |

The native launchers also fall back automatically for unsupported
configurations (see `attention/README.md`). Gated diagnostics remain available:
`YSKN_KDBG`, `YSKN_PROFILE`, `YSKN_HIP_PREFILL_DBG`, `YSKN_ATX_NATIVE_DUMP*`
(with `tests/analyze_dump.py`), all off by default.

## CUDA preservation

- The CUDA preprocessor view of every shared file is equivalent to
  upstream/main: `arch::` substitutions keep the original PTX strings, HIP
  adaptations sit inside `#if defined(__HIP_PLATFORM_AMD__)` guards, and added
  diagnostics are gated (default OFF, no-op on CUDA).
- `gpu_caps.cu` / `c_api.cpp` are byte-identical to upstream in the CUDA view;
  the C ABI is unchanged.
- Static check: fold the HIP guards to their CUDA-side view and diff against
  upstream (the `cuda_view_check.py` approach used during integration).
- **CUDA is UNVERIFIED on real hardware in this fork** (no nvcc available in
  the development environment). Static preservation checks only; a CUDA
  build + run is required before any upstream PR.

## Vision (image input)

Image input is backend-neutral above the C ABI: Python encodes the image with
llama.cpp b10488 (mtmd) and passes bfloat16 features through `yskn_set_visual`;
the engine inserts them at the `<|image_pad|>` markers. No HIP-specific engine
change was needed:

- `q35_visual_rope_kernel` / `q35_visual_embed_kernel` (`src/q35_kernels.cu`)
  are plain index/copy kernels — no WMMA, cp.async, shared memory or warp sync
  — and compile unchanged for gfx1100. `Engine35::set_visual` uses only CUDA
  names the shim maps (`cudaMemcpyAsync`, `cudaFree`, `cudaStreamSynchronize`,
  `cudaGetLastError`). Both kernels (with gfx1100 kernel descriptors) and
  `yskn_set_visual` are present in the GPU-verified HIP library.
- **CPU encoder first (the default)**: the projector runs on the CPU through
  llama.cpp b10488 built as shared libraries *without any GPU backend* —
  verified here: the CPU encoder needs no ggml GPU backend. Recipe (upstream
  BUILDING.md "Image input on Linux" minus `-DGGML_CUDA=ON`):

  ```bash
  git clone --depth 1 --branch b10488 https://github.com/ggml-org/llama.cpp llama-b10488-src
  git -C llama-b10488-src rev-parse --short=9 HEAD    # must print 9d77fa172
  cmake -S llama-b10488-src -B llama-b10488-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  cmake --build llama-b10488-build --target mtmd
  mkdir llama-b10488 && cp -a llama-b10488-build/bin/lib*.so* llama-b10488/
  ```

  `check_build` accepts the result (ggml 0.20.2, commit 9d77fa1). The
  prebuilt llama.cpp libraries of other vendors (e.g. LM Studio backends) are
  other releases and are refused — build b10488 unchanged.
- **GGUF files**: `--mmproj` must be the Qwen3.8-27B projector
  (`clip.vision.projection_dim` 5120). `--vision-model` is opened `no_alloc`:
  only settings and vocabulary are read, never weights, so any Qwen3.8-27B
  GGUF works — verified with a Q4_K_M quant; no full-precision GGUF is needed.
- CPU encoder verification (this fork, 6 threads): features `(n, 5120)` uint16
  bfloat16, finite, mtmd positions validated; 640x480 → 300 tokens (3.2 s),
  1920x1080 → 1008 tokens under the default 1024 limit (15.4 s). Opt-in test
  (skipped without the files): `YASAKANI_TEST_LLAMA_DIR=... YASAKANI_TEST_MMPROJ=...
  YASAKANI_TEST_VISION_MODEL=... python -m unittest tests.test_vision`.
- `YASAKANI_VISION_GPU=1` (projector on the GPU) stays upstream-experimental
  and is **unverified on AMD**: it would need a llama.cpp b10488 build with
  `-DGGML_HIP=ON` (`GPU_TARGETS=gfx1100`), which was not built here. The CPU
  encoder is the supported path.
- End-to-end image → generation on gfx1100: `yasakani serve` with `--mmproj`,
  `--llama-dir`, `--vision-model` and the HIP `libyskn.so` (`YASAKANI_DLL`),
  then an OpenAI-style chat request with an `image_url` part (the `chat`
  subcommand is text-only, as upstream).

## Verification status (gfx1100, v0.1.5 integration base)

- Correctness: all 13 tensors × M=17..512 ALL EXACT (bit-exact vs the lnord
  XSM reference).
- Selfcheck: top-1 98.9% vs the public RTX 5090 reference, sure-changed 0.
- Prefill: 8K 546.5 tok/s (≈18× the converted path), 78K 199 tok/s (≈7.6×).
- Decode: no regression; speculative-decode accept rates maintained.
- Kill switch `YSKN_HIP_PREFILL=0`: clean fallback, no throw.
- CPU self-checks (no GPU needed): `prefill/test_hpf_math.cpp`,
  `attention/test_atxrow_math.cpp`, `tests/test_decode_ref.py` — ALL PASS.
  Build recipes are in each file's header comment.
