# AMD-native trellis/pk3 attention (RDNA3 / gfx1100)

Native realization of the YASAKANI pk3 compressed-KV attention mechanism.
Not a CUDA-to-HIP translation: the CUDA kernel structure, fragment layouts and
processing order are not reproduced. Source of truth = the algorithm spec
(`src/q35_atxpk3.cuh` kernel math, `src/q35_atxpk.cu` decode math,
`src/kvq_atxpk.hpp` record format) + AMD official references.

## Files
- `q35_atxpk_native.hip` — kernel + `launch_q35_atx_pk_native` entry point.
- `../tests/spike_wmma.hip` — minimal WMMA builtin verification (bf16/int8 16x16x16).
- `../tests/test_decode_ref.py` — independent CPU reference for the trellis decode
  (3 formulations + encoder roundtrip + model blob parse). Run with any python3.

## Design
- WMMA: `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` (wave32), native RDNA3
  layout per AMD ROCm Optimization Guide "RDNA3 dense WMMA builtins":
  A row / B column = lane%16 (replicated across lane halves),
  C[row=2g+lane/16][col=lane%16]. Operands stored `[index][k]` row-major in LDS;
  fragment loads = 4×ds_read_b64 with strides chosen conflict-free
  (Q/Kt stride 260 bf16 ≡ 2 mod 32 dwords, Vt 20, P 36).
- Tiling: workgroup 256 threads (8 waves), 2 query tokens × 6 GQA heads = 12
  pairs padded to 16; grid (ceil(M/2), splits, KVH); KV tile = 16 keys.
  Per tile: all waves decode K → wave0 QK ∥ waves1-7 decode V (transposed Vt)
  → wave0 online softmax (m/l in LDS, 16-lane shfl reductions) → all waves PV
  (2 dim tiles each). 4 barriers/tile, single-buffered (~31 KB LDS).
- Trellis decode fused into tile load: 16 values/unit, funnelshift window
  extraction, **state resets every 64 values** (4 independent encoder chains),
  bf16 codebook LUT in LDS. No separate decode pass, no CUDA ldmatrix/cp.async.
- Epilogues: self position (dot + online-softmax update + PV), muV (`o += l*mu`),
  pm/pl/pacc stores compatible with the common `launch_q35_combine`.
- Debug: `YSKN_ATX_NATIVE_DUMP=1` (or a path) dumps the selected WG's tile-0
  decoded K/V, scaled+masked scores, m/l, final pm/pl/pacc, plus self-contained
  inputs (packed rows, LUTs, Q, geometry, mu, self rows) and per-tile regions
  (up to 4 tiles: packed inputs, scores, m/l, cf, scales; full V int8 rows for
  vpk=0). `YSKN_ATX_NATIVE_DUMP_SEQ=1` writes numbered files per non-captured
  call; `YSKN_ATX_NATIVE_DUMP_SELF=1` auto-selects the token0/kvh0 self WG;
  `YSKN_ATX_NATIVE_DUMP_BLOCK=x,y,z` selects a fixed WG;
  `YSKN_ATX_NATIVE_DUMP_LAYER=n` selects the layer (default 0).
  Buffer is process-lifetime, allocated outside graph capture; captured
  launches reuse it without readback. `backends/hip/tests/analyze_dump.py`
  recomputes the whole chain on CPU (stages 1-7, incl. multi-tile online
  softmax accumulation -> PV -> self -> mu -> pacc/pm/pl).

## Integration
- `YSKN_NATIVE_ATXPK` (CMake, default ON in HIP builds) compiles this file and
  routes the engine's two `launch_q35_atx_pk` call sites through
  `launch_q35_atx_pk_dispatch` → native. CUDA builds always use the original
  launcher (the dispatch is a direct call there).
- Runtime kill switch: `YSKN_ATX_NATIVE=0` → existing launcher.
- Fallback to the existing launcher: kb≠5, K not packed, no rec/kps, kv_fp16,
  `YSKN_ATX_PK3_QK8=1` (int8-QK experiment), M≤0 / splits≤0.
- V container: packed trellis (vpk=1) or int8 (vpk=0: layers whose V group
  has projection vectors, e.g. layers 10/14/15 of qwen38-speed; no mu term).
- CUDA build, bf16 path (KVX_MODE=0) and lnord optimizations: untouched.

## Verification status
- CPU decode reference: PASS (200 random rows × 3 formulations, 200 roundtrips;
  blob parse with the qwen38-speed kvq_u5.bin: 128 groups kk=5, 13/16 layers
  pk-packable).
- WMMA spike: compiles for gfx1100; ISA shows v_wmma_f32_16x16x16_bf16 and
  v_wmma_i32_16x16x16_iu8; runtime ALL PASS on gfx1100.
- Integration branch (v0.1.5 base, gfx1100): GPU-verified — correctness all
  tensors ALL EXACT, selfcheck top-1 98.9% vs the public RTX 5090 reference
  (sure-changed 0), no decode regression, speculative accept maintained.

## Known ceilings (ponytail)
- Single-buffered tiles, scalar b16 LUT lookups, wave0-only QK/softmax:
  correctness-first. Upgrade: double-buffer Kt/Vt + direct-to-LDS DMA,
  b128 swizzled fragment loads, balanced QK across waves — after selfcheck.
- RDNA3 only (gfx1100/01/02). Other HIP archs must keep the option OFF.
