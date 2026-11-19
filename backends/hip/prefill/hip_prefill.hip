// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// AMD-native prefill GEMM for gfx1100 (RDNA3, wave32, ROCm 7.1).
//
// Mechanism (source of truth = YASAKANI quantization spec + AMD references):
//   weights: trellis -> int8 via the proven G6 predecode (q35_g6_decode), repacked
//            to row-major [N, Kc]; cached per layer (arena) or per call (capture).
//   GEMM:    per-K-group (Kg=512) signed int8 x int8 -> int32, exact. One
//            hipblasGemmStridedBatchedEx call per tensor, batch = G = Kc/512.
//            rocBLAS 5.1 ships gfx1100 signed-int8 WMMA kernels (I8I_HPA); the
//            parent's bench measured 35-96 TOPS, CPU int32 bit-exact.
//   dequant: acc64[m,n] = sum_g round(alpha[m,g] * 2^24/max_g alpha) * partial_g
//            y[m,n] = float(acc64) * (max*2^-24) [* rsqrt norm when xsm_S set]
//            -- the exact G6/lnord-XSM integer math, so results are bit-identical
//            to the production path before tau[n] (applied by the post kernels).
// Decode (M<=16), bf16 and CUDA are untouched.
#include "hip_prefill.hpp"
#include "q35_g6.hpp"
#include "pm_math.cuh"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace yskn {

// G6 swizzled int8 blocks -> row-major [N, Kc]. Pure uint32 gather (4 consecutive K
// values of one row per word); see hpf_repack_src for the layout derivation.
__global__ void hpf_repack_kernel(const uint4* __restrict__ w8, int N, int Kc, int8_t* __restrict__ rm) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = (size_t)N * (Kc >> 2);
  if (idx >= total) return;
  const int kw = (int)(idx % (Kc >> 2)), n = (int)(idx / (Kc >> 2));
  const unsigned* src = reinterpret_cast<const unsigned*>(w8);
  reinterpret_cast<unsigned*>(rm)[idx] = src[hpf_repack_src(n, kw, Kc)];
}

// Per-group dequant epilogue, G6 float-y contract (post kernels read mode 1/3).
// Arithmetic mirrors mt_gemm6 fill()/epilogue exactly (rounding and order).
__global__ void hpf_dequant_kernel(const int32_t* __restrict__ part, const float* __restrict__ alpha,
                                   const float* __restrict__ S, float* __restrict__ y, float* __restrict__ us_out,
                                   int M, int N, int G, size_t plane, int sst, int nb, float xn, float eps, int pm) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)M * N) return;
  const int m = (int)(idx / N), n = (int)(idx % N);
  const float* al = alpha + (size_t)m * G;
  float mx = 0.f;
  for (int g = 0; g < G; ++g) mx = fmaxf(mx, __ldg(al + g));
  const float rm = mx > 0.f ? __fdiv_rn(16777216.f, mx) : 0.f;
  long long acc = 0;
  for (int g = 0; g < G; ++g) {
    const int a = __float2int_rn(__fmul_rn(__ldg(al + g), rm));
    acc += (long long)a * (long long)__ldg(part + (size_t)g * plane + idx);
  }
  const float us = ldexpf(mx, -24);
  float v = __fmul_rn(__ll2float_rn(acc), us);
  if (S) {
    const float* sr = S + (size_t)m * sst;
    float sum = 0.f;
    for (int b = 0; b < nb; ++b) sum = __fadd_rn(sum, sr[b]);
    const float a = __fadd_rn(__fdiv_rn(sum, xn), eps);
    v = __fmul_rn(v, pm ? pm_rsqrt(a) : rsqrtf(a));
  }
  y[idx] = v;
  if (n == 0 && us_out) us_out[m] = us;
}

static int hpf_dbg_env() { static const int v = [] { const char* e = std::getenv("YSKN_HIP_PREFILL_DBG"); return e ? std::atoi(e) : 0; }(); return v; }
static int hpf_batched_env() { static const int v = [] { const char* e = std::getenv("YSKN_HIP_PREFILL_BATCHED"); return e ? std::atoi(e) != 0 : 1; }(); return v; }
// rocBLAS int8 solutions were GPU-verified up to M=256; larger M is split into row
// blocks of this size (the n=512 solution selection is unverified territory).
static int hpf_mcap_env() { static const int v = [] { const char* e = std::getenv("YSKN_HIP_PREFILL_MCAP");
#if defined(__HIP_PLATFORM_AMD__)
  int def = 512;   // HIP default: single row-block at M=512 halves weight re-reads; +2.4% prefill, hpf_verify bit-exact. Opt-out: =256.
#else
  int def = 256;
#endif
  int m = e ? std::atoi(e) : def; return m >= 16 ? m : def; }(); return v; }

void HipPrefillWorkspace::init(size_t arena_b, size_t tensor_b, size_t partial_b, int mmax, cudaStream_t st) {
  if (ok) return;
  if (hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS) { handle = nullptr; return; }
  if (hipMalloc(&arena, arena_b) != hipSuccess ||
      hipMalloc(&scratch_g6, tensor_b) != hipSuccess ||
      hipMalloc(&scratch_rm, tensor_b) != hipSuccess ||
      hipMalloc(&partials, partial_b) != hipSuccess) { destroy(); return; }
  arena_bytes = arena_b; partial_bytes = partial_b; Mmax = mmax;
  // Warm the rocBLAS workspace with the largest shapes so later graph captures never
  // allocate inside capture (workspace grows monotonically per handle). M stays within
  // the GPU-verified <=256 regime (launch_hip_prefill row-blocks larger M).
  hipblasSetStream(handle, st);
  const int32_t one = 1, zero = 0;
  // largest-N shape (gate/up: N=F_, Kc=5120); tensor_b = F_*5120 for the 27B shapes
  const int Nw = std::max(5120, (int)(tensor_b / 5120)), Kw = 512, Mw = std::min(mmax, 256);
  const int Gw = std::max(1, (int)(partial_b / ((size_t)Mw * Nw * 4)));
  const int Gd = std::max(1, (int)(partial_b / ((size_t)Mw * 5120 * 4)));
  (void)hipblasGemmStridedBatchedEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, Nw, Mw, Kw, &one,
      arena, HIP_R_8I, Kw, Kw, scratch_rm, HIP_R_8I, Kw, Kw, &zero,
      partials, HIP_R_32I, Nw, (long long)Mw * Nw, Gw, HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
  (void)hipblasGemmStridedBatchedEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, 5120, Mw, Kw, &one,
      arena, HIP_R_8I, Kw, Kw, scratch_rm, HIP_R_8I, Kw, Kw, &zero,
      partials, HIP_R_32I, 5120, (long long)Mw * 5120, Gd, HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
  (void)hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, Nw, Mw, Kw, &one,
      arena, HIP_R_8I, Kw, scratch_rm, HIP_R_8I, Kw, &zero,
      partials, HIP_R_32I, Nw, HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
  (void)hipDeviceSynchronize();
  (void)hipGetLastError();   // warmup data is garbage; only the workspace matters
  ok = true;
  std::fprintf(stderr, "[hip_prefill] arena %.1f MB, tensor scratch %.1f MB x2, partials %.1f MB, Mmax %d\n",
               arena_b / 1e6, tensor_b / 1e6, partial_b / 1e6, mmax);
}

void HipPrefillWorkspace::destroy() {
  if (handle) { hipblasDestroy(handle); handle = nullptr; }
  if (arena) { hipFree(arena); arena = nullptr; }
  if (scratch_g6) { hipFree(scratch_g6); scratch_g6 = nullptr; }
  if (scratch_rm) { hipFree(scratch_rm); scratch_rm = nullptr; }
  if (partials) { hipFree(partials); partials = nullptr; }
  cached.clear(); used = 0; ok = false;
}

bool launch_hip_prefill(const GemvLaunch& P, HipPrefillWorkspace& W, int pm, cudaStream_t st) {
  if (!W.ok || !P.xsum || P.xsm_part || P.lo || P.fuse || P.draft || P.e_fuse || P.f_mode ||
      P.M < 17 || P.M > W.Mmax || P.n_seg < 1 || P.n_seg > 3 || !P.us_out) return false;
  const int Kc = P.seg[0].t.Kc;
  if (Kc < 512 || Kc % 512) return false;
  for (int s = 0; s < P.n_seg; ++s) {
    const TensorDev& t = P.seg[s].t;
    if (!t.lno || !t.lut8 || t.tile_k || t.group_worker || t.Kc != Kc || t.N <= 0 || t.N % 16 ||
        (t.k_default != 4 && t.k_default != 6 && t.k_default != 8)) return false;
    if ((size_t)(Kc / 512) * P.M * t.N * 4 > W.partial_bytes) return false;
  }
  // Graph-captured passes must not depend on host-side cache state (replay re-runs the
  // recorded kernels only): decode+repack into the scratch every call instead.
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(st, &cap) != cudaSuccess) return false;
  const bool capture = cap == cudaStreamCaptureStatusActive;

  hipblasSetStream(W.handle, st);
  const int32_t one = 1, zero = 0;
  const int G = Kc / 512, mcap = hpf_mcap_env();
  { const cudaError_t p = cudaPeekAtLastError();
    if (p != cudaSuccess) std::fprintf(stderr, "[hip_prefill] pre-existing error before launch (M=%d, engine-side): %s\n", P.M, cudaGetErrorString(p)); }
  // Any launch error is attributed to its stage and turned into a graceful fallback
  // (the engine re-runs the whole GEMV on the existing path) instead of a sticky
  // error that would abort prefill_vsp at its final check.
  auto stage_ok = [&](const char* stage, int N, int mb) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    if (hpf_dbg_env()) std::fprintf(stderr, "[hip_prefill] %s error (N=%d M=%d): %s -> fallback\n", stage, N, mb, cudaGetErrorString(e));
    return false;
  };
  for (int s = 0; s < P.n_seg; ++s) {
    const TensorDev& t = P.seg[s].t;
    const int N = t.N;
    const int8_t* w8;
    if (capture) {
      q35_g6_decode(t, W.scratch_g6, st);
      const size_t words = (size_t)N * (Kc >> 2);
      hpf_repack_kernel<<<(unsigned)((words + 255) / 256), 256, 0, st>>>(W.scratch_g6, N, Kc, W.scratch_rm);
      w8 = W.scratch_rm;
    } else {
      const auto it = W.cached.find(t.lno);
      if (it != W.cached.end()) w8 = it->second;
      else {
        const size_t bytes = (size_t)N * Kc;
        if (bytes > W.arena_bytes) return false;
        if (W.used + bytes > W.arena_bytes) { W.cached.clear(); W.used = 0; }   // tensors arrive layer by layer
        q35_g6_decode(t, W.scratch_g6, st);
        const size_t words = (size_t)N * (Kc >> 2);
        hpf_repack_kernel<<<(unsigned)((words + 255) / 256), 256, 0, st>>>(W.scratch_g6, N, Kc, W.arena + W.used);
        w8 = W.arena + W.used;
        W.cached[t.lno] = w8; W.used += bytes;
      }
    }
    if (!stage_ok("decode/repack", N, P.M)) return false;
    for (int r0 = 0; r0 < P.M; r0 += mcap) {
      const int mb = std::min(mcap, P.M - r0);
      const int8_t* qx = P.qx + (size_t)r0 * Kc;
      const size_t plane = (size_t)mb * N;
      // col-major: Y_g[N,mb] = W_g^T[N,512] x X_g[512,mb]; W row-major [N,Kc] = col [Kc,N]
      // lda=Kc opT; qx row-major [mb,Kc] = col [Kc,mb] ldb=Kc opN; C col [N,mb] ldc=N = row [mb,N].
      // Batched first; if the backend refuses it, G plain GemmEx calls (the parent-bench-proven
      // API) write the same [G,mb,N] partials layout, so the dequant below is identical.
      hipblasStatus_t bs = HIPBLAS_STATUS_SUCCESS;
      if (hpf_batched_env())
        bs = hipblasGemmStridedBatchedEx(W.handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, mb, 512, &one,
            w8, HIP_R_8I, Kc, 512, qx, HIP_R_8I, Kc, 512, &zero,
            W.partials, HIP_R_32I, N, (long long)plane, G, HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
      if (bs != HIPBLAS_STATUS_SUCCESS) {
        if (hpf_dbg_env()) std::fprintf(stderr, "[hip_prefill] batched status %d (N=%d M=%d G=%d) -> plain GemmEx loop\n", (int)bs, N, mb, G);
        (void)hipGetLastError();
        for (int g = 0; g < G; ++g) {
          bs = hipblasGemmEx(W.handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, mb, 512, &one,
              w8 + (size_t)g * 512, HIP_R_8I, Kc, qx + (size_t)g * 512, HIP_R_8I, Kc, &zero,
              W.partials + (size_t)g * plane, HIP_R_32I, N, HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
          if (bs != HIPBLAS_STATUS_SUCCESS) {
            if (hpf_dbg_env()) std::fprintf(stderr, "[hip_prefill] gemm status %d (N=%d M=%d g=%d) -> fallback\n", (int)bs, N, mb, g);
            (void)hipGetLastError();
            return false;
          }
        }
      }
      if (!stage_ok("gemm", N, mb)) return false;
      hpf_dequant_kernel<<<(unsigned)((plane + 255) / 256), 256, 0, st>>>(W.partials,
          P.alpha + (size_t)r0 * G, P.xsm_S ? P.xsm_S + (size_t)r0 * P.xsm_sst : nullptr,
          P.seg[s].part + (size_t)r0 * N, P.us_out + r0, mb, N, G, plane, P.xsm_sst, P.xsm_nb, P.xsm_n, P.xsm_eps, pm);
      if (!stage_ok("dequant", N, mb)) return false;
    }
  }
  return true;
}
}
