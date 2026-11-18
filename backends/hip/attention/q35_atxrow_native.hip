// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// AMD-native INT8-row KV attention (gfx1100, RDNA3 wave32) for the non-packed
// contexts: the MTP layer (mtp_catchup) and any full layer without trellis-packed
// KV. The converted q35_atx_kernel tiles need 67584B LDS (> gfx1100's 64KB) and
// its NS=1 fallback spills heavily (VGPR192 + 880B/lane scratch) -> ~5x prefill
// regression once it actually runs. This kernel is NOT a CUDA translation: it
// reuses the proven q35_atxpk_native patterns (bf16 WMMA 16x16x16, online
// softmax, per-pair split geometry, conflict-free LDS strides) with the trellis
// decode replaced by exact int8->bf16 dequant loads. Simpler than pk3: the self
// key is an ordinary cache row (attn_post wrote it before attention), so it is
// included in the tile loop -- no self epilogue, no mu term.
// Output contract identical to pk3/combine: pm/pl/pacc[(tk*splits+split)*NH+hd].
#include "q35_atxrow_native.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace yskn {
namespace {

constexpr int NH = Q35_NH, KVH = Q35_KVH, HD = Q35_HD, G = NH / KVH; // 24,4,256,6
constexpr int KT = 16;           // keys per tile
constexpr int QS = 260;          // Q/Kt row stride (bf16): conflict-free b64 (as pk3)
constexpr int VS = 20;           // Vt row stride (bf16)
constexpr int PS = 36;           // P row stride (bf16)
constexpr float L2E = 1.4426950408889634f;

using v4short = short __attribute__((ext_vector_type(4)));
using v16short = short __attribute__((ext_vector_type(16)));
using v8float = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v16short load16(const unsigned short* p) {
  const v4short* q = reinterpret_cast<const v4short*>(p);
  v16short r;
  r.lo.lo = q[0]; r.lo.hi = q[1]; r.hi.lo = q[2]; r.hi.hi = q[3];
  return r;
}
__device__ __forceinline__ v8float wmma(v16short a, v16short b, v8float c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
__device__ __forceinline__ unsigned short f2bf(float f) { // round-to-nearest-even (finite inputs)
  unsigned u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}

struct RowArgs {
  const uint16_t* q16; const int8_t* k8; const int8_t* v8; const float* ks; const float* vs;
  const uint8_t* wk; const int* d_pos;
  float* pm; float* pl; float* pacc;
  int M, max_ctx, splits, splits_short, long_len;
};

__global__ void __launch_bounds__(256) q35_atxrow_native_kernel(RowArgs U) {
  __shared__ __align__(16) unsigned short Q[16 * QS], Kt[KT * QS], Vt[HD * VS], P[16 * PS];
  __shared__ float ksc[KT], vsc[KT], sm[16], sl[16], cf[16];
  __shared__ int s_lo[16], s_ph[16], s_vld[16], s_klo, s_khi;

  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int t0 = blockIdx.x * 2, split = blockIdx.y, kvh = blockIdx.z;
  const int base = *U.d_pos;

  // ---- per-pair row geometry (pk3 spec math; self key INCLUDED: ph = hi) ----
  int act = 0;
  if (tid < 16) {
    const int i = tid, tk = t0 + i / G;
    const bool v = i < 2 * G && tk < U.M;
    const int dp = v ? (U.wk ? (int)U.wk[tk] : tk) : -1;
    const int len = v ? base + dp + 1 : 0;
    const int S = (U.splits_short > 0 && len < U.long_len) ? U.splits_short : U.splits;
    act = v && split < S;
    const int per = (len + S - 1) / S;
    const int lo = split * per, hi = min(len, lo + per);
    s_lo[i] = lo; s_ph[i] = hi; s_vld[i] = v;
    sm[i] = -1e30f; sl[i] = 0.f;
  }
  if (!__syncthreads_or(act)) return;
  if (tid == 0) {
    int klo = 0x7fffffff, khi = 0;
    for (int i = 0; i < 16; ++i) if (s_vld[i] && s_ph[i] > s_lo[i]) {
      klo = min(klo, s_lo[i]); khi = max(khi, s_ph[i]);
    }
    s_klo = klo; s_khi = khi;
  }
  __syncthreads();
  const int klo = s_klo, khi = s_khi;
  const int tbeg = khi > klo ? (klo / KT) * KT : 0;
  const int ntiles = khi > klo ? (khi - tbeg + KT - 1) / KT : 0;

  // ---- Q tile: 16 pair rows of 256 bf16 (invalid rows zero) ----
  for (int c = tid; c < 16 * 32; c += 256) {
    const int p = c >> 5, ch = c & 31;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (s_vld[p]) {
      const int tk = t0 + p / G, hd = kvh * G + p % G;
      v = reinterpret_cast<const uint4*>(U.q16 + ((size_t)tk * NH + hd) * HD)[ch];
    }
    const v4short* h = reinterpret_cast<const v4short*>(&v);
    v4short* dst = reinterpret_cast<v4short*>(Q + p * QS + 8 * ch);
    dst[0] = h[0]; dst[1] = h[1];
  }
  __syncthreads();

  const int8_t* kg8 = U.k8 + (size_t)kvh * U.max_ctx * HD;
  const int8_t* vg8 = U.v8 + (size_t)kvh * U.max_ctx * HD;
  const float* ksg = U.ks + (size_t)kvh * U.max_ctx;
  const float* vsg = U.vs + (size_t)kvh * U.max_ctx;

  v8float o0 = {}, o1 = {}; // PV accumulators: dim tiles 2*wave, 2*wave+1

  for (int t = 0; t < ntiles; ++t) {
    const int k0 = tbeg + t * KT;
    // ---- K load (all waves): int8 -> bf16 exact, scales kept for softmax ----
    {
      const int key = tid >> 4, c16 = tid & 15, kidx = k0 + key;
      unsigned short* krow = Kt + key * QS + 16 * c16;
      if (kidx >= klo && kidx < khi) {
        const int8_t* kr = kg8 + (size_t)kidx * HD + 16 * c16;
#pragma unroll
        for (int i = 0; i < 16; ++i) krow[i] = f2bf((float)kr[i]);
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) krow[i] = 0;
      }
      if (tid < KT) {
        const int kidx2 = k0 + tid;
        const bool in = kidx2 >= klo && kidx2 < khi;
        ksc[tid] = in ? ksg[kidx2] : 0.f;
        vsc[tid] = in ? vsg[kidx2] : 0.f;
      }
    }
    __syncthreads(); // B1: Kt + scales ready

    // ---- QK (wave 0) || V load (waves 1-7, transposed [dim][key]) ----
    v8float c = {};
    if (wave == 0) {
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        const v16short a = load16(Q + (lane % 16) * QS + 16 * ks);
        const v16short b = load16(Kt + (lane % 16) * QS + 16 * ks);
        c = wmma(a, b, c);
      }
    } else {
      for (int u = tid - 32; u < KT * 16; u += 224) {
        const int key = u >> 4, c16 = u & 15, kidx = k0 + key;
        if (kidx >= klo && kidx < khi) {
          const int8_t* vr = vg8 + (size_t)kidx * HD + 16 * c16;
#pragma unroll
          for (int i = 0; i < 16; ++i) Vt[(16 * c16 + i) * VS + key] = f2bf((float)vr[i]);
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) Vt[(16 * c16 + i) * VS + key] = 0;
        }
      }
    }
    __syncthreads(); // B2: scores + Vt ready

    // ---- online softmax (wave 0; C layout: pair = 2g + lane/16, key = lane%16) ----
    if (wave == 0) {
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const int p = 2 * g + lane / 16, key = lane % 16, kidx = k0 + key;
        float s = c[g] * (ksc[key] * 0.0625f);
        s = (kidx >= s_lo[p] && kidx < s_ph[p]) ? s : -INFINITY;
        float mx = s;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float m_old = sm[p], mn = fmaxf(m_old, mx);
        const float cfac = exp2f((m_old - mn) * L2E);
        const float pv = exp2f(s * L2E - mn * L2E);
        float sum = pv;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane % 16 == 0) { sm[p] = mn; sl[p] = sl[p] * cfac + sum; cf[p] = cfac; }
        P[p * PS + key] = f2bf(pv * vsc[key]);
      }
    }
    __syncthreads(); // B3: P + cf + m/l ready

    // ---- PV (all waves): wave w owns dim tiles 2w, 2w+1 ----
    {
      const v16short a = load16(P + (lane % 16) * PS);
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const float f = cf[2 * g + lane / 16];
        o0[g] *= f; o1[g] *= f;
      }
      const v16short b0 = load16(Vt + (16 * (2 * wave) + lane % 16) * VS);
      const v16short b1 = load16(Vt + (16 * (2 * wave + 1) + lane % 16) * VS);
      o0 = wmma(a, b0, o0);
      o1 = wmma(a, b1, o1);
    }
    __syncthreads(); // B4: tiles consumed; next iteration may overwrite
  }

  // ---- outputs (combine contract; no mu/self terms for row KV) ----
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const int p = 2 * g + lane / 16;
    if (s_vld[p]) {
      const int tk = t0 + p / G, hd = kvh * G + p % G;
      const size_t pidx = ((size_t)tk * U.splits + split) * NH + hd;
      U.pacc[pidx * HD + 16 * (2 * wave) + lane % 16] = o0[g];
      U.pacc[pidx * HD + 16 * (2 * wave + 1) + lane % 16] = o1[g];
      if (lane % 16 == 0) { U.pm[pidx] = sm[p]; U.pl[pidx] = sl[p]; }
    }
  }
}

int row_off_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATXROW_NATIVE"); return e ? std::atoi(e) == 0 : 0; }(); return v; }

} // namespace

bool launch_q35_atxrow_native(const Q35AttnArgs& A, const uint8_t* wk, cudaStream_t st) {
  if (row_off_env() || !A.k8 || !A.v8 || !A.ks || !A.vs || A.kv_fp16 || A.M <= 0 || A.splits <= 0) return false;
  RowArgs U{A.q16, A.k8, A.v8, A.ks, A.vs, wk, A.d_pos, A.pm, A.pl, A.pacc,
            A.M, A.max_ctx, A.splits, A.splits_short, A.long_len};
  const dim3 grid((A.M + 1) / 2, A.splits, KVH);
  hipLaunchKernelGGL(q35_atxrow_native_kernel, grid, dim3(256), 0, st, U);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) {
    std::fprintf(stderr, "[atxrow_native] launch fail: M=%d splits=%d grid=(%d,%d,%d): %s -> fallback\n",
                 A.M, A.splits, grid.x, grid.y, grid.z, hipGetErrorString(e));
    return false;
  }
  return true;
}
}
