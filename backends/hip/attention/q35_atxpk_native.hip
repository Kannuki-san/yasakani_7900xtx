// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// AMD-native trellis/pk3 compressed-KV attention (KVX_MODE=3, kb=5, K packed;
// V packed trellis or V int8). NOT a CUDA translation: native RDNA3 WMMA
// layout, native LDS tiling, wave32.
// RDNA3 (gfx1100/gfx1101/gfx1102) only; other targets must not enable this file.
//
// Mechanism (source of truth = algorithm spec + AMD official references):
//   * packed row = 160B = 40 words; value e lives in trellis chunk e>>6 (state
//     resets every 64 values); symbol s[e] = bits[5*(e&63)-3 .. 5*(e&63)+4] of
//     the chunk (MSB-first bit numbering), top 3 bits zero at chunk start.
//     decoded value = codebook_bf16[s[e]]  (rec[0..511]).
//   * score = (Q_bf16 . K_dec) * kps[row] * (1/16)          [1/sqrt(HD)]
//   * o = sum p * vps[row] * V_dec_bf16  +  l * muV[dim]    (muV = rec_v+512)
//   * online softmax (exp2f), causal range [lo,ph) per (token,head) pair,
//     self position (len-1) handled in an epilogue, split-KV outputs
//     pm/pl/pacc (unnormalized) consumed by the common combine kernel.
//
// WMMA layout (AMD ROCm Optimization Guide, "RDNA3 dense WMMA builtins"):
//   __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32, wave32, matrix replication:
//   A: lane L holds row L%16 (16 bf16); B: lane L holds column L%16 of the
//   K x N tile (16 bf16); lanes 0-15 / 16-31 carry identical copies.
//   C: lane L, VGPR g -> row 2g + L/16, col L%16 (8 f32).
//   Both operands are therefore stored [index][k] row-major in LDS:
//   Q[pair][hd], Kt[key][hd] (QK: M=pair,N=key,K=hd), Vt[dim][key] (PV:
//   M=pair,N=dim,K=key), P[pair][key]. Fragment loads = 4 x ds_read_b64.
//
// Workgroup = 256 threads (8 waves), 2 query tokens x 6 GQA heads = 12 valid
// pairs padded to 16. Grid = (ceil(M/2), splits, KVH). KV tile = 16 keys.
// Per tile: all waves decode K; wave 0 runs QK while waves 1-7 decode V;
// wave 0 does online softmax (m/l live in LDS, per-pair values replicated
// across the 16-lane half by shfl_xor reduction); all waves run PV on their
// two 16-dim output tiles. Self/mu epilogue, then pm/pl/pacc store.
//
// ponytail: correctness-first single-buffer tiling, scalar b16 LUT lookups,
// no async pipeline. Upgrade path: double-buffer Kt/Vt + global->LDS DMA
// (direct-to-lds builtins) and b128 swizzled fragment loads after selfcheck.
#include "q35_atxpk.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace yskn {
namespace {

constexpr int NH = Q35_NH, KVH = Q35_KVH, HD = Q35_HD, G = NH / KVH; // 24,4,256,6
constexpr int PB = 160;          // packed row bytes, kb=5
constexpr int KT = 16;           // keys per tile
constexpr int QS = 260;          // Q/Kt row stride (bf16): 130 dw = 2 mod 32 -> conflict-free b64
constexpr int VS = 20;           // Vt row stride (bf16): 10 dw -> conflict-free b64
constexpr int PS = 36;           // P row stride (bf16): 18 dw -> conflict-free b64
constexpr float L2E = 1.4426950408889634f;

using v4short = short __attribute__((ext_vector_type(4)));
using v16short = short __attribute__((ext_vector_type(16)));
using v8float = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v16short load16(const unsigned short* p) {
  const v4short* q = reinterpret_cast<const v4short*>(p); // 4 x ds_read_b64
  v16short r;
  r.lo.lo = q[0]; r.lo.hi = q[1]; r.hi.lo = q[2]; r.hi.hi = q[3];
  return r;
}
__device__ __forceinline__ void store16(unsigned short* p, v16short v) {
  v4short* q = reinterpret_cast<v4short*>(p);
  q[0] = v.lo.lo; q[1] = v.lo.hi; q[2] = v.hi.lo; q[3] = v.hi.hi;
}
__device__ __forceinline__ v8float wmma(v16short a, v16short b, v8float c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
__device__ __forceinline__ float bf2f(unsigned short h) { return __uint_as_float((unsigned)h << 16); }
__device__ __forceinline__ unsigned short f2bf(float f) { // round-to-nearest-even (finite inputs)
  unsigned u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}

// Trellis kk=5 symbol for value e (0..255) from a packed row (40 words).
// State resets every 64 values (encoder runs 4 independent 64-step chains).
__device__ __forceinline__ unsigned sym5(const unsigned* rw, int e) {
  const unsigned* cw = rw + (e >> 6) * 10;
  const int em1 = 5 * (e & 63) + 4, j = em1 >> 5, r = 31 - (em1 & 31);
  return __funnelshift_r(cw[j], j ? cw[j - 1] : 0u, (unsigned)r) & 255u;
}

// Decode 16 consecutive values [16*c16 .. 16*c16+15] of one packed row.
__device__ __forceinline__ v16short dec16(const unsigned* rw, int c16, const unsigned short* tab) {
  const int q = c16 & 3, c64 = c16 >> 2, ec0 = 16 * q;
  const int jfirst = (5 * ec0 + 4) >> 5, wbase = jfirst - 1;
  unsigned w[5];
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    const int wi = wbase + j;
    w[j] = (wi >= 0 && wi < 10) ? rw[10 * c64 + wi] : 0u;
  }
  v16short val;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int em1 = 5 * (ec0 + i) + 4, j = em1 >> 5, r = 31 - (em1 & 31);
    const unsigned s = __funnelshift_r(w[j - wbase], w[j - 1 - wbase], (unsigned)r) & 255u;
    val[i] = tab[s];
  }
  return val;
}

struct NatPkArgs {
  const uint16_t* q16; const uint8_t* kp; const uint8_t* vp; const float* kps; const float* vps;
  const int8_t* v8; const float* vs; // V int8 container (vpk == 0)
  const unsigned char* rec; const uint8_t* wk; const int* d_pos;
  float* pm; float* pl; float* pacc;
  int M, max_ctx, splits, splits_short, long_len, layer, vpk;
  int dbg_x, dbg_y, dbg_z; // which block dumps (env-selected)
  float* dbg; // optional intermediate dump, null when disabled
};

// Debug dump layout (f32 words; env-selected block writes). Self-contained:
// packed inputs + LUTs + Q + geometry let a CPU reference recompute the whole
// chain (decode -> QK -> softmax -> PV -> self -> mu) independently of the GPU.
//   kt[16*256] decoded K tile0 [key][hd]; vt[16*256] V tile0 [key][hd];
//   sc[16*16] scaled+masked scores tile0 [pair][key]; ml[32] m/l after tile0;
//   pm[16], pl[16], pacc[16*256] final outputs of this WG;
//   pk[2*16*40] packed K/V rows tile0 (u32 bit patterns); tab[512] bf16->f32;
//   q[16*256] bf16->f32; geo[102] base,k0,klo,khi,ntiles,lo[16],ph[16],vld[16],
//   slf[16],ksc[16],vsc[16],vpk; mu[256] muV; self chain + per-tile regions.
constexpr int DBG_KT = 0, DBG_VT = 16 * 256, DBG_SC = 2 * 16 * 256, DBG_ML = DBG_SC + 16 * 16;
constexpr int DBG_PM = DBG_ML + 32, DBG_PL = DBG_PM + 16, DBG_PA = DBG_PL + 16;
constexpr int DBG_PK = DBG_PA + 16 * 256, DBG_TAB = DBG_PK + 2 * 16 * 40, DBG_Q = DBG_TAB + 512;
constexpr int DBG_GEO = DBG_Q + 16 * 256, DBG_MU = DBG_GEO + 102;
// self-epilogue chain: sc/cfac/ps[16] + ksc_own/vsc_own[2], decoded self rows,
// packed self rows (tok0 K/V, tok1 K/V, 40 words each).
constexpr int DBG_SELF = DBG_MU + 256, DBG_KOWN = DBG_SELF + 52, DBG_VOWN = DBG_KOWN + 512;
constexpr int DBG_PSELF = DBG_VOWN + 512;
// per-tile regions (up to 4 tiles): pk[1280] sc[256] ml[32] cf[16] ksc[16] vsc[16]
constexpr int DBG_T0 = DBG_PSELF + 160, DBG_TS = 1616, DBG_MAXT = 4;
// vpk=0 only: full V int8 rows (64 words/key): 4 tiles x 16 keys + 2 self rows
constexpr int DBG_V8 = DBG_T0 + DBG_TS * DBG_MAXT;
constexpr int DBG_WORDS = DBG_V8 + 4 * 1024 + 128;

__global__ void __launch_bounds__(256) q35_atxpk_native_kernel(NatPkArgs U) {
  __shared__ __align__(16) unsigned short Q[16 * QS], Kt[KT * QS], Vt[HD * VS], P[16 * PS];
  __shared__ __align__(16) unsigned short tabk[256], tabv[256], kown[2 * 256], vown[2 * 256];
  __shared__ float ksc[KT], vsc[KT], sm[16], sl[16], cf[16];
  __shared__ float s_scf[16], s_spv[16]; // self-epilogue factors, shared across waves
  __shared__ int s_lo[16], s_ph[16], s_vld[16], s_slf[16], s_klo, s_khi;

  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int t0 = blockIdx.x * 2, split = blockIdx.y, kvh = blockIdx.z;
  const int base = *U.d_pos;
  bool dbg = false; // resolved after row geometry (DUMP_SELF needs s_slf)

  // ---- per-pair row geometry (common spec math) ----
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
    const int ph = max(lo, min(hi, len - 1));
    s_lo[i] = lo; s_ph[i] = ph; s_vld[i] = v;
    s_slf[i] = v && lo <= len - 1 && len - 1 < lo + per;
    sm[i] = -1e30f; sl[i] = 0.f;
    s_scf[i] = 1.f; s_spv[i] = 0.f;
  }
  if (!__syncthreads_or(act)) return;
  if (U.dbg) {
    if (U.dbg_x < 0) dbg = blockIdx.x == 0 && blockIdx.z == 0 && s_slf[0] != 0; // DUMP_SELF: token0/kvh0 self WG
    else dbg = (int)blockIdx.x == U.dbg_x && (int)blockIdx.y == U.dbg_y && (int)blockIdx.z == U.dbg_z;
  }
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

  // ---- codebooks (rec group = 2048B: tab bf16[256] | mu f32[256] | ...) ----
  const unsigned char* rk = U.rec + (size_t)((U.layer * KVH + kvh) * 2) * 2048;
  const unsigned char* rv = rk + 2048;
  if (tid < 32) reinterpret_cast<uint4*>(tabk)[tid] = reinterpret_cast<const uint4*>(rk)[tid];
  else if (tid < 64) reinterpret_cast<uint4*>(tabv)[tid - 32] = reinterpret_cast<const uint4*>(rv)[tid - 32];

  // ---- Q tile: 16 pair rows of 256 bf16 (invalid rows zero), 16B chunks ----
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

  const uint8_t* kpg = U.kp + (size_t)kvh * U.max_ctx * PB;
  const uint8_t* vpg = U.vpk ? U.vp + (size_t)kvh * U.max_ctx * PB : nullptr;
  const int8_t* vg8 = U.vpk ? nullptr : U.v8 + (size_t)kvh * U.max_ctx * HD;
  const float* ksg = U.kps + (size_t)kvh * U.max_ctx;
  const float* vsg = (U.vpk ? U.vps : U.vs) + (size_t)kvh * U.max_ctx;

  v8float o0 = {}, o1 = {}; // PV accumulators: dim tiles 2*wave, 2*wave+1

  for (int t = 0; t < ntiles; ++t) {
    const int k0 = tbeg + t * KT;
    // ---- K decode (all waves): unit = key*16 + c16, 16 values per unit ----
    {
      const int key = tid >> 4, c16 = tid & 15, kidx = k0 + key;
      unsigned short* krow = Kt + key * QS + 16 * c16;
      if (kidx >= klo && kidx < khi) {
        const unsigned* rw = reinterpret_cast<const unsigned*>(kpg + (size_t)kidx * PB);
        store16(krow, dec16(rw, c16, tabk));
      } else {
        store16(krow, v16short{});
      }
      if (tid < KT) {
        const int kidx2 = k0 + tid;
        const bool in = kidx2 >= klo && kidx2 < khi;
        ksc[tid] = in ? ksg[kidx2] : 0.f;
        vsc[tid] = in ? vsg[kidx2] : 0.f;
      }
    }
    __syncthreads(); // B1: Kt + scales ready

    // ---- QK (wave 0) || V decode (waves 1-7) ----
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
          if (U.vpk) {
            const unsigned* rw = reinterpret_cast<const unsigned*>(vpg + (size_t)kidx * PB);
            const v16short val = dec16(rw, c16, tabv);
#pragma unroll
            for (int i = 0; i < 16; ++i) Vt[(16 * c16 + i) * VS + key] = val[i];
          } else {
            const int8_t* vr = vg8 + (size_t)kidx * HD + 16 * c16;
#pragma unroll
            for (int i = 0; i < 16; ++i) Vt[(16 * c16 + i) * VS + key] = f2bf((float)vr[i]); // int8 -> bf16 exact
          }
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
        if (dbg && t == 0) U.dbg[DBG_SC + p * 16 + key] = s;
        if (dbg && t < DBG_MAXT) U.dbg[DBG_T0 + t * DBG_TS + 1280 + p * 16 + key] = s;
        float mx = s;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float m_old = sm[p], mn = fmaxf(m_old, mx);
        const float cfac = exp2f((m_old - mn) * L2E);
        const float pv = exp2f(s * L2E - mn * L2E);
        float sum = pv;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane % 16 == 0) {
          sm[p] = mn; sl[p] = sl[p] * cfac + sum; cf[p] = cfac;
          if (dbg && t < DBG_MAXT) {
            float* T = U.dbg + DBG_T0 + t * DBG_TS;
            T[1536 + p] = mn; T[1536 + 16 + p] = sl[p]; T[1568 + p] = cfac;
          }
        }
        P[p * PS + key] = f2bf(pv * vsc[key]);
      }
      if (dbg && t == 0 && lane % 16 == 0) {
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          const int p = 2 * g + lane / 16;
          U.dbg[DBG_ML + p] = sm[p]; U.dbg[DBG_ML + 16 + p] = sl[p];
        }
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
    if (dbg && t == 0) {
      for (int i = tid; i < KT * 256; i += 256) {
        U.dbg[DBG_KT + i] = bf2f(Kt[(i >> 8) * QS + (i & 255)]);
        U.dbg[DBG_VT + i] = bf2f(Vt[(i & 255) * VS + (i >> 8)]);
        U.dbg[DBG_Q + i] = bf2f(Q[(i >> 8) * QS + (i & 255)]);
      }
      for (int i = tid; i < 2 * KT * 40; i += 256) {
        const int kv = i >= KT * 40, j = kv ? i - KT * 40 : i;
        const int key = j / 40, w = j % 40, kidx = k0 + key;
        unsigned word = 0u;
        if (kidx >= klo && kidx < khi) {
          if (kv && !U.vpk) word = reinterpret_cast<const unsigned*>(vg8 + (size_t)kidx * HD)[w];
          else word = reinterpret_cast<const unsigned*>((kv ? vpg : kpg) + (size_t)kidx * PB)[w];
        }
        U.dbg[DBG_PK + i] = __uint_as_float(word);
      }
      if (tid < 256) {
        U.dbg[DBG_TAB + tid] = bf2f(tabk[tid]);
        U.dbg[DBG_TAB + 256 + tid] = bf2f(tabv[tid]);
      }
      if (tid == 0) {
        U.dbg[DBG_GEO + 0] = (float)base; U.dbg[DBG_GEO + 1] = (float)k0;
        U.dbg[DBG_GEO + 2] = (float)klo; U.dbg[DBG_GEO + 3] = (float)khi;
        U.dbg[DBG_GEO + 4] = (float)ntiles; U.dbg[DBG_GEO + 101] = (float)U.vpk;
      }
      if (tid < 16) {
        U.dbg[DBG_GEO + 5 + tid] = (float)s_lo[tid]; U.dbg[DBG_GEO + 21 + tid] = (float)s_ph[tid];
        U.dbg[DBG_GEO + 37 + tid] = (float)s_vld[tid]; U.dbg[DBG_GEO + 53 + tid] = (float)s_slf[tid];
        U.dbg[DBG_GEO + 69 + tid] = ksc[tid]; U.dbg[DBG_GEO + 85 + tid] = vsc[tid];
      }
    }
    if (dbg && t < DBG_MAXT) {
      float* T = U.dbg + DBG_T0 + t * DBG_TS;
      for (int i = tid; i < 2 * KT * 40; i += 256) {
        const int kv = i >= KT * 40, j = kv ? i - KT * 40 : i;
        const int key = j / 40, w = j % 40, kidx = k0 + key;
        unsigned word = 0u;
        if (kidx >= klo && kidx < khi) {
          if (kv && !U.vpk) word = reinterpret_cast<const unsigned*>(vg8 + (size_t)kidx * HD)[w];
          else word = reinterpret_cast<const unsigned*>((kv ? vpg : kpg) + (size_t)kidx * PB)[w];
        }
        T[i] = __uint_as_float(word);
      }
      if (tid < 16) { T[1584 + tid] = ksc[tid]; T[1600 + tid] = vsc[tid]; }
      if (!U.vpk) {
        float* V8 = U.dbg + DBG_V8 + t * 1024;
        for (int i = tid; i < KT * 64; i += 256) {
          const int key = i >> 6, w = i & 63, kidx = k0 + key;
          const unsigned word = (kidx >= klo && kidx < khi)
            ? reinterpret_cast<const unsigned*>(vg8 + (size_t)kidx * HD)[w] : 0u;
          V8[i] = __uint_as_float(word);
        }
      }
    }
    __syncthreads(); // B4: tiles consumed; next iteration may overwrite
  }

  // ---- self-position epilogue: decode this WG's (at most 2) self rows ----
  for (int u = tid; u < 2 * 2 * 256; u += 256) {
    const int tok = u >> 9, kv = (u >> 8) & 1, e = u & 255;
    unsigned short v = 0;
    if (t0 + tok < U.M) { // never read past the written rows (max_ctx boundary)
      const size_t row = (size_t)(base + t0 + tok);
      if (kv && !U.vpk) v = f2bf((float)vg8[row * HD + e]);
      else {
        const unsigned* rw = reinterpret_cast<const unsigned*>((kv ? vpg : kpg) + row * PB);
        v = (kv ? tabv : tabk)[sym5(rw, e)];
      }
    }
    (kv ? vown : kown)[tok * 256 + e] = v;
  }
  __syncthreads();
  if (dbg) {
    for (int i = tid; i < 512; i += 256) {
      U.dbg[DBG_KOWN + i] = bf2f(kown[i]);
      U.dbg[DBG_VOWN + i] = bf2f(vown[i]);
    }
    for (int i = tid; i < 160; i += 256) {
      const int tok = i / 80, kv = (i / 40) & 1, w = i % 40;
      unsigned word = 0u;
      if (t0 + tok < U.M) {
        const size_t row = (size_t)(base + t0 + tok);
        if (kv && !U.vpk) word = reinterpret_cast<const unsigned*>(vg8 + row * HD)[w];
        else word = reinterpret_cast<const unsigned*>((kv ? vpg : kpg) + row * PB)[w];
      }
      U.dbg[DBG_PSELF + i] = __uint_as_float(word);
    }
    if (tid < 2) {
      const bool tv = t0 + tid < U.M;
      U.dbg[DBG_SELF + 48 + tid] = tv ? ksg[base + t0 + tid] : 0.f;
      U.dbg[DBG_SELF + 50 + tid] = tv ? vsg[base + t0 + tid] : 0.f;
    }
    if (!U.vpk) {
      for (int i = tid; i < 128; i += 256) {
        const int tok = i >> 6, w = i & 63;
        unsigned word = 0u;
        if (t0 + tok < U.M) word = reinterpret_cast<const unsigned*>(vg8 + (size_t)(base + t0 + tok) * HD)[w];
        U.dbg[DBG_V8 + 4 * 1024 + i] = __uint_as_float(word);
      }
    }
  }
  {
    const int p = tid >> 4, tt = tid & 15; // 16 threads per pair; wave = p/2
    if (s_slf[p]) {
      const int tok = p / G;
      const size_t own = (size_t)(base + t0 + tok);
      float part = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i)
        part = __fadd_rn(part, __fmul_rn(bf2f(Q[p * QS + 16 * tt + i]), bf2f(kown[tok * 256 + 16 * tt + i])));
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, o));
      const float sc = part * (ksg[own] * 0.0625f);
      const float m_old = sm[p], mn = fmaxf(m_old, sc);
      const float cfac = exp2f((m_old - mn) * L2E), ps = exp2f((sc - mn) * L2E);
      if (tt == 0) {
        sm[p] = mn; sl[p] = sl[p] * cfac + ps;
        s_scf[p] = cfac; s_spv[p] = ps * vsg[own];
        if (dbg) { U.dbg[DBG_SELF + p] = sc; U.dbg[DBG_SELF + 16 + p] = cfac; U.dbg[DBG_SELF + 32 + p] = ps; }
      }
    }
  }
  __syncthreads(); // self factors + m/l final
  // Every wave holds ALL 16 pairs for its own 32 dims: apply each pair's self
  // rescale + contribution in every wave (the dot was computed by wave p/2 only).
  {
    const int d0 = 16 * (2 * wave) + lane % 16, d1 = 16 * (2 * wave + 1) + lane % 16;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const int p = 2 * g + lane / 16;
      const int tok = p < 2 * G ? p / G : 0;
      const float c = s_scf[p], pv = s_spv[p];
      const float v0 = bf2f(vown[tok * 256 + d0]), v1 = bf2f(vown[tok * 256 + d1]);
      o0[g] = o0[g] * c + pv * v0;
      o1[g] = o1[g] * c + pv * v1;
    }
  }

  // ---- muV epilogue (packed V only) + outputs ----
  const float* muv = reinterpret_cast<const float*>(rv + 512);
  if (dbg && tid < 256) U.dbg[DBG_MU + tid] = muv[tid];
  const float mu0 = U.vpk ? muv[16 * (2 * wave) + lane % 16] : 0.f;
  const float mu1 = U.vpk ? muv[16 * (2 * wave + 1) + lane % 16] : 0.f;
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const int p = 2 * g + lane / 16;
    const float l = sl[p];
    o0[g] += l * mu0; o1[g] += l * mu1;
    if (s_vld[p]) {
      const int tk = t0 + p / G, hd = kvh * G + p % G;
      const size_t pidx = ((size_t)tk * U.splits + split) * NH + hd;
      U.pacc[pidx * HD + 16 * (2 * wave) + lane % 16] = o0[g];
      U.pacc[pidx * HD + 16 * (2 * wave + 1) + lane % 16] = o1[g];
      if (lane % 16 == 0) { U.pm[pidx] = sm[p]; U.pl[pidx] = sl[p]; }
      if (dbg) {
        U.dbg[DBG_PA + p * 256 + 16 * (2 * wave) + lane % 16] = o0[g];
        U.dbg[DBG_PA + p * 256 + 16 * (2 * wave + 1) + lane % 16] = o1[g];
        if (lane % 16 == 0) { U.dbg[DBG_PM + p] = sm[p]; U.dbg[DBG_PL + p] = sl[p]; }
      }
    }
  }
}

int nat_off_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_NATIVE"); return e ? std::atoi(e) == 0 : 0; }(); return v; }
int nat_qk8_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_PK3_QK8"); return e ? std::atoi(e) != 0 : 0; }(); return v; }
const char* nat_dump_env() { static const char* v = std::getenv("YSKN_ATX_NATIVE_DUMP"); return v; }
int nat_dump_layer_env() { static const int v = [] { const char* e = std::getenv("YSKN_ATX_NATIVE_DUMP_LAYER"); return e ? std::atoi(e) : 0; }(); return v; }
struct DbgBlock { int x, y, z; };
const DbgBlock& nat_dump_block() {
  static const DbgBlock b = [] {
    DbgBlock t{0, 0, 0};
    const char* e = std::getenv("YSKN_ATX_NATIVE_DUMP_BLOCK");
    if (e) std::sscanf(e, "%d,%d,%d", &t.x, &t.y, &t.z);
    const char* s = std::getenv("YSKN_ATX_NATIVE_DUMP_SELF");
    if (s && std::atoi(s) != 0) t.x = -1; // self-WG mode
    return t;
  }();
  return b;
}

} // namespace

// Native entry point. Unsupported configurations fall back to the existing
// (CUDA-reference) launcher so the engine interface is always satisfied.
// Covered: kb=5 with K packed, V packed (trellis) or V int8 (vpk=0, layers
// whose V group has projection vectors and is not pk-packable).
void launch_q35_atx_pk_native(const Q35AttnArgs& A, const uint8_t* wk, const Q35PkArgs& P, cudaStream_t st) {
  const bool vpk = P.vp != nullptr;
  const bool vok = vpk ? (P.vps != nullptr) : (A.v8 != nullptr && A.vs != nullptr);
  if (!nat_off_env() && !nat_qk8_env() && P.kb == 5 && P.kp && P.kps && P.rec && vok && !A.kv_fp16 && A.M > 0 && A.splits > 0) {
    const DbgBlock& db = nat_dump_block();
    NatPkArgs U{A.q16, P.kp, P.vp, P.kps, P.vps, A.v8, A.vs, P.rec, wk, A.d_pos, A.pm, A.pl, A.pacc,
                A.M, A.max_ctx, A.splits, A.splits_short, A.long_len, P.layer, vpk ? 1 : 0, db.x, db.y, db.z, nullptr};
    // Debug dump: the buffer is process-lifetime and allocated OUTSIDE graph
    // capture (hipMalloc/hipMemset/hipStreamSynchronize are illegal while
    // capturing). Captured launches reuse the buffer but never read it back.
    static float* g_dbg = nullptr;
    const char* dump = nat_dump_env();
    float* dbg = nullptr;
    bool captured = false;
    if (dump && dump[0] && P.layer == nat_dump_layer_env()) {
      hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
      (void)hipStreamIsCapturing(st, &cs);
      captured = cs != hipStreamCaptureStatusNone;
      if (!captured && !g_dbg && hipMalloc(&g_dbg, DBG_WORDS * 4) == hipSuccess)
        (void)hipMemset(g_dbg, 0, DBG_WORDS * 4);
      dbg = g_dbg; // null when the first call is captured: dump stays disabled
      U.dbg = dbg;
    }
    const dim3 grid((A.M + 1) / 2, A.splits, KVH);
    hipLaunchKernelGGL(q35_atxpk_native_kernel, grid, dim3(256), 0, st, U);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("q35_atxpk_native: launch failed: ") + hipGetErrorString(e));
    if (dbg && !captured) {
      std::vector<float> host(DBG_WORDS);
      if (hipStreamSynchronize(st) == hipSuccess &&
          hipMemcpy(host.data(), dbg, DBG_WORDS * 4, hipMemcpyDeviceToHost) == hipSuccess) {
        char pathbuf[1024];
        const char* path = (dump[0] == '1' && !dump[1]) ? "atxpk_native_dump.bin" : dump;
        static const bool seq = [] { const char* e = std::getenv("YSKN_ATX_NATIVE_DUMP_SEQ"); return e && std::atoi(e) != 0; }();
        static int counter = 0;
        if (seq) std::snprintf(pathbuf, sizeof(pathbuf), "%s.%04d", path, counter++);
        else std::snprintf(pathbuf, sizeof(pathbuf), "%s", path);
        FILE* f = std::fopen(pathbuf, "wb");
        if (f) {
          std::fwrite(host.data(), 4, DBG_WORDS, f);
          std::fclose(f);
          std::fprintf(stderr, "[atxpk_native] dump -> %s (layer=%d M=%d splits=%d block=%d,%d,%d)\n",
                       pathbuf, P.layer, A.M, A.splits, db.x, db.y, db.z);
        }
      }
    }
    return;
  }
  launch_q35_atx_pk(A, wk, P, st);
}

} // namespace yskn
