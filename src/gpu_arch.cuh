// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Architecture primitives: the only place with NVIDIA PTX inline asm. CUDA builds use the PTX;
// HIP builds use the AMD fallbacks. Shared-memory arguments are GENERIC pointers on both backends
// (the CUDA side converts with cvta internally; the HIP side dereferences directly).
// The HIP mma/ldmatrix paths below serve the CONVERTED CUDA kernels (NVIDIA fragment layouts
// rebuilt on RDNA3 WMMA / LDS reads). The native attention kernels in backends/hip use the
// WMMA layouts directly and do not go through these wrappers.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace yskn {
namespace arch {

// ---- async copy (cp.async) ----
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
#if defined(__HIP_PLATFORM_AMD__)
  *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<const uint4*>(gmem);   // synchronous staging (pipeline degrades, math unchanged)
#else
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem) : "memory");
#endif
}
__device__ __forceinline__ void cp8(void* smem, const void* gmem) {
#if defined(__HIP_PLATFORM_AMD__)
  *reinterpret_cast<uint2*>(smem) = *reinterpret_cast<const uint2*>(gmem);
#else
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem) : "memory");
#endif
}
__device__ __forceinline__ void cp4(void* smem, const void* gmem) {
#if defined(__HIP_PLATFORM_AMD__)
  *reinterpret_cast<unsigned*>(smem) = *reinterpret_cast<const unsigned*>(gmem);
#else
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem) : "memory");
#endif
}
__device__ __forceinline__ void cp_commit() {
#if !defined(__HIP_PLATFORM_AMD__)
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
template <int N> __device__ __forceinline__ void cp_wait() {
#if defined(__HIP_PLATFORM_AMD__)
  __syncthreads();   // conservative: copies are synchronous already; barrier keeps the staging order
#else
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}
__device__ __forceinline__ void cp_wait_all() {
#if defined(__HIP_PLATFORM_AMD__)
  __syncthreads();
#else
  asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

// ---- matrix multiply (mma.sync) ----
#if defined(__HIP_PLATFORM_AMD__)
// NVIDIA mma.m16n8k32 (M=16,N=8,K=32) on RDNA3 WMMA 16x16x16 (two independent 8x16x16 half-waves).
// Verified layout (native-probes/wmma_verify.hip, 12800/12800 bit-exact vs CPU):
//   A: row r -> lane (r<8 ? 2r : 2r+1), v4i byte (reg*4+byte) = k; other lanes ignored
//   B: col n, half h -> lane n+16h, byte = k;  D: col n, half h -> lane n+16h, reg = row-in-half
// NVIDIA fragments (lane l: g=l>>2, t=l&3): a0=A[g][4t..], a1=A[g+8][4t..], a2/a3 = k+16;
//   b0=B[4t..][g], b1=B[4t+16..][g]; d0=D[g][2t], d1=D[g][2t+1], d2=D[g+8][2t], d3=D[g+8][2t+1].
// K=32 -> two chained K=16 WMMAs; N=8 -> use cols 0-7 of the 16-wide WMMA (rest zero, discarded).
// Layout conversion is cross-lane shuffles (Phase 7: fold into the decode layout to remove them).
typedef int yskn_v4i __attribute__((ext_vector_type(4)));
typedef int yskn_v8i __attribute__((ext_vector_type(8)));
typedef float yskn_v8f __attribute__((ext_vector_type(8)));
typedef short yskn_v16s __attribute__((ext_vector_type(16)));
typedef _Float16 yskn_v16h __attribute__((ext_vector_type(16)));
template <bool A_SIGNED, bool ZERO_C>
__device__ __forceinline__ void mma_m16n8k32_amd(int* d, const unsigned* a, unsigned b0, unsigned b1) {
  const unsigned FULL = 0xffffffffu;
  const int L = threadIdx.x & 31;
  const int g = L >> 2, t = L & 3;
  const int n = L & 15, hw = L >> 4;
  const bool a_hw1 = (L >= 17) && (L & 1);
  const bool a_used = (L < 16 && !(L & 1)) || a_hw1;
  const int rA = a_hw1 ? (L - 17) / 2 : L / 2;
  yskn_v4i A0 = {0, 0, 0, 0}, A1 = {0, 0, 0, 0}, B0 = {0, 0, 0, 0}, B1 = {0, 0, 0, 0};
  yskn_v8i C = {0, 0, 0, 0, 0, 0, 0, 0};
  // All shuffles run convergently and exchange RAW registers; the half/register selection happens
  // AFTER the shuffle (a shuffle reads the value as computed on the SOURCE lane).
  {
    const int sb = 4 * (a_used ? rA : 0);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned x0 = __shfl_sync(FULL, a[0], sb + j), x1 = __shfl_sync(FULL, a[1], sb + j);
      const unsigned x2 = __shfl_sync(FULL, a[2], sb + j), x3 = __shfl_sync(FULL, a[3], sb + j);
      if (a_used) { A0[j] = a_hw1 ? x1 : x0; A1[j] = a_hw1 ? x3 : x2; }
    }
  }
  {
    const int ne = n & 7;
    const int sb = 4 * ne;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned x0 = __shfl_sync(FULL, b0, sb + j), x1 = __shfl_sync(FULL, b1, sb + j);
      if (n < 8) { B0[j] = x0; B1[j] = x1; }
    }
    if (!ZERO_C) {
      const int cn = ne >> 1;
      #pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        const int src = 4 * rr + cn;
        const unsigned c0 = __shfl_sync(FULL, d[0], src), c1 = __shfl_sync(FULL, d[1], src);
        const unsigned c2 = __shfl_sync(FULL, d[2], src), c3 = __shfl_sync(FULL, d[3], src);
        if (n < 8) C[rr] = (n & 1) ? (hw ? c3 : c1) : (hw ? c2 : c0);
      }
    }
  }
  yskn_v8i D0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(A_SIGNED, A0, true, B0, C, false);
  yskn_v8i D1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(A_SIGNED, A1, true, B1, D0, false);
  const int s0 = 2 * t, s1 = 2 * t + 1, s2 = 16 + 2 * t, s3 = 17 + 2 * t;
  #pragma unroll
  for (int rr = 0; rr < 8; ++rr) {
    const unsigned v0 = __shfl_sync(FULL, D1[rr], s0);
    const unsigned v1 = __shfl_sync(FULL, D1[rr], s1);
    const unsigned v2 = __shfl_sync(FULL, D1[rr], s2);
    const unsigned v3 = __shfl_sync(FULL, D1[rr], s3);
    if (rr == g) { d[0] = (int)v0; d[1] = (int)v1; d[2] = (int)v2; d[3] = (int)v3; }
  }
}
// NVIDIA mma.m16n8k16 (bf16/f16, M=16,N=8,K=16) on RDNA3 WMMA 16x16x16 f32-accum.
// Same half-wave layout as the i8 instruction (verified): A row r -> lane (r<8?2r:2r+1),
// v8i dword j = element pair (k=2j,2j+1); B col n half h -> lane n+16h; D lane n+16h reg rr = row rr+8h.
// NVIDIA fragments (g=l>>2, t=l&3): a0=A[g][2t..2t+1], a1=A[g+8][..], a2=A[g][2t+8..], a3=A[g+8][2t+8..];
// b0=B[2t..2t+1][g], b1=B[2t+8..][g]; d0..d3 same positions as the s8 case. K=16 -> one WMMA.
template <bool BF16>
__device__ __forceinline__ void mma_m16n8k16_amd(float* d, const unsigned* a, unsigned b0, unsigned b1) {
  const unsigned FULL = 0xffffffffu;
  const int L = threadIdx.x & 31;
  const int g = L >> 2, t = L & 3;
  const int n = L & 15, hw = L >> 4;
  const bool a_hw1 = (L >= 17) && (L & 1);
  const bool a_used = (L < 16 && !(L & 1)) || a_hw1;
  const int rA = a_hw1 ? (L - 17) / 2 : L / 2;
  yskn_v8i A = {0, 0, 0, 0, 0, 0, 0, 0}, B = {0, 0, 0, 0, 0, 0, 0, 0}, C = {0, 0, 0, 0, 0, 0, 0, 0};
  {
    const int sb = 4 * (a_used ? rA : 0);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned reg = (j < 4) ? a[0] : a[2];
      const unsigned reg1 = (j < 4) ? a[1] : a[3];
      const int src = sb + (j & 3);
      const unsigned x0 = __shfl_sync(FULL, reg, src), x1 = __shfl_sync(FULL, reg1, src);
      if (a_used) A[j] = a_hw1 ? x1 : x0;
    }
  }
  {
    const int ne = n & 7;
    const int sb = 4 * ne;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned reg = (j < 4) ? b0 : b1;
      const unsigned x0 = __shfl_sync(FULL, reg, sb + (j & 3));
      if (n < 8) B[j] = x0;
    }
    {
      const int cn = ne >> 1;
      #pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        const int src = 4 * rr + cn;
        const unsigned c0 = __shfl_sync(FULL, __float_as_uint(d[0]), src);
        const unsigned c1 = __shfl_sync(FULL, __float_as_uint(d[1]), src);
        const unsigned c2 = __shfl_sync(FULL, __float_as_uint(d[2]), src);
        const unsigned c3 = __shfl_sync(FULL, __float_as_uint(d[3]), src);
        if (n < 8) C[rr] = (n & 1) ? (hw ? c3 : c1) : (hw ? c2 : c0);
      }
    }
  }
  yskn_v8i D;
  if (BF16) D = __builtin_bit_cast(yskn_v8i, __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
      __builtin_bit_cast(yskn_v16s, A), __builtin_bit_cast(yskn_v16s, B), __builtin_bit_cast(yskn_v8f, C)));
  else      D = __builtin_bit_cast(yskn_v8i, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
      __builtin_bit_cast(yskn_v16h, A), __builtin_bit_cast(yskn_v16h, B), __builtin_bit_cast(yskn_v8f, C)));
  const int s0 = 2 * t, s1 = 2 * t + 1, s2 = 16 + 2 * t, s3 = 17 + 2 * t;
  #pragma unroll
  for (int rr = 0; rr < 8; ++rr) {
    const unsigned v0 = __shfl_sync(FULL, D[rr], s0);
    const unsigned v1 = __shfl_sync(FULL, D[rr], s1);
    const unsigned v2 = __shfl_sync(FULL, D[rr], s2);
    const unsigned v3 = __shfl_sync(FULL, D[rr], s3);
    if (rr == g) { d[0] = __uint_as_float(v0); d[1] = __uint_as_float(v1); d[2] = __uint_as_float(v2); d[3] = __uint_as_float(v3); }
  }
}

#endif

// d[4] = a[4] x b[2] + d[4], int8 m16n8k32 (the engine's exact-integer workhorse)
__device__ __forceinline__ void mma_s8(int* d, const unsigned* a, unsigned b0, unsigned b1) {
#if defined(__HIP_PLATFORM_AMD__)
  mma_m16n8k32_amd<true, false>(d, a, b0, b1);
#else
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
#if defined(__HIP_PLATFORM_AMD__)
// AMD-native accumulator path: C stays in WMMA v8i layout across calls (no per-call C-gather/D-scatter
// shuffles). Only A/B fragment gather (24 shfl) + 2 WMMA. ~3.8x faster than mma_s8 in lnord-like loops.
// Integer accumulation is exact, so results are bit-identical to the NVIDIA-layout path.
__device__ __forceinline__ void mma_s8_acc_amd(yskn_v8i& C, const unsigned* a, unsigned b0, unsigned b1) {
  const unsigned FULL = 0xffffffffu;
  const int L = threadIdx.x & 31;
  const int n = L & 15;
  const bool a_hw1 = (L >= 17) && (L & 1);
  const bool a_used = (L < 16 && !(L & 1)) || a_hw1;
  const int rA = a_hw1 ? (L - 17) / 2 : L / 2;
  yskn_v4i A0 = {0, 0, 0, 0}, A1 = {0, 0, 0, 0}, B0 = {0, 0, 0, 0}, B1 = {0, 0, 0, 0};
  const int sb = 4 * (a_used ? rA : 0);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned x0 = __shfl_sync(FULL, a[0], sb + j), x1 = __shfl_sync(FULL, a[1], sb + j);
    const unsigned x2 = __shfl_sync(FULL, a[2], sb + j), x3 = __shfl_sync(FULL, a[3], sb + j);
    if (a_used) { A0[j] = a_hw1 ? x1 : x0; A1[j] = a_hw1 ? x3 : x2; }
  }
  const int sb2 = 4 * (n & 7);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned x0 = __shfl_sync(FULL, b0, sb2 + j), x1 = __shfl_sync(FULL, b1, sb2 + j);
    if (n < 8) { B0[j] = x0; B1[j] = x1; }
  }
  yskn_v8i D0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, A0, true, B0, C, false);
  C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, A1, true, B1, D0, false);
}
// AMD-native mma with A already in WMMA layout (A0/A1 pre-staged, e.g. via LDS transpose).
// Only B fragment gather (8 shfl) + 2 WMMA; no A gather. Used by the LDS-staging lnord path.
__device__ __forceinline__ void mma_s8_acc_amd_prestaged(yskn_v8i& C, const yskn_v4i& A0, const yskn_v4i& A1, unsigned b0, unsigned b1) {
  const unsigned FULL = 0xffffffffu;
  const int n = threadIdx.x & 15;
  yskn_v4i B0 = {0, 0, 0, 0}, B1 = {0, 0, 0, 0};
  const int sb2 = 4 * (n & 7);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned x0 = __shfl_sync(FULL, b0, sb2 + j), x1 = __shfl_sync(FULL, b1, sb2 + j);
    if (n < 8) { B0[j] = x0; B1[j] = x1; }
  }
  yskn_v8i D0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, A0, true, B0, C, false);
  C = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, A1, true, B1, D0, false);
}
// Convert AMD WMMA v8i accumulator -> NVIDIA m16n8k32 d[4] fragment (one-time, e.g. before reduction).
__device__ __forceinline__ void amd_to_nvidia_s8(const yskn_v8i& C, int* d) {
  const unsigned FULL = 0xffffffffu;
  const int L = threadIdx.x & 31;
  const int g = L >> 2, t = L & 3;
  const int s0 = 2 * t, s1 = 2 * t + 1, s2 = 16 + 2 * t, s3 = 17 + 2 * t;
  #pragma unroll
  for (int rr = 0; rr < 8; ++rr) {
    const unsigned v0 = __shfl_sync(FULL, C[rr], s0), v1 = __shfl_sync(FULL, C[rr], s1);
    const unsigned v2 = __shfl_sync(FULL, C[rr], s2), v3 = __shfl_sync(FULL, C[rr], s3);
    if (rr == g) { d[0] = (int)v0; d[1] = (int)v1; d[2] = (int)v2; d[3] = (int)v3; }
  }
}
#endif  // __HIP_PLATFORM_AMD__ (AMD-only accumulator helpers; CUDA uses the d[4] fragment below)
// Accumulator abstraction for repeated mma_s8 accumulation. On AMD the accumulator stays in native
// WMMA layout (avoids per-mma layout-conversion shuffles); on CUDA it is the NVIDIA d[4] fragment.
#if defined(__HIP_PLATFORM_AMD__)
typedef yskn_v8i mma_s8_acc;
__device__ __forceinline__ void mma_s8_acc_zero(mma_s8_acc& c) { c = (yskn_v8i){0, 0, 0, 0, 0, 0, 0, 0}; }
__device__ __forceinline__ void mma_s8_acc_mma(mma_s8_acc& c, const unsigned* a, unsigned b0, unsigned b1) { mma_s8_acc_amd(c, a, b0, b1); }
__device__ __forceinline__ void mma_s8_acc_load(const mma_s8_acc& c, int* d) { amd_to_nvidia_s8(c, d); }
#else
struct mma_s8_acc { int d[4]; };
__device__ __forceinline__ void mma_s8_acc_zero(mma_s8_acc& c) { c.d[0] = c.d[1] = c.d[2] = c.d[3] = 0; }
__device__ __forceinline__ void mma_s8_acc_mma(mma_s8_acc& c, const unsigned* a, unsigned b0, unsigned b1) { mma_s8(c.d, a, b0, b1); }
__device__ __forceinline__ void mma_s8_acc_load(const mma_s8_acc& c, int* d) { d[0] = c.d[0]; d[1] = c.d[1]; d[2] = c.d[2]; d[3] = c.d[3]; }
#endif
// d[4] = a[4] x b[2] (C = 0)
__device__ __forceinline__ void mma_s8_0(int* d, const unsigned* a, unsigned b0, unsigned b1) {
#if defined(__HIP_PLATFORM_AMD__)
  mma_m16n8k32_amd<true, true>(d, a, b0, b1);
#else
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
#endif
}
// unsigned8 x signed8 variant (q35_kz)
__device__ __forceinline__ void mma_u8s8(int* d, const unsigned* a, unsigned b0, unsigned b1) {
#if defined(__HIP_PLATFORM_AMD__)
  mma_m16n8k32_amd<false, false>(d, a, b0, b1);
#else
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void mma_bf16(float* d, const unsigned* a, unsigned b0, unsigned b1) {
#if defined(__HIP_PLATFORM_AMD__)
  mma_m16n8k16_amd<true>(d, a, b0, b1);
#else
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void mma_f16(float* d, const unsigned* a, unsigned b0, unsigned b1) {
#if defined(__HIP_PLATFORM_AMD__)
  mma_m16n8k16_amd<false>(d, a, b0, b1);
#else
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// ---- ldmatrix ----
// ldmatrix m8n8.x4: lane l supplies the address of row (l%8) of matrix (l/8); result r[i] of lane l
// = matrix i [l/4][2*(l%4) .. +1] (the mma b16 fragment layout). HIP: shuffle the four row-0 base
// addresses, then read the needed dwords straight from LDS (no cross-lane data movement).
__device__ __forceinline__ void ldm4(unsigned* r, const void* smem) {
#if defined(__HIP_PLATFORM_AMD__)
  const int l = threadIdx.x & 31;
  const unsigned long long addr = (unsigned long long)(uintptr_t)smem;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned long long base = __shfl_sync(0xffffffffull, addr, 8 * i);
    r[i] = *reinterpret_cast<const unsigned*>((uintptr_t)base + (l / 4) * 16 + (l % 4) * 4);
  }
#else
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(smem)));
#endif
}
// .trans variant: r[i] = matrix i [2*(l%4)][l/4] | matrix i [2*(l%4)+1][l/4] << 16.
__device__ __forceinline__ void ldm4t(unsigned* r, const void* smem) {
#if defined(__HIP_PLATFORM_AMD__)
  const int l = threadIdx.x & 31;
  const unsigned long long addr = (unsigned long long)(uintptr_t)smem;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned long long base = __shfl_sync(0xffffffffull, addr, 8 * i);
    const unsigned lo = *reinterpret_cast<const unsigned short*>((uintptr_t)base + (2 * (l % 4)) * 16 + (l / 4) * 2);
    const unsigned hi = *reinterpret_cast<const unsigned short*>((uintptr_t)base + (2 * (l % 4) + 1) * 16 + (l / 4) * 2);
    r[i] = lo | (hi << 16);
  }
#else
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(smem)));
#endif
}

// ---- shared / global loads ----
__device__ __forceinline__ uint4 ld_shared_v4(const void* smem) {
#if defined(__HIP_PLATFORM_AMD__)
  return *reinterpret_cast<const uint4*>(smem);
#else
  uint4 r;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "r"((unsigned)__cvta_generic_to_shared(smem)));
  return r;
#endif
}
__device__ __forceinline__ unsigned ld_nc_u32(const void* p) {
#if defined(__HIP_PLATFORM_AMD__)
  return __ldg(reinterpret_cast<const unsigned*>(p));
#else
  unsigned x;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(x) : "l"(p));
  return x;
#endif
}
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
#if defined(__HIP_PLATFORM_AMD__)
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#else
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
#endif
}

// ---- barriers / hints / timer ----
__device__ __forceinline__ void bar_sync(int id, int nthreads) {
#if defined(__HIP_PLATFORM_AMD__)
  (void)id; (void)nthreads; __syncthreads();   // conservative full-block barrier (sub-block barriers: Phase 3)
#else
  asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(nthreads) : "memory");
#endif
}
__device__ __forceinline__ void prefetch_l1(const void* p) {
#if !defined(__HIP_PLATFORM_AMD__)
  asm volatile("prefetch.global.L1 [%0];" :: "l"(p));
#else
  (void)p;
#endif
}
__device__ __forceinline__ void prefetch_l2(const void* p) {
#if !defined(__HIP_PLATFORM_AMD__)
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#else
  (void)p;
#endif
}
__device__ __forceinline__ unsigned long long globaltimer() {
#if defined(__HIP_PLATFORM_AMD__)
  return 0;   // trace-only path (YSKN_Q35_FLOW_TRACE, off by default); a real cycle counter comes in Phase 7
#else
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
#endif
}

}  // namespace arch
}  // namespace yskn
