// SPDX-License-Identifier: Apache-2.0
// Minimal native WMMA spike for gfx1100 (RDNA3, wave32) using compiler builtins.
// Layout source of truth: AMD ROCm Optimization Guide, "RDNA3 dense WMMA builtins"
//   A: lane L holds row L%16 (16 bf16, v16short), replicated on lanes 0-15 / 16-31.
//   B: lane L holds column L%16 of the K x N tile (16 bf16), replicated.
//   C: lane L, VGPR g -> row 2g + L/16, col L%16 (v8float).
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1100 spike_wmma.hip -o spike_wmma
// Run:   ./spike_wmma   (prints PASS/FAIL per test; needs /dev/kfd)
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

using v4short = short __attribute__((ext_vector_type(4)));
using v16short = short __attribute__((ext_vector_type(16)));
using v8float = float __attribute__((ext_vector_type(8)));
using v4int = int __attribute__((ext_vector_type(4)));
using v8int = int __attribute__((ext_vector_type(8)));

#define HIPCHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// 16 bf16 = 4 x ds_read_b64 (8B-aligned; works with padded LDS strides).
__device__ __forceinline__ v16short load16(const unsigned short* p) {
  const v4short* q = reinterpret_cast<const v4short*>(p);
  v16short r;
  r.lo.lo = q[0]; r.lo.hi = q[1]; r.hi.lo = q[2]; r.hi.hi = q[3];
  return r;
}

// C[16x16] = A[16x16] * B[16x16]; A row-major, BT row-major = B col-major (BT[n][k]).
__global__ void k_bf16_global(const unsigned short* A, const unsigned short* BT, float* C) {
  const int l = threadIdx.x & 31;
  v16short a = load16(A + (l % 16) * 16);
  v16short b = load16(BT + (l % 16) * 16);
  v8float c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, v8float{});
#pragma unroll
  for (int g = 0; g < 8; ++g) C[(2 * g + l / 16) * 16 + (l % 16)] = c[g];
}

// K=32 (two steps), operands staged in LDS with padded stride 20 (40B rows).
__global__ void k_bf16_lds32(const unsigned short* A, const unsigned short* BT, float* C) {
  __shared__ __align__(16) unsigned short sa[16 * 20], sb[16 * 20];
  const int l = threadIdx.x & 31;
  v8float c = {};
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
      sa[(i / 16) * 20 + (i % 16)] = A[(i / 16) * 32 + 16 * ks + (i % 16)];
      sb[(i / 16) * 20 + (i % 16)] = BT[(i / 16) * 32 + 16 * ks + (i % 16)];
    }
    __syncthreads();
    v16short a = load16(sa + (l % 16) * 20);
    v16short b = load16(sb + (l % 16) * 20);
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    __syncthreads();
  }
#pragma unroll
  for (int g = 0; g < 8; ++g) C[(2 * g + l / 16) * 16 + (l % 16)] = c[g];
}

// int8 -> int32, exact. A row-major [16][16] int8, BT[n][k] int8.
__global__ void k_i8_global(const signed char* A, const signed char* BT, int* C) {
  const int l = threadIdx.x & 31;
  v4int a = *reinterpret_cast<const v4int*>(A + (l % 16) * 16);
  v4int b = *reinterpret_cast<const v4int*>(BT + (l % 16) * 16);
  v8int c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, v8int{}, false);
#pragma unroll
  for (int g = 0; g < 8; ++g) C[(2 * g + l / 16) * 16 + (l % 16)] = c[g];
}

static float bf2f(unsigned short h) {
  unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static unsigned short f2bf(float f) {
  unsigned u; memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}

int main() {
  hipDeviceProp_t prop;
  HIPCHK(hipGetDeviceProperties(&prop, 0));
  printf("device: %s (arch %s, wavefront %d, LDS %zu B)\n",
         prop.name, prop.gcnArchName, prop.warpSize, (size_t)prop.sharedMemPerBlock);

  const int N = 16;
  std::vector<unsigned short> hA(N * N), hB(N * N), hA2(N * N), hB2(N * N);
  std::vector<signed char> hA8(N * N), hB8(N * N);
  std::vector<float> ref(N * N), ref32(N * N);
  std::vector<int> ref8(N * N);
  for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
    hA[i * N + j] = f2bf(((i * 3 + j * 5) % 17 - 8) * 0.25f);
    hB[i * N + j] = f2bf(((i * 7 + j * 11) % 13 - 6) * 0.5f);
    hA2[i * N + j] = f2bf(((i * 13 + j) % 19 - 9) * 0.125f);
    hB2[i * N + j] = f2bf(((i * 17 + j) % 23 - 11) * 0.25f);
    hA8[i * N + j] = (signed char)((i * 7 + j * 11) % 255 - 127);
    hB8[i * N + j] = (signed char)((i * 5 + j * 3) % 255 - 127);
  }
  for (int m = 0; m < N; ++m) for (int n = 0; n < N; ++n) {
    float s = 0.f, s32 = 0.f; int s8 = 0;
    for (int k = 0; k < N; ++k) {
      s += bf2f(hA[m * N + k]) * bf2f(hB[n * N + k]);
      s32 += bf2f(hA2[m * N + k]) * bf2f(hB2[n * N + k]);
      s8 += (int)hA8[m * N + k] * (int)hB8[n * N + k];
    }
    ref[m * N + n] = s; ref32[m * N + n] = s + s32; ref8[m * N + n] = s8;
  }

  void *dA, *dB, *dC;
  HIPCHK(hipMalloc(&dA, N * 32 * 2)); HIPCHK(hipMalloc(&dB, N * 32 * 2)); HIPCHK(hipMalloc(&dC, N * N * 4));
  std::vector<float> out(N * N);
  int fails = 0;
  auto check_f = [&](const char* name, const std::vector<float>& r, float tol) {
    HIPCHK(hipMemcpy(out.data(), dC, N * N * 4, hipMemcpyDeviceToHost));
    float mx = 0.f;
    for (int i = 0; i < N * N; ++i) mx = fmaxf(mx, fabsf(out[i] - r[i]));
    printf("%-22s max_err=%.6g %s\n", name, mx, mx <= tol ? "PASS" : "FAIL");
    if (mx > tol) ++fails;
  };

  HIPCHK(hipMemcpy(dA, hA.data(), N * N * 2, hipMemcpyHostToDevice));
  HIPCHK(hipMemcpy(dB, hB.data(), N * N * 2, hipMemcpyHostToDevice));
  k_bf16_global<<<1, 32>>>((const unsigned short*)dA, (const unsigned short*)dB, (float*)dC);
  HIPCHK(hipGetLastError()); HIPCHK(hipDeviceSynchronize());
  check_f("bf16 global", ref, 1e-3f);

  {
    std::vector<unsigned short> AA(N * 32), BB(N * 32);
    for (int m = 0; m < N; ++m) for (int k = 0; k < 32; ++k) {
      AA[m * 32 + k] = k < 16 ? hA[m * N + k] : hA2[m * N + (k - 16)];
      BB[m * 32 + k] = k < 16 ? hB[m * N + k] : hB2[m * N + (k - 16)];
    }
    HIPCHK(hipMemcpy(dA, AA.data(), N * 32 * 2, hipMemcpyHostToDevice));
    HIPCHK(hipMemcpy(dB, BB.data(), N * 32 * 2, hipMemcpyHostToDevice));
    k_bf16_lds32<<<1, 128>>>((const unsigned short*)dA, (const unsigned short*)dB, (float*)dC);
    HIPCHK(hipGetLastError()); HIPCHK(hipDeviceSynchronize());
    check_f("bf16 LDS pad K=32", ref32, 1e-3f);
  }

  HIPCHK(hipMemcpy(dA, hA8.data(), N * N, hipMemcpyHostToDevice));
  HIPCHK(hipMemcpy(dB, hB8.data(), N * N, hipMemcpyHostToDevice));
  k_i8_global<<<1, 32>>>((const signed char*)dA, (const signed char*)dB, (int*)dC);
  HIPCHK(hipGetLastError()); HIPCHK(hipDeviceSynchronize());
  {
    std::vector<int> o8(N * N);
    HIPCHK(hipMemcpy(o8.data(), dC, N * N * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N * N; ++i) if (o8[i] != ref8[i]) ++bad;
    printf("%-22s mismatches=%d %s\n", "int8 global exact", bad, bad ? "FAIL" : "PASS");
    if (bad) ++fails;
  }

  HIPCHK(hipFree(dA)); HIPCHK(hipFree(dB)); HIPCHK(hipFree(dC));
  printf(fails ? "SPIKE FAILED (%d)\n" : "SPIKE ALL PASS\n", fails);
  return fails ? 1 : 0;
}
