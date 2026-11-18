// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Software e4m3 fp8 shim for the HIP backend (RDNA3 has no fp8 hardware; the engine only uses
// e4m3 as a quantization container in kvx_codec.cuh). Must match CUDA's __nv_fp8_e4m3 exactly:
// round-to-nearest-even, SATFINITE clamps to +/-448, 0x7F/0xFF are NaN (e4m3 has no infinity).
#pragma once
#include <cmath>

enum __nv_fp8_interpretation_t { __NV_E4M3 = 0, __NV_E5M2 = 1 };
enum __nv_saturation_t { __NV_NOSAT = 0, __NV_SATFINITE = 1 };

struct __nv_fp8_e4m3 {
  unsigned char __x;
  __host__ __device__ operator float() const {
    unsigned sign = __x >> 7, exp = (__x >> 3) & 0xFu, man = __x & 7u;
    float v;
    if (exp == 0xFu && man == 7u) return NAN;
    if (exp == 0) v = std::ldexp((float)man, -9);              // subnormal: man * 2^-9
    else v = std::ldexp(1.0f + man * 0.125f, (int)exp - 7);
    return sign ? -v : v;
  }
};

__host__ __device__ inline unsigned char __nv_cvt_float_to_fp8(float x, __nv_saturation_t sat, __nv_fp8_interpretation_t interp) {
  (void)sat; (void)interp;   // the engine only calls this with (__NV_SATFINITE, __NV_E4M3)
  if (std::isnan(x)) return 0x7F;
  unsigned char sign = std::signbit(x) ? 0x80 : 0;
  float a = std::fabs(x);
  if (a >= 448.0f) return sign | 0x7E;                        // max finite (satfinite)
  if (a < 0x1p-6f) {                                          // subnormal range: multiples of 2^-9
    int q = (int)std::rint(a * 512.0f);                       // rint = round-to-nearest-even
    return sign | (unsigned char)q;
  }
  int e;
  std::frexp(a, &e);                                          // a = m * 2^e, m in [0.5, 1)
  int E = e + 6;                                              // e4m3 exponent field (bias 7)
  float scaled = std::ldexp(a, -(E - 7));                     // in [1, 2)
  int man = (int)std::rint((scaled - 1.0f) * 8.0f);
  if (man == 8) { man = 0; ++E; }
  if (E > 15 || (E == 15 && man == 7)) return sign | 0x7E;    // would be NaN/overflow -> satfinite
  return sign | (unsigned char)(E << 3) | (unsigned char)man;
}
