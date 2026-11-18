// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include <hip/hip_bf16.h>

#define __nv_bfloat162  __hip_bfloat162
// HIP's __float2bfloat16 converts via clang's __bf16 cast: round-to-nearest-even, same as CUDA's _rn.
#define __float2bfloat16_rn __float2bfloat16

#ifndef YSKN_HIP_BF162_SHIM
#define YSKN_HIP_BF162_SHIM
__host__ __device__ inline __hip_bfloat162 __floats2bfloat162_rn(float lo, float hi) {
  __hip_bfloat162 r;
  r.x = __float2bfloat16(lo);
  r.y = __float2bfloat16(hi);
  return r;
}
#endif
