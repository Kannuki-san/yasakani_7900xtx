// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// AMD-native prefill GEMM (gfx1100): trellis weights expanded to int8 (G6 predecode),
// per-K-group int8 GEMM via hipBLAS (int32 exact partials), fused per-group dequant
// epilogue producing the G6 float-y contract (post kernels apply tau[n]).
// Decode (M<=16) stays on lnord; this path is prefill M>=17 only.
#pragma once
#include "kernels.hpp"
#include "hpf_layout.hpp"
#include <hipblas/hipblas.h>
#include <map>
#include <cstdint>
namespace yskn {

struct HipPrefillWorkspace {
  hipblasHandle_t handle = nullptr;
  int8_t* arena = nullptr;      // row-major int8 weight cache, one layer's worth
  size_t arena_bytes = 0, used = 0;
  uint4* scratch_g6 = nullptr;  // G6-layout predecode temp (largest tensor)
  int8_t* scratch_rm = nullptr; // row-major temp for graph-capture passes (largest tensor)
  int32_t* partials = nullptr;  // [G, M, N] int32 per-group dot products
  size_t partial_bytes = 0;
  std::map<const void*, const int8_t*> cached;   // t.lno -> row-major int8 [N, Kc]
  int Mmax = 0;
  bool ok = false;
  void init(size_t arena_b, size_t tensor_b, size_t partial_b, int mmax, cudaStream_t st);
  void destroy();
};

// Returns false when the launch is not eligible or hipBLAS refuses it; the caller
// then falls back to the existing GEMM path. No environment check here (the engine
// hook gates on YSKN_HIP_PREFILL; the test hook calls this directly).
// Env: YSKN_HIP_PREFILL_BATCHED=0 forces the plain-GemmEx per-group loop (also the
// automatic fallback when the batched call is refused); YSKN_HIP_PREFILL_DBG=1 logs.
bool launch_hip_prefill(const GemvLaunch& P, HipPrefillWorkspace& W, int pm, cudaStream_t st);
}
