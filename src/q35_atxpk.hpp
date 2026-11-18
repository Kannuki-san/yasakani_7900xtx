// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once







#include "q35.hpp"
namespace yskn {
struct Q35PkArgs { const uint8_t* kp = nullptr; const uint8_t* vp = nullptr; const float* kps = nullptr; const float* vps = nullptr; const unsigned char* rec = nullptr; int layer = 0; int kb = 4; int variant = 0; };
void launch_q35_atx_pk(const Q35AttnArgs&, const uint8_t* wk, const Q35PkArgs&, cudaStream_t);

// AMD-native trellis/pk3 attention lives in backends/hip/attention (RDNA3).
// With YSKN_HIP_NATIVE_ATXPK (HIP build default ON) the native launcher is the HIP
// default; it falls back to the existing launcher for unsupported configs and via
// the runtime kill switch YSKN_ATX_NATIVE=0. CUDA builds always use the original.
#if defined(YSKN_HIP_NATIVE_ATXPK)
void launch_q35_atx_pk_native(const Q35AttnArgs&, const uint8_t* wk, const Q35PkArgs&, cudaStream_t);
inline void launch_q35_atx_pk_dispatch(const Q35AttnArgs& A, const uint8_t* wk, const Q35PkArgs& P, cudaStream_t st) { launch_q35_atx_pk_native(A, wk, P, st); }
#else
inline void launch_q35_atx_pk_dispatch(const Q35AttnArgs& A, const uint8_t* wk, const Q35PkArgs& P, cudaStream_t st) { launch_q35_atx_pk(A, wk, P, st); }
#endif
}  
