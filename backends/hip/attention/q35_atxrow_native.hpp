// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once
#include "q35.hpp"
namespace yskn {
// AMD-native INT8-row KV attention (backends/hip/attention, RDNA3) for non-packed
// contexts (MTP layer / layers without trellis-packed KV). Returns false when the
// config is unsupported or the launch fails; the caller falls back to the
// converted q35_atx NS=1 path. Kill switch: YSKN_ATXROW_NATIVE=0.
#if defined(YSKN_HIP_NATIVE_ATXROW)
bool launch_q35_atxrow_native(const Q35AttnArgs&, const uint8_t* wk, cudaStream_t);
#endif
}
