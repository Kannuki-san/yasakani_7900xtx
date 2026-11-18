// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// CUDA -> HIP name compatibility shim (HIP backend only; found via include path before the real headers).
// The engine source stays single-source: CUDA builds use the real NVIDIA headers, HIP builds use this shim.
// Only names the engine actually uses are mapped; add a mapping here when new CUDA API use is introduced.
#pragma once
#include <hip/hip_runtime.h>
// Include the fp16/bf16 headers BEFORE the name macros below: they declare warp-sync functions
// whose declarations would otherwise be mangled by the macros.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// types
#define cudaError_t                  hipError_t
#define cudaStream_t                 hipStream_t
#define cudaEvent_t                  hipEvent_t
#define cudaGraph_t                  hipGraph_t
#define cudaGraphExec_t              hipGraphExec_t
#define cudaDeviceProp               hipDeviceProp_t
#define cudaFuncAttributes           hipFuncAttributes
#define cudaLaunchConfig_t           hipLaunchConfig_t
#define cudaLaunchAttribute          hipLaunchAttribute

// constants / enums
#define cudaSuccess                  hipSuccess
#define cudaErrorInvalidValue        hipErrorInvalidValue
#define cudaErrorUnknown             hipErrorUnknown
#define cudaMemcpyHostToDevice       hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost       hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice     hipMemcpyDeviceToDevice
#define cudaStreamNonBlocking        hipStreamNonBlocking
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaEventDisableTiming       hipEventDisableTiming
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin

// functions
#define cudaGetDevice                hipGetDevice
#define cudaSetDevice                hipSetDevice
#define cudaGetDeviceProperties      hipGetDeviceProperties
#define cudaDeviceGetAttribute       hipDeviceGetAttribute
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaDeviceSynchronize        hipDeviceSynchronize
#define cudaGetLastError             hipGetLastError
#define cudaPeekAtLastError          hipPeekAtLastError
#define cudaGetErrorName             hipGetErrorName
#define cudaGetErrorString           hipGetErrorString
// devmem.hpp redefines these four as function-like tracking macros; the engine bypasses them
// with the (cudaMalloc)(...) idiom, which needs a real function here (as in CUDA).
// cudaMalloc/cudaMallocHost are templates in CUDA (any pointer type).
template <typename T> inline hipError_t cudaMalloc(T** p, size_t n) { return hipMalloc((void**)p, n); }
inline hipError_t cudaFree(void* p) { return hipFree(p); }
template <typename T> inline hipError_t cudaMallocHost(T** p, size_t n) { return hipHostMalloc((void**)p, n); }
inline hipError_t cudaFreeHost(void* p) { return hipHostFree(p); }
#define cudaHostGetDevicePointer     hipHostGetDevicePointer
#define cudaMemcpy                   hipMemcpy
#define cudaMemcpyAsync              hipMemcpyAsync
#define cudaMemcpyToSymbolAsync      hipMemcpyToSymbolAsync
#define cudaStreamIsCapturing        hipStreamIsCapturing
#define cudaStreamCaptureStatus      hipStreamCaptureStatus
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusNone   hipStreamCaptureStatusNone
#define cudaMemset                   hipMemset
#define cudaMemsetAsync              hipMemsetAsync
#define cudaMemGetInfo               hipMemGetInfo
#define cudaStreamCreateWithFlags    hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy            hipStreamDestroy
#define cudaStreamSynchronize        hipStreamSynchronize
#define cudaStreamWaitEvent          hipStreamWaitEvent
#define cudaStreamBeginCapture       hipStreamBeginCapture
#define cudaStreamEndCapture         hipStreamEndCapture
#define cudaEventCreate              hipEventCreate
#define cudaEventCreateWithFlags     hipEventCreateWithFlags
#define cudaEventDestroy             hipEventDestroy
#define cudaEventRecord              hipEventRecord
#define cudaEventSynchronize         hipEventSynchronize
#define cudaEventElapsedTime         hipEventElapsedTime
#define cudaGraphDestroy             hipGraphDestroy
#define cudaGraphExecDestroy         hipGraphExecDestroy
#define cudaGraphLaunch              hipGraphLaunch
#define cudaFuncGetAttributes        hipFuncGetAttributes
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaLaunchKernelEx           hipLaunchKernelEx
// CUDA 12+ 3-arg cudaGraphInstantiate == HIP's WithFlags variant (the plain hipGraphInstantiate is the old 5-arg form).
#define cudaGraphInstantiate         hipGraphInstantiateWithFlags

// ---- Native ROCm math (ocml) ----
// expf/sinf/cosf/powf/log1pf/exp2f/rsqrtf use HIP's native ocml implementations (Radeon ecosystem).
// Small ulp-level differences vs NVIDIA are expected and acceptable (tolerance selfcheck mode).
#if defined(__HIP_DEVICE_COMPILE__)
// HIP's default __fsqrt_rn is the hardware-approximate native sqrt; the engine expects IEEE
// correctly-rounded sqrt.rn. Redirect to ocml sqrt with the correctly-rounded config bitcode
// linked (oclc_correctly_rounded_sqrt_on.bc, see CMakeLists.txt). This is ocml-native.
extern "C" __device__ float __ocml_sqrt_f32(float);
#define __fsqrt_rn __ocml_sqrt_f32
#endif

// __byte_perm: HIP's native implementation ignores selector bit3 (sign-replication), but every
// engine selector uses nibbles <= 7, where native HIP is bit-exact with CUDA (verified).
// Do NOT use __builtin_amdgcn_perm here: it is not v_perm_b32 and returns garbage (0xFFFF...).
// If bit3 selectors are ever needed, implement via inline asm v_perm_b32 instead.

// ---- name-level differences that need wrappers, not macros ----

// cudaFuncSetAttribute/GetAttributes accept any function pointer in CUDA; HIP wants const void*.
#undef cudaFuncSetAttribute
#undef cudaFuncGetAttributes
template <typename T> inline hipError_t cudaFuncSetAttribute(T func, hipFuncAttribute attr, int value) {
  return hipFuncSetAttribute((const void*)func, attr, value);
}
template <typename T> inline hipError_t cudaFuncGetAttributes(hipFuncAttributes* attr, T func) {
  return hipFuncGetAttributes(attr, (const void*)func);
}

// ---- Cache-policy loads/stores (gfx1100/RDNA3; Q2) ----
// CUDA hint -> RDNA3 cache bits (LLVM 21 SIMemoryLegalizer semantics, ISA probe-verified):
//   __ldg/__ldcs (read-only / streaming) -> nontemporal load = slc+dlc:
//     L0/L1 HIT_EVICT, L2 STREAM (evict-first, still allocates), MALL NOALLOC.
//     Weights are streamed read-once and never fit the 96MB MALL; NOALLOC keeps KV/activations
//     cached there. Small hot read-only data (LUT/qx/alpha) still hits L2 (STREAM allocates).
//     RDNA3 has no non-coherent texture path, so this is the RDNA3 tuning of the read-only hint.
//   __ldcg (cache-global = L2-only) -> relaxed atomic load = glc: L0/L1 MISS_EVICT, read at L2.
//     acc partials are produced by atomicAdd at L2; glc reads see them without L0/L1 staleness.
//     16B types (uint4 KV rows): no compile-safe glc load builtin exists -> plain load (still
//     L2-cached). Inline asm was rejected: the compiler would not insert s_waitcnt for it.
//   __stcg -> plain store: RDNA3 vector stores write through to L2 with no L0/L1 allocation
//     (LLVM lowers monotonic atomic stores to plain stores), i.e. already cg-equivalent.
// Hints are performance-only on AMD (coherence is maintained), so values are bit-identical.
// HIP 7.1 hip_ldg.h declares native non-template __ldg overloads (plain loads) that beat generic
// templates in overload resolution, so the new mappings are applied via function-like macros.
// Kill switch: YSKN_HIP_CACHE_HINTS=0 restores the old plain loads/stores (cmake env/-D, rebuild).
#ifndef YSKN_HIP_CACHE_HINTS
#define YSKN_HIP_CACHE_HINTS 1
#endif
#if YSKN_HIP_CACHE_HINTS
template <typename T> __device__ __forceinline__ T yskn_ld_nt(const T* p) {
  if constexpr (sizeof(T) == 1)
    return __builtin_bit_cast(T, __builtin_nontemporal_load(reinterpret_cast<const unsigned char*>(p)));
  else if constexpr (sizeof(T) == 2)
    return __builtin_bit_cast(T, __builtin_nontemporal_load(reinterpret_cast<const unsigned short*>(p)));
  else {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, "yskn_ld_nt: unsupported size");
    using V = unsigned __attribute__((ext_vector_type(sizeof(T) / 4)));
    return __builtin_bit_cast(T, __builtin_nontemporal_load(reinterpret_cast<const V*>(p)));
  }
}
template <typename T> __device__ __forceinline__ T yskn_ld_glc(const T* p) {
  if constexpr (sizeof(T) == 1)
    return __builtin_bit_cast(T, __atomic_load_n(reinterpret_cast<const unsigned char*>(p), __ATOMIC_RELAXED));
  else if constexpr (sizeof(T) == 2)
    return __builtin_bit_cast(T, __atomic_load_n(reinterpret_cast<const unsigned short*>(p), __ATOMIC_RELAXED));
  else if constexpr (sizeof(T) == 4)
    return __builtin_bit_cast(T, __atomic_load_n(reinterpret_cast<const unsigned*>(p), __ATOMIC_RELAXED));
  else if constexpr (sizeof(T) == 8)
    return __builtin_bit_cast(T, __atomic_load_n(reinterpret_cast<const unsigned long long*>(p), __ATOMIC_RELAXED));
  else
    return *p;   // 16B: plain fallback (see comment above)
}
#define __ldg(p)  yskn_ld_nt(p)
#define __ldcs(p) yskn_ld_nt(p)
#define __ldcg(p) yskn_ld_glc(p)
template <typename T> __device__ inline void __stcg(T* p, T v) { *p = v; }
#else
// Old Phase-0 behavior: cache hints dropped (plain loads/stores).
template <typename T> __device__ inline T __ldg(const T* p) { return *p; }
template <typename T> __device__ inline T __ldcg(const T* p) { return *p; }
template <typename T> __device__ inline T __ldcs(const T* p) { return *p; }
template <typename T> __device__ inline void __stcg(T* p, T v) { *p = v; }
#endif

// Warp-sync masks: HIP requires mask == activemask at the call site (__hip_check_mask traps
// otherwise), while CUDA code passes 0xffffffff even under divergence. Use the actual active
// mask: identical semantics for convergent code, and safe for the engine's divergent reductions.
// Self-referential macros are not re-expanded, so the real HIP template is called.
#define __shfl_sync(m, ...)      __shfl_sync(__activemask(), __VA_ARGS__)
#define __shfl_xor_sync(m, ...)  __shfl_xor_sync(__activemask(), __VA_ARGS__)
#define __shfl_down_sync(m, ...) __shfl_down_sync(__activemask(), __VA_ARGS__)
#define __shfl_up_sync(m, ...)   __shfl_up_sync(__activemask(), __VA_ARGS__)
// HIP 7.1's native __any_sync/__all_sync/__ballot_sync are broken on gfx1100 wave32
// (any(all-false) returns 1, ballot returns garbage, divergent calls trap the queue).
// Implement them directly from the wave32 ballot builtin (verified correct).
__device__ __forceinline__ unsigned yskn_ballot_w32(int p) { return __builtin_amdgcn_ballot_w32(p); }
__device__ __forceinline__ int yskn_any_sync(unsigned long long m, int p) { return (yskn_ballot_w32(p) & (unsigned)m) != 0; }
__device__ __forceinline__ int yskn_all_sync(unsigned long long m, int p) { unsigned b = yskn_ballot_w32(p); return (b & (unsigned)m) == (unsigned)m; }
#define __any_sync(m, ...)       yskn_any_sync((unsigned long long)(m), __VA_ARGS__)
#define __all_sync(m, ...)       yskn_all_sync((unsigned long long)(m), __VA_ARGS__)
#define __ballot_sync(m, ...)    yskn_ballot_w32(__VA_ARGS__)
__device__ inline void yskn_syncwarp(unsigned long long m = 0xffffffffull) { __syncwarp(__activemask()); }
#define __syncwarp yskn_syncwarp

__device__ inline void __nanosleep(unsigned ns) {
#ifdef __HIP_DEVICE_COMPILE__
  (void)ns; __builtin_amdgcn_s_sleep(8);   // spin-wait hint; s_sleep needs an immediate, a fixed short sleep is fine
#else
  (void)ns;
#endif
}
__device__ inline void __trap() { __builtin_trap(); }

// int8x4 dot product (signed bytes, accumulate into int). Exact integer math; the backend
// pattern-matches this to v_dot4_i32_i8 where available. (The sdot4 builtin needs dot1-insts
// enabled explicitly; the plain form is portable and just as exact.)
__device__ inline int __dp4a(int a, int b, int c) {
  int r = c;
  #pragma unroll
  for (int i = 0; i < 4; ++i) r += ((a << (24 - 8 * i)) >> 24) * ((b << (24 - 8 * i)) >> 24);
  return r;
}

// Generic->shared address: on AMD the LDS offset is the low 32 bits of the generic pointer.
// (The arch wrappers in gpu_arch.cuh take generic pointers directly; this is a fallback.)
__device__ inline unsigned __cvta_generic_to_shared(const void* p) { return (unsigned)(uintptr_t)p; }

// Phase 0: launch bounds dropped on HIP (clang rejects template-dependent __launch_bounds__
// expressions). Occupancy tuning returns in Phase 7 with amdgpu_waves_per_eu attributes.
#undef __launch_bounds__
#define __launch_bounds__(...)
