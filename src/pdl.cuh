// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once






#include <cuda_runtime.h>
#include <utility>
#include <map>
#include <cstdio>
#include <cstdlib>
#if !defined(_WIN32)
#include <dlfcn.h>   // dladdr (POSIX only); the MSVC CUDA build keeps the profiler without symbol names
#endif
#include <cstring>
namespace yskn {
// Lightweight per-kernel GPU-time profiler (YSKN_PROFILE=1). Serializes launches (hipEventSynchronize)
// to measure each kernel accurately; key = host kernel stub address (match via nm libyskn.so).
inline uint64_t& kprof_lno_bytes() { static uint64_t b = 0; return b; }
inline bool kprof_on() { static const int v = std::getenv("YSKN_PROFILE") ? std::atoi(std::getenv("YSKN_PROFILE")) : 0; return v != 0; }
inline std::map<const void*, std::pair<double, long>>& kprof_map() { static std::map<const void*, std::pair<double, long>> m; return m; }
inline const char* kprof_sym(const void* p) {
#if !defined(_WIN32)
  Dl_info di; if (dladdr(p, &di) && di.dli_sname) return di.dli_sname;
#else
  (void)p;
#endif
  return "?";
}
inline void kprof_dump() {
  auto& m = kprof_map(); double tot = 0; for (auto& kv : m) tot += kv.second.first;
  std::fprintf(stderr, "==== KPROFILE total %.1f ms, %zu kernels ====\n", tot, m.size());
  double lno_ms = 0; for (auto& kv : m) { if (std::strstr(kprof_sym(kv.first), "lnord_kernel")) lno_ms += kv.second.first; }
  if (lno_ms > 0 && kprof_lno_bytes() > 0) std::fprintf(stderr, "LNORD weight bytes %.3f GB / %.1f ms = %.1f GB/s (DRAM read-only estimate)\n", kprof_lno_bytes()/1e9, lno_ms, kprof_lno_bytes()/1e9/(lno_ms/1e3));
  std::map<double, const void*> bytime; for (auto& kv : m) bytime[kv.second.first] = kv.first;
  int rank = 0;
  for (auto it = bytime.rbegin(); it != bytime.rend() && rank < 30; ++it, ++rank) {
    auto& e = m[it->second];
    const char* nm = kprof_sym(it->second);
    std::fprintf(stderr, "%9.2f ms  %6.2f%%  %7ld calls  %8.3f us/call  %s\n", e.first, 100.0 * e.first / tot, e.second, 1000.0 * e.first / e.second, nm);
  }
}
inline void kprof_init() { static bool done = false; if (!done && kprof_on()) { std::atexit(kprof_dump); done = true; } }
// Launch-failure attribution diagnostic (YSKN_KDBG=1, HIP only): after every launch,
// peek the runtime error (non-clearing, so normal error semantics are preserved) and
// log the kernel name (dladdr on the host stub), grid, block and dynamic LDS. The
// FIRST "[kdbg] FAIL" line in a run names the culprit kernel; later lines are echoes
// of the same sticky error. Used to localize gfx1100 launch-limit failures (LDS>64KB).
#if defined(__HIP_PLATFORM_AMD__)
inline bool kdbg_on() { static const bool v = [] { const char* e = std::getenv("YSKN_KDBG"); return e && e[0] == '1'; }(); return v; }
inline void kdbg_check(const void* kern, dim3 grid, dim3 block, size_t smem) {
  if (!kdbg_on()) return;
  const hipError_t e = hipPeekAtLastError();
  if (e == hipSuccess) return;
  Dl_info info; const char* nm = (dladdr(kern, &info) && info.dli_sname) ? info.dli_sname : "?";
  std::fprintf(stderr, "[kdbg] FAIL %s grid=(%u,%u,%u) block=(%u,%u,%u) smem=%zu: %s\n",
               nm, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, hipGetErrorString(e));
}
#else
inline void kdbg_check(const void*, dim3, dim3, size_t) {}
#endif
// Programmatic dependent launch (griddepcontrol) needs compute capability 9.0. The sm_86 / sm_89 code gets empty stubs, and
// pdl_enabled() (engine.cpp) never sets the launch attribute unless the GPU runs sm_90+ code (gpu_caps.cu).
#if !defined(__HIP_PLATFORM_AMD__) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900)
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
#else
__device__ __forceinline__ void pdl_wait() {}
__device__ __forceinline__ void pdl_trigger() {}
#endif
bool pdl_enabled();
template <typename... KArgs, typename... Args>
inline void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  if (kprof_on()) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cs);
    if (cs == cudaStreamCaptureStatusNone) {   // never synchronize during graph capture
      kprof_init();
      static cudaEvent_t a = nullptr, b = nullptr;
      if (!a) { cudaEventCreate(&a); cudaEventCreate(&b); }
      cudaEventRecord(a, st);
      kern<<<grid, block, smem, st>>>(args...);
      kdbg_check((const void*)kern, grid, block, smem);
      cudaEventRecord(b, st); cudaEventSynchronize(b);
      float ms = 0; cudaEventElapsedTime(&ms, a, b);
      auto& e = kprof_map()[(const void*)kern]; e.first += ms; e.second++;
      return;
    }
  }
  if (!pdl_enabled()) { kern<<<grid, block, smem, st>>>(args...); kdbg_check((const void*)kern, grid, block, smem); return; }
#if defined(__HIP_PLATFORM_AMD__)
  kern<<<grid, block, smem, st>>>(args...);   // programmatic dependent launch is NVIDIA-only; pdl_enabled() is false on AMD
  kdbg_check((const void*)kern, grid, block, smem);
#else
  cudaLaunchConfig_t cfg{}; cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, args...);
#endif
}
} 
