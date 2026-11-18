cmake_minimum_required(VERSION 3.24)
# GPU backend: CUDA (NVIDIA, the release build) or HIP (AMD Radeon, native port in progress).
set(YSKN_BACKEND "CUDA" CACHE STRING "GPU backend: CUDA or HIP")
if(YSKN_BACKEND STREQUAL "HIP")
  # RDNA3 (gfx1100 = RX 7900 XTX). wave32 is the native width and matches CUDA's warp size.
  set(CMAKE_HIP_ARCHITECTURES "gfx1100" CACHE STRING "HIP GPU architectures")
  project(yasakani_engine LANGUAGES CXX HIP)
else()
  # GPU code for RTX 30 (sm_86), RTX 40 (sm_89) and RTX 50 (sm_120) in one library. PTX is kept for compute_120 only, so that
  # newer 12.x GPUs can compile it at load time. To build one architecture, pass e.g. -DCMAKE_CUDA_ARCHITECTURES=120 (or 86, 89).
  set(CMAKE_CUDA_ARCHITECTURES "86-real;89-real;120" CACHE STRING "CUDA architectures to build (default: RTX 30, 40 and 50 series)")
  project(yasakani_engine LANGUAGES CXX CUDA)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(YSKN_BACKEND STREQUAL "HIP")
  set(CMAKE_HIP_STANDARD 17)
else()
  set(CMAKE_CUDA_STANDARD 17)
  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 LESS 86)
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES contains ${arch}: the engine needs compute capability 8.6 or newer (86, 89, 120).")
    endif()
  endforeach()
  find_package(CUDAToolkit REQUIRED)
endif()

set(YSKN_SOURCES
  src/attention_gqa.cu
  src/attention_kv4q.cu
  src/c_api.cpp
  src/devmem.cpp
  src/draft_kernels.cu
  src/engine.cpp
  src/engine35.cpp
  src/gemm_compressed.cu
  src/gemv_sigma_int.cu
  src/gpu_caps.cu
  src/kvx_probe.cu
  src/lnord.cu
  src/loader.cpp
  src/misc_kernels.cu
  src/pfa.cu
  src/pm_test.cu
  src/post_kernels.cu
  src/q35_atx.cu
  src/q35_atxpk.cu
  src/q35_bdraft.cu
  src/q35_chk.cu
  src/q35_fa.cu
  src/q35_flow.cu
  src/q35_g6.cu
  src/q35_kernels.cu
  src/q35_kz.cu
  src/q35_prefill.cu
  src/q35_sample.cu
  src/repack.cu
  src/specx.cpp
  src/test_hooks.cpp
)
set(YSKN_KVX_MODE 3 CACHE STRING "KV codec mode: 0=bf16 plain attention, 1=fp8, 2/3=rotated INT8 (3=trellis)")
add_library(yskn SHARED ${YSKN_SOURCES})
if(YSKN_BACKEND STREQUAL "HIP")
  # Same sources, compiled as HIP; the shim headers in src/hip_shim map cuda* names to hip*.
  set_source_files_properties(${YSKN_SOURCES} PROPERTIES LANGUAGE HIP)
  target_include_directories(yskn PRIVATE src/hip_shim src include third_party)
  target_compile_definitions(yskn PRIVATE YSKN_BUILD YSKN_KVX_MODE=${YSKN_KVX_MODE} NOMINMAX)
  # -ffp-contract=off is the HIP equivalent of nvcc --fmad=false (keeps the arithmetic identical to the reference).
  # -include: nvcc implicitly includes cuda_runtime.h in every TU; HIP does not, so force the shim (which includes hip_runtime.h).
  target_compile_options(yskn PRIVATE -O3 -ffp-contract=off -include cuda_runtime.h)
  # Native ROCm math (ocml) is used for expf/sinf/cosf/etc. No NVIDIA libdevice dependency.
  # ocml correctly-rounded sqrt config: makes __ocml_sqrt_f32 (the __fsqrt_rn redirect) IEEE RN.
  execute_process(COMMAND "${CMAKE_HIP_COMPILER}" -print-resource-dir OUTPUT_VARIABLE YSKN_CLANG_RES OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(YSKN_SQRT_CFG "${YSKN_CLANG_RES}/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc")
  if(EXISTS "${YSKN_SQRT_CFG}")
    target_compile_options(yskn PRIVATE "SHELL:-Xclang -mlink-builtin-bitcode -Xclang ${YSKN_SQRT_CFG}")
  endif()
  # AMD-native trellis/pk3 attention (RDNA3, backends/hip/attention) is the HIP default
  # (verified on gfx1100: selfcheck 98.9% top-1 vs the public RTX 5090 reference, sure-changed 0).
  # The native launcher falls back to the existing path for unsupported configs and via YSKN_ATX_NATIVE=0.
  option(YSKN_NATIVE_ATXPK "Build the AMD-native trellis/pk3 attention (gfx1100)" ON)
  if(YSKN_NATIVE_ATXPK)
    target_sources(yskn PRIVATE backends/hip/attention/q35_atxpk_native.hip)
    set_source_files_properties(backends/hip/attention/q35_atxpk_native.hip PROPERTIES LANGUAGE HIP)
    target_compile_definitions(yskn PRIVATE YSKN_HIP_NATIVE_ATXPK)
  endif()
  # AMD-native INT8-row KV attention for non-packed contexts (MTP layer etc.):
  # the converted q35_atx tiles exceed gfx1100's 64KB LDS and its NS=1 fallback
  # spills heavily; this reuses the proven pk3-native WMMA/softmax patterns.
  option(YSKN_NATIVE_ATXROW "Build the AMD-native INT8-row KV attention (gfx1100)" ON)
  if(YSKN_NATIVE_ATXROW)
    target_sources(yskn PRIVATE backends/hip/attention/q35_atxrow_native.hip)
    set_source_files_properties(backends/hip/attention/q35_atxrow_native.hip PROPERTIES LANGUAGE HIP)
    target_compile_definitions(yskn PRIVATE YSKN_HIP_NATIVE_ATXROW)
  endif()
  # AMD-native prefill GEMM (hipBLAS int8 per-group + fused dequant, backends/hip/prefill).
  # Compiled in and enabled by default on HIP (GPU-verified: all-tensor bit-exact vs lnord,
  # selfcheck 98.9%); kill switch YSKN_HIP_PREFILL=0 restores the PFA/lnord path.
  # Decode (M<=16) stays on lnord; the CUDA build is unaffected.
  option(YSKN_PREFILL_GEMM "Build the AMD-native hipBLAS prefill GEMM (gfx1100)" ON)
  if(YSKN_PREFILL_GEMM)
    target_sources(yskn PRIVATE backends/hip/prefill/hip_prefill.hip)
    set_source_files_properties(backends/hip/prefill/hip_prefill.hip PROPERTIES LANGUAGE HIP)
    target_compile_definitions(yskn PRIVATE YSKN_HIP_PREFILL_GEMM)
    target_link_libraries(yskn PRIVATE hipblas)
  endif()
  set_target_properties(yskn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
else()
  target_include_directories(yskn PRIVATE src include third_party)
  target_compile_definitions(yskn PRIVATE YSKN_BUILD YSKN_KVX_MODE=${YSKN_KVX_MODE} NOMINMAX _CRT_SECURE_NO_WARNINGS)
  if(MSVC)
    target_compile_options(yskn PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj;/O2;/d1trimfile:${CMAKE_SOURCE_DIR}/>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8;-Xcompiler=/bigobj;-Xcompiler=/d1trimfile:${CMAKE_SOURCE_DIR}/;-O3;--fmad=false>)
  else()
    # Same CUDA code generation as the MSVC build: --fmad=false keeps the output identical to the reference.
    target_compile_options(yskn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false>)
    # build/bin/libyskn.so is one of the places the Python package looks, so --dll is not needed.
    set_target_properties(yskn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  endif()
  target_link_libraries(yskn PRIVATE CUDA::cudart)
endif()
