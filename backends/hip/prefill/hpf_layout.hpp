// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Pure index math for the G6 predecode block layout -> row-major repack.
// No HIP/CUDA dependencies so a host compiler can unit-check it (test_hpf_math.cpp).
#pragma once
#include <cstdint>
#include <cstddef>
#if defined(__CUDACC__) || defined(__HIP__)
#define HPF_HD __host__ __device__ __forceinline__
#else
#define HPF_HD inline
#endif
namespace yskn {
// G6 predecode block layout (mt_predecode2, src/q35_g6.cu): per (16-row tile, 256-K block),
// 256 uint4 words; word (32*m + 4*gid + tig) holds four int8 of one row at K offset
// 32*m + 16*half + 4*tig, component = 2*hi + half (hi = row>=8 within the tile).
// Each uint32 component is 4 consecutive K values of row (tile*16 + 8*hi + gid).
HPF_HD size_t hpf_repack_src(int n, int k4, int Kc) {
  const int tile = n >> 4, r = n & 15, gid = r & 7, hi = r >> 3;
  const int k = k4 * 4, bi = k >> 8, kk = k & 255, m = kk >> 5, rem = kk & 31;
  const int half = rem >> 4, tig = (rem & 15) >> 2;
  const size_t word = ((size_t)tile * (Kc >> 8) + bi) * 256 + 32 * m + 4 * gid + tig;
  return word * 4 + 2 * hi + half;   // uint32 index into the G6-layout buffer
}
}
