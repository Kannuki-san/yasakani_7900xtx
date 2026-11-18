// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// CPU self-check for the hip prefill GEMM support math (no GPU needed):
//   1. hpf_repack_src inverts the mt_predecode2 store layout (src/q35_g6.cu) exactly.
//   2. per-group int32 partials cannot overflow (512*127*127 < 2^31) and the
//      G6/lnord-XSM dequant (round(alpha*2^24/max) * partial, int64 sum, *2^-24 max)
//      equals the same scaling applied to full int64 group dots.
// Build: g++ -O2 -std=c++17 -I. test_hpf_math.cpp -o /tmp/test_hpf_math && /tmp/test_hpf_math
#include "hpf_layout.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
using namespace yskn;

// mt_predecode2 store, transcribed: s_t[warp][row][kw] = the uint32 holding 4 int8 of
// row (tile*16+row) at K = bi*256 + 4*kw; dst[32*m+lane] = uint4(s_t[gid][8m+tig],
// s_t[gid][8m+4+tig], s_t[gid+8][8m+tig], s_t[gid+8][8m+4+tig]), lane = 4*gid+tig.
static void build_g6_layout(const std::vector<int8_t>& rm, int N, int Kc, std::vector<uint32_t>& g6) {
  g6.assign((size_t)(N / 16) * (Kc / 256) * 256 * 4, 0);
  auto word = [&](int row, int k) {
    uint32_t w; std::memcpy(&w, &rm[(size_t)row * Kc + k], 4); return w;
  };
  for (int tile = 0; tile < N / 16; ++tile)
    for (int bi = 0; bi < Kc / 256; ++bi) {
      uint32_t* dst = g6.data() + ((size_t)tile * (Kc / 256) + bi) * 256 * 4;
      for (int m = 0; m < 8; ++m)
        for (int lane = 0; lane < 32; ++lane) {
          const int gid = lane >> 2, tig = lane & 3;
          const int r0 = tile * 16 + gid, r1 = r0 + 8, k0 = bi * 256 + 32 * m + 4 * tig;
          uint32_t v[4] = {word(r0, k0), word(r0, k0 + 16), word(r1, k0), word(r1, k0 + 16)};
          std::memcpy(dst + (32 * m + lane) * 4, v, 16);
        }
    }
}

static void check_repack(int N, int Kc, std::mt19937& rng) {
  std::vector<int8_t> rm((size_t)N * Kc);
  for (auto& v : rm) v = (int8_t)((int)(rng() % 255) - 127);
  std::vector<uint32_t> g6; build_g6_layout(rm, N, Kc, g6);
  const uint32_t* src = g6.data();
  const uint32_t* want = reinterpret_cast<const uint32_t*>(rm.data());
  for (int n = 0; n < N; ++n)
    for (int k4 = 0; k4 < Kc / 4; ++k4) {
      const size_t idx = (size_t)n * (Kc / 4) + k4;
      assert(src[hpf_repack_src(n, k4, Kc)] == want[idx]);
    }
  std::printf("repack N=%d Kc=%d: all %zu words exact\n", N, Kc, (size_t)N * Kc / 4);
}

static void check_dequant(int M, int N, int Kc, std::mt19937& rng, bool full) {
  const int G = Kc / 512;
  std::vector<int8_t> qx((size_t)M * Kc), w((size_t)N * Kc);
  std::vector<float> alpha((size_t)M * G);
  for (auto& v : qx) v = (int8_t)((int)(rng() % 255) - 127);
  for (auto& v : w) v = (int8_t)((int)(rng() % 255) - 127);
  for (int m = 0; m < M; ++m) {
    for (int g = 0; g < G; ++g) alpha[(size_t)m * G + g] = (float)(1e-3 + (double)(rng() % 1000000) / 1e6);
    if (m % 7 == 2) alpha[(size_t)m * G + (m % G)] = 3.0e4f;
    if (m % 11 == 3) alpha[(size_t)m * G + ((m + 3) % G)] = 0.f;
    if (m % 13 == 4) alpha[(size_t)m * G + ((m + 5) % G)] = 1.0e-30f;
    if (m == M - 1 && M > 20) for (int g = 0; g < G; ++g) alpha[(size_t)m * G + g] = 0.f;
  }
  const int pairs = full ? M * N : 512;
  for (int p = 0; p < pairs; ++p) {
    const int m = full ? p / N : (int)(rng() % M), n = full ? p % N : (int)(rng() % N);
    float mx = 0.f;
    for (int g = 0; g < G; ++g) mx = std::fmax(mx, alpha[(size_t)m * G + g]);
    const float rm = mx > 0.f ? 16777216.f / mx : 0.f;
    long long acc32 = 0, acc64 = 0;
    for (int g = 0; g < G; ++g) {
      long long dot = 0;
      for (int k = g * 512; k < (g + 1) * 512; ++k)
        dot += (int)qx[(size_t)m * Kc + k] * (int)w[(size_t)n * Kc + k];
      assert(dot >= -8257536 && dot <= 8257536);                 // 512*127*127: int32 safe
      const int32_t partial = (int32_t)dot;                      // what hipBLAS writes
      const int a = (int)std::lrint((double)alpha[(size_t)m * G + g] * rm);   // __float2int_rn
      acc32 += (long long)a * partial;
      acc64 += (long long)a * dot;
    }
    assert(acc32 == acc64);                                      // int32 partials lose nothing
    assert(acc64 > -(1LL << 62) && acc64 < (1LL << 62));         // int64 headroom
    const float y = (float)acc32 * std::ldexp(mx, -24);
    assert(std::isfinite(y));
  }
  std::printf("dequant M=%d N=%d Kc=%d (G=%d): %d pairs exact, int32/int64 bounds ok\n", M, N, Kc, G, pairs);
}

int main() {
  std::mt19937 rng(123);
  check_repack(48, 1536, rng);            // multi-tile, multi-block, small
  check_repack(5120, 6144, rng);          // out_proj shape, every word
  check_repack(17408, 5120, rng);         // gate/up shape, every word
  check_dequant(37, 48, 1536, rng, true);
  check_dequant(64, 5120, 6144, rng, false);
  check_dequant(512, 5120, 5120, rng, false);
  std::printf("ALL PASS\n");
  return 0;
}
