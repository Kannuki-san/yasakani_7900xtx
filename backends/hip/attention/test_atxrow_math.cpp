// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// CPU self-check of the row-KV attention geometry (q35_atxrow_native.hip):
// per-pair split ranges must partition [0, len) exactly once INCLUDING the self
// key (len-1), and the pair -> (token, q-head) mapping must cover each pair once.
// Build: g++ -O2 -std=c++17 test_atxrow_math.cpp -o /tmp/test_atxrow_math && /tmp/test_atxrow_math
#include <cassert>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

static void check_case(int M, int base, int splits, int splits_short, int long_len, const std::vector<int>& wk, int G, int NH) {
  const int ncta = (M + 1) / 2;
  for (int bx = 0; bx < ncta; ++bx) {
    const int t0 = bx * 2;
    // pair coverage: each valid (tk, hd-within-kvh) appears exactly once across the 16 pair slots
    std::vector<int> seen(NH, 0);
    for (int kvh = 0; kvh < NH / G; ++kvh) {
      for (int i = 0; i < 16; ++i) {
        const int tk = t0 + i / G;
        const bool v = i < 2 * G && tk < M;
        if (!v) continue;
        const int hd = kvh * G + i % G;
        seen[hd]++;
        const int dp = wk.empty() ? tk : wk[tk];
        const int len = base + dp + 1;
        // split ranges partition [0, len) exactly once, self included
        std::vector<int> cover(len, 0);
        for (int split = 0; split < splits; ++split) {
          const int S = (splits_short > 0 && len < long_len) ? splits_short : splits;
          if (split >= S) continue;                       // act = v && split < S
          const int per = (len + S - 1) / S;
          const int lo = split * per, hi = std::min(len, lo + per);
          assert(hi <= len && lo >= 0);   // lo > hi = empty range (kernel masks it out; pk3-proven)
          for (int k = lo; k < hi; ++k) cover[k]++;
        }
        for (int k = 0; k < len; ++k) assert(cover[k] == 1);   // exact partition incl. self k=len-1
        // tile range math (KT=16) covers [klo,khi) of this CTA's active pairs
      }
    }
    for (int hd = 0; hd < NH; ++hd) {
      // each head appears once per valid token of this CTA (t0 and t0+1)
      const int want = (t0 < M ? 1 : 0) + (t0 + 1 < M ? 1 : 0);
      assert(seen[hd] == want);
    }
  }
}

int main() {
  std::mt19937 rng(7);
  const int G = 6, NH = 24;
  for (int iter = 0; iter < 300; ++iter) {
    const int M = 1 + (int)(rng() % 130);
    const int base = (int)(rng() % 8192);
    const int splits = 1 + (int)(rng() % 42);
    const int splits_short = (iter % 3 == 0) ? 1 + (int)(rng() % splits) : 0;   // engine invariant: short <= splits
    const int long_len = 4096;
    std::vector<int> wk;
    if (iter % 4 == 0) { wk.resize(M); for (int i = 0; i < M; ++i) wk[i] = (int)(rng() % 16); }
    check_case(M, base, splits, splits_short, long_len, wk, G, NH);
  }
  // tile range: tbeg/ntiles must cover every key in [klo,khi)
  for (int iter = 0; iter < 1000; ++iter) {
    const int klo = (int)(rng() % 8192), khi = klo + (int)(rng() % 512);
    const int KT = 16, tbeg = (klo / KT) * KT, ntiles = (khi - tbeg + KT - 1) / KT;
    assert(tbeg <= klo && tbeg + ntiles * KT >= khi);
    assert(ntiles == 0 || tbeg + (ntiles - 1) * KT < khi);
  }
  std::printf("ALL PASS (geometry partition incl. self, pair coverage, tile ranges)\n");
  return 0;
}
