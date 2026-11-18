// SPDX-License-Identifier: Apache-2.0
// Trellis requant (encode) probe: runs the engine's launch_tb_fast on known
// synthetic rows and dumps packed codes + scales + reconstruction, so an
// independent CPU Viterbi reference (test_requant_ref.py) can verify the
// encode side. Links against libyskn.so (no CUDA sources compiled here).
//
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1100 -I backends/hip/shim -I src \
//        -include cuda_runtime.h probe_requant.hip -o probe_requant \
//        -L build-integ/bin -lyskn -Wl,-rpath,$PWD/build-integ/bin
// Run:   ./probe_requant <kvq_u5.bin> <out.bin>   (env YSKN_TB_VIT2=0/1)
#include "kvq_fast.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIPCHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

using namespace yskn;

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: probe_requant <blob> <out.bin>\n"); return 1; }
  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { printf("cannot open blob\n"); return 1; }
  std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> blob(n);
  if (std::fread(blob.data(), 1, n, f) != (size_t)n) return 1;
  std::fclose(f);

  int nm = 0, has_cdq = 0;
  if (!tbf_parse_blob(blob.data(), n, nm, has_cdq)) { printf("blob parse failed\n"); return 1; }
  printf("blob: nm=%d has_cdq=%d\n", nm, has_cdq);

  // packable test groups (m == 0): layer0 K/V, layer10 K (vpk=0 layer), layer14 K
  const int groups[] = {0, 1, 80, 112};
  const int NROW = 4;
  std::vector<float> yin(NROW * 256), yout(NROW * 256);
  std::vector<int> grp(NROW), pos(NROW, 100);
  unsigned seed = 12345;
  auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (int)(seed >> 8) - 8388608; };
  for (int r = 0; r < NROW; ++r) {
    grp[r] = groups[r];
    for (int d = 0; d < 256; ++d) yin[r * 256 + d] = (float)rnd() / 8388608.0f * 2.0f;
  }

  float *d_yin, *d_yout, *d_scl0, *d_scl1; unsigned char *d_pay0, *d_pay1, *d_scr;
  int *d_grp, *d_pos; unsigned char* d_blob;
  HIPCHK(hipMalloc(&d_yin, NROW * 256 * 4)); HIPCHK(hipMalloc(&d_yout, NROW * 256 * 4));
  HIPCHK(hipMalloc(&d_pay0, NROW * 160)); HIPCHK(hipMalloc(&d_pay1, NROW * 160));
  HIPCHK(hipMalloc(&d_scl0, NROW * 4)); HIPCHK(hipMalloc(&d_scl1, NROW * 4));
  HIPCHK(hipMalloc(&d_scr, tbf_scratch_bytes()));
  HIPCHK(hipMalloc(&d_grp, NROW * 4)); HIPCHK(hipMalloc(&d_pos, NROW * 4));
  HIPCHK(hipMalloc(&d_blob, n));
  HIPCHK(hipMemcpy(d_yin, yin.data(), NROW * 256 * 4, hipMemcpyHostToDevice));
  HIPCHK(hipMemcpy(d_grp, grp.data(), NROW * 4, hipMemcpyHostToDevice));
  HIPCHK(hipMemcpy(d_pos, pos.data(), NROW * 4, hipMemcpyHostToDevice));
  HIPCHK(hipMemcpy(d_blob, blob.data(), n, hipMemcpyHostToDevice));
  HIPCHK(hipMemset(d_pay0, 0, NROW * 160)); HIPCHK(hipMemset(d_pay1, 0, NROW * 160));

  TbFastArgs A{};
  A.blob = d_blob; A.nm = nm; A.has_cdq = has_cdq;
  A.max_ctx = 4096; A.d_pos = nullptr; A.layer = 0; A.tok0 = 0;
  A.yin = d_yin; A.grp_l = d_grp; A.pos_l = d_pos; A.yout = d_yout; A.row0 = 0; A.nrow = NROW;
  A.scr = d_scr;
  A.pk_pay = d_pay0; A.pk_pay1 = d_pay1; A.pk_stride = 160;
  A.pk_scl0 = d_scl0; A.pk_scl1 = d_scl1;
  A.stage = 3;
  launch_tb_fast(A, nullptr);
  HIPCHK(hipGetLastError()); HIPCHK(hipDeviceSynchronize());

  std::vector<unsigned char> pay0(NROW * 160), pay1(NROW * 160);
  std::vector<float> scl0(NROW), scl1(NROW);
  HIPCHK(hipMemcpy(pay0.data(), d_pay0, NROW * 160, hipMemcpyDeviceToHost));
  HIPCHK(hipMemcpy(pay1.data(), d_pay1, NROW * 160, hipMemcpyDeviceToHost));
  HIPCHK(hipMemcpy(scl0.data(), d_scl0, NROW * 4, hipMemcpyDeviceToHost));
  HIPCHK(hipMemcpy(scl1.data(), d_scl1, NROW * 4, hipMemcpyDeviceToHost));
  HIPCHK(hipMemcpy(yout.data(), d_yout, NROW * 256 * 4, hipMemcpyDeviceToHost));

  FILE* o = std::fopen(argv[2], "wb");
  if (!o) return 1;
  const int hdr[4] = {NROW, nm, has_cdq, 160};
  std::fwrite(hdr, 4, 4, o);
  for (int r = 0; r < NROW; ++r) {
    std::fwrite(&grp[r], 4, 1, o);
    std::fwrite(&yin[r * 256], 4, 256, o);
    // yin mode: tbf_row sets kind = 0 for ALL rows, so the kernel writes every
    // row to pk_pay/pk_scl0 regardless of the group's K/V kind.
    const unsigned char* pay = pay0.data() + r * 160;
    const float scl = scl0[r];
    std::fwrite(pay, 1, 160, o);
    std::fwrite(&scl, 4, 1, o);
    std::fwrite(&yout[r * 256], 4, 256, o);
  }
  std::fclose(o);
  printf("wrote %s (nrow=%d; YSKN_TB_VIT2=%s)\n", argv[2], NROW,
         std::getenv("YSKN_TB_VIT2") ? std::getenv("YSKN_TB_VIT2") : "1(default)");
  return 0;
}
