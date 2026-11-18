#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU verification of the native pk3 attention debug dump.

Reads the binary dump produced by YSKN_ATX_NATIVE_DUMP (WG(0,0,0), tile 0)
and recomputes the chain on the CPU from the dumped packed inputs:
  stage 1: trellis decode (3-formulation-verified reference) vs dumped K/V tiles
  stage 2: QK dot products + scale + causal mask vs dumped scores
  stage 3: online-softmax max/sum after tile 0 vs dumped m/l
  stage 4: sanity of final pm/pl/pacc (finite, pl>=0)
  stage 5: self-row decode + self dot + softmax update (pm/pl when ntiles<=1)
  stage 6: PV + self + mu -> pacc, full chain (only when ntiles==1)
Usage: python3 analyze_dump.py <dump.bin>
"""
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_decode_ref import decode_a  # noqa: E402

KT, HD, L2E = 16, 256, 1.4426950408889634
# layout must match DBG_* in q35_atxpk_native.hip
O_KT, O_VT = 0, KT * HD
O_SC = 2 * KT * HD
O_ML = O_SC + 16 * KT
O_PM = O_ML + 32
O_PL = O_PM + 16
O_PA = O_PL + 16
O_PK = O_PA + 16 * HD
O_TAB = O_PK + 2 * KT * 40
O_Q = O_TAB + 512
O_GEO = O_Q + 16 * HD
O_MU = O_GEO + 102
O_SELF = O_MU + 256
O_KOWN = O_SELF + 52
O_VOWN = O_KOWN + 512
O_PSELF = O_VOWN + 512
O_T0 = O_PSELF + 160
TS, MAXT = 1616, 4
O_V8 = O_T0 + TS * MAXT
WORDS = O_V8 + 4 * 1024 + 128


def v8_row(words64, d):
    b = (words64[d // 4] >> (8 * (d % 4))) & 255
    return bf2f(f2bf(float(b - 256 if b > 127 else b)))


def f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def bits2f(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def f2bf(f):
    u = f32bits(f)
    u += 0x7FFF + ((u >> 16) & 1)
    return (u >> 16) & 0xFFFF


def bf2f(h):
    return bits2f(h << 16)


def relerr(got, ref, floor=1.0):
    if ref == float("-inf"):
        return 0.0 if got == float("-inf") else 1e30
    if not math.isfinite(got):
        return 1e30
    return abs(got - ref) / max(floor, abs(ref))


def main():
    path = sys.argv[1]
    data = Path(path).read_bytes()
    assert len(data) == WORDS * 4, f"dump size {len(data)} != {WORDS * 4}"
    w = struct.unpack(f"<{WORDS}f", data)
    geo = w[O_GEO:O_GEO + 102]
    base, k0, klo, khi, ntiles = (int(geo[i]) for i in range(5))
    vpk = int(geo[101])
    lo = [int(geo[5 + i]) for i in range(16)]
    ph = [int(geo[21 + i]) for i in range(16)]
    vld = [int(geo[37 + i]) for i in range(16)]
    slf = [int(geo[53 + i]) for i in range(16)]
    ksc = list(geo[69:85])
    vsc = list(geo[85:101])
    print(f"dump: base={base} k0={k0} klo={klo} khi={khi} ntiles={ntiles} vpk={vpk} slf={slf[:12]}")
    fails = 0

    def report(name, mx, tol):
        nonlocal fails
        ok = mx <= tol
        print(f"  {name:<32} max_err={mx:.6g} {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails += 1

    tab = [f32bits(v) >> 16 for v in w[O_TAB:O_TAB + 512]]
    tabk, tabv = tab[:256], tab[256:]
    q = [list(w[O_Q + p * HD: O_Q + (p + 1) * HD]) for p in range(16)]
    kt = [list(w[O_KT + k * HD: O_KT + (k + 1) * HD]) for k in range(KT)]
    vt = [list(w[O_VT + k * HD: O_VT + (k + 1) * HD]) for k in range(KT)]
    mu = list(w[O_MU:O_MU + 256])

    if ntiles == 0:
        print("  (no tiles for this WG; stages 1-3,6 skipped)")
    else:
        pk = [f32bits(v) for v in w[O_PK:O_PK + 2 * KT * 40]]
        for name, off, tb, out_off in (("K", 0, tabk, O_KT), ("V", KT * 40, tabv, O_VT)):
            if name == "V" and not vpk:
                print("  stage1 decode V: skipped (V int8, exact conversion)")
                continue
            mx = 0.0
            for key in range(KT):
                if not (klo <= k0 + key < khi):
                    # out-of-range keys: kernel zero-fills the tile; the CPU
                    # reference would decode zero words to tab[0]. Both are
                    # masked out of the softmax, so only check the zero fill.
                    for d in range(HD):
                        if w[out_off + key * HD + d] != 0.0:
                            mx = 1e30
                    continue
                words = pk[off + key * 40: off + (key + 1) * 40]
                ref = decode_a(words, tb)
                for d in range(HD):
                    got = w[out_off + key * HD + d]
                    want = bf2f(ref[d])
                    if got != want:
                        mx = max(mx, abs(got - want) if math.isfinite(got) else 1e30)
            report(f"stage1 decode {name} (exact)", mx, 0.0)
        mx = 0.0
        for p in range(16):
            for k in range(KT):
                kidx = k0 + k
                s = math.fsum(q[p][d] * kt[k][d] for d in range(HD)) * (ksc[k] * 0.0625)
                if not (lo[p] <= kidx < ph[p]):
                    s = float("-inf")
                mx = max(mx, relerr(w[O_SC + p * KT + k], s))
        report("stage2 QK+scale+mask (rel)", mx, 2e-3)
        mxm = mxl = 0.0
        for p in range(16):
            ss = [w[O_SC + p * KT + k] for k in range(KT)]
            m = max([-1e30] + ss)
            l = math.fsum(0.0 if s == float("-inf") else 2.0 ** ((s - m) * L2E) for s in ss)
            mxm = max(mxm, relerr(w[O_ML + p], m))
            mxl = max(mxl, relerr(w[O_ML + 16 + p], l))
        report("stage3 softmax m (rel)", mxm, 1e-4)
        report("stage3 softmax l (rel)", mxl, 1e-4)

    # stage 4: final outputs sanity
    bad = 0
    for p in range(16):
        if not vld[p]:
            continue
        pm, pl = w[O_PM + p], w[O_PL + p]
        if not (math.isfinite(pm) or pm == -1e30) or pl < 0 or not math.isfinite(pl):
            bad += 1
        for d in range(HD):
            if not math.isfinite(w[O_PA + p * HD + d]):
                bad += 1
    report("stage4 pm/pl/pacc finite", float(bad), 0.0)

    # stage 5: self epilogue
    ksc_own = list(w[O_SELF + 48:O_SELF + 50])
    vsc_own = list(w[O_SELF + 50:O_SELF + 52])
    pself = [f32bits(v) for v in w[O_PSELF:O_PSELF + 160]]
    kown = [list(w[O_KOWN + t * 256: O_KOWN + (t + 1) * 256]) for t in range(2)]
    vown = [list(w[O_VOWN + t * 256: O_VOWN + (t + 1) * 256]) for t in range(2)]
    mxd = mxs = mxu = 0.0
    self_ref = {}
    for p in range(16):
        if not (vld[p] and slf[p]):
            continue
        tok = p // 6
        for name, tb, out, off in (("K", tabk, kown, tok * 80), ("V", tabv, vown, tok * 80 + 40)):
            if name == "V" and not vpk:
                v8w = [f32bits(v) for v in w[O_V8 + 4 * 1024 + tok * 64: O_V8 + 4 * 1024 + (tok + 1) * 64]]
                for d in range(HD):
                    if out[tok][d] != v8_row(v8w, d):
                        mxd = 1e30
                continue
            ref = decode_a(pself[off:off + 40], tb)
            for d in range(HD):
                if out[tok][d] != bf2f(ref[d]):
                    mxd = 1e30
        sc = math.fsum(q[p][d] * kown[tok][d] for d in range(HD)) * (ksc_own[tok] * 0.0625)
        mxs = max(mxs, relerr(w[O_SELF + p], sc))
        m_before = w[O_ML + p] if ntiles >= 1 else -1e30
        l_before = w[O_ML + 16 + p] if ntiles >= 1 else 0.0
        mn = max(m_before, sc)
        cf = 2.0 ** ((m_before - mn) * L2E)
        ps = 2.0 ** ((sc - mn) * L2E)
        self_ref[p] = (mn, cf, ps, l_before * cf + ps)
        if ntiles <= 1:
            mxu = max(mxu, relerr(w[O_SELF + 16 + p], cf, 1e-3))
            mxu = max(mxu, relerr(w[O_SELF + 32 + p], ps, 1e-3))
            mxu = max(mxu, relerr(w[O_PM + p], mn))
            mxu = max(mxu, relerr(w[O_PL + p], l_before * cf + ps))
    if any(vld[p] and slf[p] for p in range(16)):
        report("stage5 self decode (exact)", mxd, 0.0)
        report("stage5 self dot (rel)", mxs, 2e-3)
        if ntiles <= 1:
            report("stage5 self update pm/pl (rel)", mxu, 1e-3)
    else:
        print("  (no self pairs in this WG; stage 5 skipped)")

    # stage 6: full chain PV + self + mu -> pacc (single-tile WGs only)
    if ntiles == 1:
        mx = 0.0
        for p in range(16):
            if not vld[p]:
                continue
            m0 = w[O_ML + p]
            pv = []
            for k in range(KT):
                s = w[O_SC + p * KT + k]
                pv.append(0.0 if s == float("-inf") else bf2f(f2bf(2.0 ** ((s - m0) * L2E) * vsc[k])))
            mn, cf, ps, pl = self_ref.get(p, (m0, 1.0, 0.0, w[O_ML + 16 + p]))
            tok = p // 6
            for d in range(HD):
                o = math.fsum(pv[k] * vt[k][d] for k in range(KT))
                if p in self_ref:
                    o = o * cf + ps * vsc_own[tok] * vown[tok][d]
                o += w[O_PL + p] * (mu[d] if vpk else 0.0)
                mx = max(mx, relerr(w[O_PA + p * HD + d], o, 1e-3))
        report("stage6 PV+self+mu pacc (rel)", mx, 2e-3)

    # stage 7: complete multi-tile chain recomputed from per-tile packed inputs
    if 1 <= ntiles <= MAXT:
        m = [-1e30] * 16
        l = [0.0] * 16
        o = [[0.0] * HD for _ in range(16)]
        mx_ml = mx_cf = 0.0
        for t in range(ntiles):
            T = w[O_T0 + t * TS: O_T0 + (t + 1) * TS]
            pkt = [f32bits(v) for v in T[0:1280]]
            ksc_t, vsc_t = list(T[1584:1600]), list(T[1600:1616])
            k0t = k0 + KT * t
            kt_t, vt_t = [], []
            for key in range(KT):
                kidx = k0t + key
                if klo <= kidx < khi:
                    kt_t.append([bf2f(h) for h in decode_a(pkt[key * 40:(key + 1) * 40], tabk)])
                    if vpk:
                        vt_t.append([bf2f(h) for h in decode_a(pkt[640 + key * 40:640 + (key + 1) * 40], tabv)])
                    else:
                        v8w = [f32bits(v) for v in w[O_V8 + t * 1024 + key * 64: O_V8 + t * 1024 + (key + 1) * 64]]
                        vt_t.append([v8_row(v8w, d) for d in range(HD)])
                else:
                    kt_t.append([0.0] * HD)
                    vt_t.append([0.0] * HD)
            for p in range(16):
                ss = []
                for k in range(KT):
                    kidx = k0t + k
                    s = math.fsum(q[p][d] * kt_t[k][d] for d in range(HD)) * (ksc_t[k] * 0.0625)
                    if not (lo[p] <= kidx < ph[p]):
                        s = float("-inf")
                    ss.append(s)
                mn = max(m[p], max(ss))
                cf = 2.0 ** ((m[p] - mn) * L2E)
                pvs = [0.0 if s == float("-inf") else 2.0 ** ((s - mn) * L2E) for s in ss]
                l[p] = l[p] * cf + math.fsum(pvs)
                m[p] = mn
                mx_ml = max(mx_ml, relerr(T[1536 + p], mn), relerr(T[1536 + 16 + p], l[p]))
                mx_cf = max(mx_cf, relerr(T[1568 + p], cf, 1e-3))
                Pb = [bf2f(f2bf(pvs[k] * vsc_t[k])) for k in range(KT)]
                for d in range(HD):
                    o[p][d] = o[p][d] * cf + math.fsum(Pb[k] * vt_t[k][d] for k in range(KT))
        for p in range(16):
            if not (vld[p] and slf[p]):
                continue
            tok = p // 6
            sc = math.fsum(q[p][d] * kown[tok][d] for d in range(HD)) * (ksc_own[tok] * 0.0625)
            mn = max(m[p], sc)
            cf = 2.0 ** ((m[p] - mn) * L2E)
            ps = 2.0 ** ((sc - mn) * L2E)
            l[p] = l[p] * cf + ps
            m[p] = mn
            for d in range(HD):
                o[p][d] = o[p][d] * cf + ps * vsc_own[tok] * vown[tok][d]
        mx_pa = mx_pm = mx_pl = 0.0
        for p in range(16):
            if not vld[p]:
                continue
            mx_pm = max(mx_pm, relerr(w[O_PM + p], m[p]))
            mx_pl = max(mx_pl, relerr(w[O_PL + p], l[p]))
            for d in range(HD):
                ref = o[p][d] + l[p] * (mu[d] if vpk else 0.0)
                mx_pa = max(mx_pa, relerr(w[O_PA + p * HD + d], ref, 1e-3))
        report(f"stage7 {ntiles}-tile m/l (rel)", mx_ml, 1e-4)
        report("stage7 cf (rel)", mx_cf, 1e-3)
        report("stage7 full-chain pm (rel)", mx_pm, 1e-3)
        report("stage7 full-chain pl (rel)", mx_pl, 1e-3)
        report("stage7 full-chain pacc (rel)", mx_pa, 2e-3)
    print("ANALYZE FAILED" if fails else "ANALYZE ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
