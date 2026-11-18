#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU Viterbi reference for the trellis requant (encode) side.

Verifies the probe_requant.hip output (engine launch_tb_fast on known rows)
against a from-scratch port of the encode MECHANISM (tbf_pre / tbf_chain /
tbf_sel math in src/kvq_fast.cuh), including the GPU reduction order.

Metrics per row:
  - packed code agreement (field symbols), scale exactness
  - reconstruction MSE of GPU yout vs CPU-reference yout against the input y:
    MSE_gpu >> MSE_cpu  -> encode picks suboptimal paths (translation bug)
    MSE_gpu ~= MSE_cpu  -> encode is optimal; codes may differ only by ties
Usage: python3 test_requant_ref.py <kvq_u5.bin> <probe_out.bin>
"""
import struct
import sys
import numpy as np

F32 = np.float32
BIG = 0x40000000


def gpu_sum(x):
    """tbf_sum order: per-32-lane shfl_xor tree (16,8,4,2,1), then sequential
    sum of the wave partials."""
    nw = x.size // 32
    a = x.reshape(nw, 32).astype(F32).copy()
    h = 16
    while h:
        a = (a[:, :h] + a[:, h:2 * h]).astype(F32)
        h //= 2
    t = F32(0)
    for v in a[:, 0]:
        t = F32(t + v)
    return t


def f16r(v):
    return F32(np.float16(v))


def viterbi_chain(xi, ci, mode="vit2"):
    """64-step Viterbi over 256 symbols (state = sym>>5, branch = sym&31).
    Transition: predecessor symbols p with p&7 == state.
    mode='vit2': tbf2_work math — cost packed as KQ*(c-X)^2 + mk[state] + (p&31)
    (int32, +jj breaks ties toward the smallest branch), mk offset by mk[0].
    mode='vit1': tbf_chain math — plain (xi-ci)^2 costs, ffs tie-break (smallest
    predecessor index). Both trace back with stout[t-1] = bp[t][stout[t]>>5]."""
    KQ = 32
    ar = np.arange(256, dtype=np.int32)
    bp = np.zeros((65, 8), dtype=np.int32)  # bp[t][g] = predecessor of step-t state-g winner
    with np.errstate(over="ignore"):
        if mode == "vit2":
            mk = np.full(8, KQ << 24, dtype=np.int32)
            mk[0] = 0
            for t in range(63):
                X = np.int32(xi[t])
                dx = ci.astype(np.int32) - X
                val = (np.int32(KQ) * dx * dx + mk[ar >> 5] + (ar & 31)).astype(np.int32)
                mk_new = np.empty(8, dtype=np.int32)
                for g in range(8):
                    idx = np.arange(g, 256, 8)
                    w = int(val[idx].argmin())
                    p = int(idx[w])
                    mk_new[g] = np.int32((int(val[p]) - int(mk[0])) & ~31)
                    bp[t + 1][g] = p
                mk = mk_new
            X = np.int32(xi[63])
            dx = ci.astype(np.int32) - X
            val = (np.int32(KQ) * dx * dx + mk[ar >> 5] + (ar & 31)).astype(np.int32)
            cost = val >> np.int32(5)  # arithmetic shift (GPU int semantics)
            cm = cost.min()
            fin = int(np.where(cost == cm)[0][0])
        else:
            cost = np.full(256, 0x40000000, dtype=np.int32)
            cost[0] = 0
            for t in range(64):
                c = cost.reshape(32, 8)
                mn = c.min(axis=0)
                jb = c.argmin(axis=0)
                for g in range(8):
                    bp[t][g] = 8 * int(jb[g]) + g
                dx = int(xi[t]) - ci.astype(np.int64)
                cost = (dx * dx).astype(np.int32) + mn[ar >> 5]
            fin = int(cost.argmin())
    st = fin
    stout = np.zeros(64, dtype=np.int32)
    for t in range(63, -1, -1):
        stout[t] = st
        if t > 0:
            st = int(bp[t][st >> 5])
    return stout


def encode_row(y, ci, gain, mvals, mud, mode="vit2"):
    """Full tbf_pre + scale search + 4-chain Viterbi + tbf_sel reconstruction."""
    r = (y - mud).astype(F32) if mud is not None else y.astype(F32).copy()
    ssq = gpu_sum((r * r).astype(F32))
    rms = F32(np.sqrt(F32(ssq / F32(256))))
    if rms == 0:
        rms = F32(1)
    cdec = ((ci.astype(F32) * F32(0.00390625)) * F32(gain)).astype(F32)
    best = None
    for mi, mv in enumerate(mvals):
        s = f16r(F32(rms * F32(mv)))
        if s == 0:
            s = F32(1)
        xi = np.rint((r / s).astype(F32) * F32(256)).astype(np.int64)
        xi = np.clip(xi, -2048, 2048)
        stouts, wos = [], []
        for c in range(4):
            stout = viterbi_chain(xi[c * 64:(c + 1) * 64], ci, mode)
            stouts.append(stout)
            rec = (cdec[stout] * s).astype(F32)
            dd = (r[c * 64:(c + 1) * 64] - rec).astype(F32)
            v = (dd * dd).astype(F32)
            # tbf_chain: two 32-lane wave sums -> wo[2c], wo[2c+1]
            a = v.reshape(2, 32).copy()
            h = 16
            while h:
                a = (a[:, :h] + a[:, h:2 * h]).astype(F32)
                h //= 2
            wos.extend([a[0, 0], a[1, 0]])
        e = F32(0)
        for v in wos:                    # tbf_sel: sequential over 8 partials
            e = F32(e + v)
        if best is None or e < best[0]:
            best = (e, mi, s, np.concatenate(stouts))
    e, mi, s, st = best
    xq = (cdec[st] * s).astype(F32)
    yout = xq.copy()
    if mud is not None:
        yout = (yout + mud).astype(F32)
    fields = (st & 31).astype(np.uint32)
    words = np.zeros(40, dtype=np.uint32)
    for g in range(1280):
        eidx, jb = divmod(g, 5)
        bit = (int(fields[eidx]) >> (4 - jb)) & 1
        words[g >> 5] |= np.uint32(bit << (31 - (g & 31)))
    return words, F32(s), yout, float(e), mi


def main():
    mode = sys.argv[3] if len(sys.argv) > 3 else "vit2"
    blob = open(sys.argv[1], "rb").read()
    u16 = lambda o: blob[o] | (blob[o + 1] << 8)
    u32 = lambda o: u16(o) | (u16(o + 2) << 16)
    nm = u16(24)
    off_m, off_t, off_g = u32(32), u32(36), u32(40)
    mvals = [struct.unpack_from("<f", blob, off_m + 4 * i)[0] for i in range(nm)]

    data = open(sys.argv[2], "rb").read()
    nrow, nm2, has_cdq, stride = struct.unpack_from("<4i", data, 0)
    assert nm2 == nm and stride == 160
    off = 16
    fails = 0
    for r in range(nrow):
        grp = struct.unpack_from("<i", data, off)[0]; off += 4
        y = np.array(struct.unpack_from("<256f", data, off), dtype=F32); off += 1024
        pay = np.array(struct.unpack_from("<40I", data, off), dtype=np.uint32); off += 160
        scl = struct.unpack_from("<f", data, off)[0]; off += 4
        yout = np.array(struct.unpack_from("<256f", data, off), dtype=F32); off += 1024

        go = u32(off_g + 4 * grp)
        mode, kk, tix, guard, m, has_mu = blob[go], blob[go + 1], blob[go + 2], blob[go + 3], blob[go + 4], blob[go + 5]
        assert mode == 0 and kk == 5 and m == 0, f"group {grp} not packable-simple"
        to = u32(off_t + 4 * tix)
        gain = struct.unpack_from("<f", blob, to + 4)[0]
        ci = np.array(struct.unpack_from("<256h", blob, to + 12), dtype=np.int64)
        mud = None
        if has_mu:
            mud = np.array([np.float16(struct.unpack_from("<H", blob, go + 8 + 2 * d)[0]) for d in range(256)], dtype=F32)

        words, s, yout_ref, e_ref, mi = encode_row(y, ci, gain, mvals, mud, mode)
        code_diff = int(np.sum(words != pay))
        scl_ok = struct.pack("<f", scl) == struct.pack("<f", float(s))
        mse_gpu = float(np.mean((yout - y) ** 2))
        mse_cpu = float(np.mean((yout_ref - y) ** 2))
        yout_diff = float(np.max(np.abs(yout - yout_ref)))
        ratio = mse_gpu / max(mse_cpu, 1e-30)
        ok = code_diff == 0 and scl_ok and ratio < 1.05
        print(f"row {r} grp {grp}: code_word_diffs={code_diff}/40 scale_match={scl_ok} "
              f"mi={mi} MSE_gpu={mse_gpu:.6g} MSE_cpu={mse_cpu:.6g} ratio={ratio:.4f} "
              f"yout_maxdiff={yout_diff:.6g} {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails += 1
            # symbol-level detail for the first failing row
            if code_diff:
                def fields_of(ws):
                    out = []
                    for e in range(256):
                        v = 0
                        for jb in range(5):
                            g = 5 * e + jb
                            v |= ((int(ws[g >> 5]) >> (31 - (g & 31))) & 1) << (4 - jb)
                        out.append(v)
                    return out
                gf, rf = fields_of(pay), fields_of(words)
                diffs = [(e, gf[e], rf[e]) for e in range(256) if gf[e] != rf[e]]
                print(f"  field diffs (e, gpu, cpu): {diffs[:12]}{'...' if len(diffs) > 12 else ''} total={len(diffs)}")
    print("REQUANT REF FAILED" if fails else "REQUANT REF ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
