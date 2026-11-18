#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU reference for the trellis kk=5 packed-row decode (pk3 path).

Three independent formulations of the same mechanism must agree:
  A) per-value 8-bit window extraction (pk_state_g math, MSB-first bit numbering)
  B) pk_conv16_k5 chunk/half window math (the task's CPU reference formula)
  C) sequential state recurrence s[e] = ((s[e-1]&7)<<5) | field5[e]
Plus an encoder-side packer (tbf_sel bit layout) roundtrip, and a KVQP blob
codebook parse check against the real model file.

Pure Python ints (no numpy uint16 overflow). Run: python3 test_decode_ref.py [kvq_u5.bin]
"""
import random
import struct
import sys
from pathlib import Path

MASK32 = 0xFFFFFFFF


def funnelshift_r(lo, hi, s):
    return (((hi << 32) | lo) >> s) & MASK32


def decode_a(words, tab):
    """A: per-value 8-bit window (pk_state_g math). The trellis state RESETS
    every 64 values: each 10-word chunk starts with initial state 0, so the
    top 3 bits of s[64k] are zero (encoder: tbf_chain runs 4 x 64 steps)."""
    out = []
    for e in range(256):
        base = (e >> 6) * 10
        em1 = 5 * (e & 63) + 4         # last bit index of the window, chunk-local
        j, r = em1 >> 5, 31 - (em1 & 31)
        b = words[base + j]
        a = words[base + j - 1] if j else 0
        s = funnelshift_r(b, a, r) & 255
        out.append(tab[s])
    return out


def decode_b(words, tab):
    """B: pk_conv16_k5<HALF>(rw, chp, tab) exactly as specified."""
    out = [None] * 256
    for chp in range(8):
        for half in (0, 1):
            wv = [0] * 6
            if half == 0:
                wv[0] = words[5 * chp - 1] if (chp & 1) else 0
                wv[1], wv[2], wv[3] = words[5 * chp], words[5 * chp + 1], words[5 * chp + 2]
            else:
                wv[2], wv[3] = words[5 * chp + 1], words[5 * chp + 2]
                wv[4], wv[5] = words[5 * chp + 3], words[5 * chp + 4]
            for i in range(16):
                em1 = 31 + 5 * (16 * half + i + 1)
                j = em1 >> 5
                s = funnelshift_r(wv[j], wv[j - 1], 31 - (em1 & 31)) & 255
                out[32 * chp + 16 * half + i] = tab[s]
    return out


def decode_c(words, tab):
    """C: sequential recurrence over 5-bit fields, state reset every 64 values."""
    s, out = 0, []
    for e in range(256):
        if e % 64 == 0:
            s = 0
        base = (e >> 6) * 10
        em1 = 5 * (e & 63) + 4         # last bit of the 5-bit field, chunk-local
        j, r = em1 >> 5, 31 - (em1 & 31)
        b = words[base + j]
        a = words[base + j - 1] if j else 0
        f = funnelshift_r(b, a, r) & 31
        s = ((s & 7) << 5) | f
        out.append(tab[s])
    return out


def pack_fields(fields):
    """Encoder bit layout (tbf_sel_kernel): global bit g=5e+jb carries
    bit (4-jb) of field[e], MSB-first across the 40-word row."""
    words = [0] * 40
    for g in range(1280):
        e, jb = divmod(g, 5)
        bit = (fields[e] >> (4 - jb)) & 1
        words[g >> 5] |= bit << (31 - (g & 31))
    return words


def symbols_from_fields(fields):
    s, out = 0, []
    for e, f in enumerate(fields):
        if e % 64 == 0:
            s = 0
        s = ((s & 7) << 5) | f
        out.append(s)
    return out


def main():
    rng = random.Random(12345)
    tab = [rng.randrange(65536) for _ in range(256)]  # arbitrary bf16 bit patterns

    # 1) three formulations agree on arbitrary random rows (not only packable ones)
    for t in range(200):
        words = [rng.randrange(1 << 32) for _ in range(40)]
        a, b, c = decode_a(words, tab), decode_b(words, tab), decode_c(words, tab)
        assert a == b == c, f"formulation mismatch at random row {t}"
    print("PASS: 200 random rows, formulations A/B/C identical (256 values each)")

    # 2) encoder pack -> decode roundtrip reproduces the trellis symbols
    for t in range(200):
        fields = [rng.randrange(32) for _ in range(256)]
        words = pack_fields(fields)
        want = [tab[s] for s in symbols_from_fields(fields)]
        assert decode_a(words, tab) == want, f"roundtrip mismatch at {t}"
        assert decode_b(words, tab) == want
        assert decode_c(words, tab) == want
    print("PASS: 200 pack/decode roundtrips match trellis state recurrence")

    # 3) initial state: first symbol's top 3 bits are zero regardless of word0 high bits
    words = [0xFFFFFFFF] * 40
    s0 = funnelshift_r(words[0], 0, 31 - 4) & 255
    assert s0 == 31, s0  # bits 0..4 all ones, bits -3..-1 = 0
    print("PASS: initial-state zero padding (s[0] = field[0])")

    # 4) real model blob: KVQP header + codebook record build (kvqaxpk_build port).
    # Optional: pass the kvq_u5.bin path as argv[1] (no hardcoded local model paths).
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if model is not None and model.exists():
        b = model.read_bytes()
        u16 = lambda o: b[o] | (b[o + 1] << 8)
        u32 = lambda o: u16(o) | (u16(o + 2) << 16)
        ngrp, off_t, off_g = u32(28), u32(36), u32(40)
        assert ngrp == 128, ngrp
        kb_cnt = {4: 0, 5: 0}
        mu_cnt = 0
        m_cnt = {}
        packable_layers = 0
        for g in range(ngrp):
            go = u32(off_g + 4 * g)
            mode, kk, tix, guard, m, has_mu = b[go], b[go + 1], b[go + 2], b[go + 3], b[go + 4], b[go + 5]
            if mode != 0:
                continue
            kb_cnt[kk] = kb_cnt.get(kk, 0) + 1
            m_cnt[m] = m_cnt.get(m, 0) + 1
            mu_cnt += has_mu
            to = u32(off_t + 4 * tix)
            gain = struct.unpack_from("<f", b, to + 4)[0]
            cmax = 0.0
            for i in range(256):
                ci = struct.unpack_from("<h", b, to + 12 + 2 * i)[0]
                c = ci * 0.00390625 * gain
                cmax = max(cmax, abs(c))
            st = cmax / 127 if cmax > 0 else 1.0
            assert st > 0
        # kvqaxpk_packable: per layer/kind all 4 kv heads must be mode0/kk/guard0/m0
        for layer in range(16):
            ok = True
            for kind in range(2):
                for h in range(4):
                    g = (layer * 4 + h) * 2 + kind
                    go = u32(off_g + 4 * g)
                    if b[go] != 0 or b[go + 1] != 5 or b[go + 3] != 0 or b[go + 4] != 0:
                        ok = False
            packable_layers += ok
        print(f"PASS: blob parse ngrp={ngrp} kk={kb_cnt} m_hist={m_cnt} mu={mu_cnt} packable_layers={packable_layers}/16")
        assert kb_cnt.get(5, 0) == 128, "model must be all kk=5"
        assert packable_layers >= 1, "some layers must be pk-packable for the pk3 path"
        # non-packable layers (m>0 projection groups) use the engine's int8 atx path instead
    else:
        print("SKIP: model blob check (pass kvq_u5.bin as argv[1] to enable)")

    print("ALL DECODE REFERENCE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
