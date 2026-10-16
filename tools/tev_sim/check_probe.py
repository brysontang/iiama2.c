#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
What did the GPU probes (source/gx_probe.c) get that the simulator didn't expect?

    uv run tools/tev_sim/check_probe.py models/stories15M.bin SD:/llama/probe.bin

Every probe's framebuffer copy is compared byte for byte with what Dolphin's
TEV math (tev.py) says it should be, and the differences are summarized so the
pattern shows: which texel got sampled, which konst channel came back, whether
out-of-range writes wrap or saturate, and so on.

The first real-Wii run showed PREV isn't reset per pixel (it keeps what an
earlier pixel left), so no probe or matmul reads PREV before writing it.
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cost_model import region
from export_wii import quantize_q8
from gxmatmul import BlendMatrix, quantize_x
from llama import Weights
from tev import (
    BIAS_ADDHALF,
    BIAS_SUBHALF,
    BIAS_ZERO,
    HALF,
    K_1_8,
    OP_ADD,
    OP_SUB,
    SCALE_1,
    SCALE_DIVIDE2,
    color_stage,
    to_efb,
)

# must match gx_probe.c
PREV_VALUES = [(77, 150, 200), (-50, 300, 1000), (-1, 256, 511), (0, 255, -1024),
               (1023, -129, 128), (-200, 600, 384), (129, 383, -256), (640, -640, 3)]
KONST_VALUES = [0, 1, 2, 64, 127, 128, 129, 200, 254, 255]
BLEND_VALUES = [((10, 100, 200), (20, 100, 100), (5, 250, 0)), ((255, 0, 128), (1, 0, 127), (0, 1, 255)),
                ((128, 64, 32), (127, 191, 223), (200, 100, 50)), ((0, 0, 0), (0, 255, 1), (1, 255, 0)),
                ((200, 50, 250), (100, 50, 10), (150, 49, 255)), ((77, 33, 99), (88, 44, 11), (66, 22, 0)),
                ((255, 255, 255), (255, 255, 255), (255, 0, 1)), ((1, 2, 3), (4, 5, 6), (7, 8, 9))]


def read_probe(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"IIPROBE1", "not a probe.bin"
    out, i = {}, 8
    while i < len(data):
        name = data[i:i + 8].rstrip(b"\0").decode()
        w, h = struct.unpack(">II", data[i + 8:i + 16])
        raw = np.frombuffer(data, np.uint8, w * h * 4, i + 16)
        out[name] = untile(raw, w, h)
        i += 16 + w * h * 4
    return out


def untile(raw, w, h):
    """RGBA8 texture tiles (4x4 pixels: 16 AR pairs, then 16 GB pairs) -> (h, w, 3) RGB."""
    t = raw.reshape(h // 4, w // 4, 2, 4, 4, 2)  # tile row, tile col, AR/GB, y, x, pair
    r = t[:, :, 0, :, :, 1]
    g = t[:, :, 1, :, :, 0]
    b = t[:, :, 1, :, :, 1]
    rgb = np.stack([r, g, b], -1)  # (th, tw, 4, 4, 3)
    return rgb.transpose(0, 2, 1, 3, 4).reshape(h, w, 3).astype(np.int32)


def matmul_efb(w, x, mode):
    """The framebuffer bytes gx_matmul.c's first batch should produce, (ph * rows_of_slots, cw, 3)."""
    rows, cols = w.shape
    q, _, _ = quantize_q8(w, 32)
    wu = np.full((-(-rows // 3) * 3, cols), 128, np.int32)
    wu[:rows] = q.reshape(rows, cols).astype(np.int32) + 128
    pw, ph = region(rows)
    npx = -(-rows // 3)
    img = np.full((ph * pw, 3, cols), 128, np.int32)
    img[:npx] = wu.reshape(npx, 3, cols)
    img = img.reshape(ph, pw, 3, cols)
    pc = 16 if mode == "c8" else 8
    passes = cols // pc
    xs = x[:cols].reshape(passes, pc)
    c, _, _ = quantize_x(xs, 0.5 if mode == "c8" else 1.0)
    op = np.where(xs < 0, OP_SUB, OP_ADD)
    bias = np.where(xs < 0, BIAS_ADDHALF, BIAS_SUBHALF)
    draws = 1 if mode == "c8" else 2
    slots_x = 640 // pw
    slots = slots_x * (480 // ph)
    used = min(passes, slots // draws) * draws
    cw, ch = min(used, slots_x) * pw, -(-used // slots_x) * ph
    efb = np.full((ch, cw, 3), -1, np.int32)  # -1: not drawn
    for p in range(used // draws):
        v = np.full((ph, pw, 3), 128 if mode == "c8" else 0, np.int32)
        for k in range(pc):
            v = color_stage(HALF, img[:, :, :, p * pc + k], c[p, k], v, op[p, k], bias[p, k], SCALE_1,
                            clamp=mode == "c8" and k == pc - 1)
        outs = [to_efb(v)]
        if mode == "e11":
            hh = color_stage(0, 0, 0, v, OP_ADD, BIAS_ADDHALF, SCALE_DIVIDE2)
            hh = color_stage(0, 0, 0, hh, OP_ADD, BIAS_ADDHALF, SCALE_DIVIDE2)
            hh = color_stage(K_1_8, 0, 0, hh, OP_ADD, BIAS_ADDHALF, SCALE_DIVIDE2, clamp=True)
            outs.append(to_efb(hh))
        for d, o in enumerate(outs):
            s = p * draws + d
            y0, x0 = (s // slots_x) * ph, (s % slots_x) * pw
            efb[y0:y0 + ph, x0:x0 + pw] = o
    return efb, pw, ph, slots_x


def expected_tex():
    e = np.zeros((16, 128, 3), np.int32)
    for j in range(8):
        for py in range(16):
            for px in range(16):
                e[py, j * 16 + px] = (py * 16 + px, j * 32 + px, j * 32 + py)
    return e


def expected_konst():
    e = np.zeros((32, 64, 3), np.int32)
    for r in range(8):
        for k in range(16):
            e[r * 4:r * 4 + 4, k * 4:k * 4 + 4] = (r * 37 + (k >> 2) * 16 + (k & 3) * 4 + 1) & 255
    return e


def expected_reg():
    e = np.zeros((8, 4 * len(PREV_VALUES), 3), np.int32)
    for i, v in enumerate(PREV_VALUES):
        for clamp in (0, 1):
            e[clamp * 4:clamp * 4 + 4, i * 4:i * 4 + 4] = to_efb(
                color_stage(0, 0, 0, np.array(v), OP_ADD, BIAS_ZERO, SCALE_1, clamp=bool(clamp)))
    return e


def expected_arith():
    v = np.arange(256).reshape(16, 16)
    tex = np.stack([v, 255 - v, (v * 7) & 255], -1)
    e = np.zeros((16, 320, 3), np.int32)
    for sub in (0, 1):
        for i, k in enumerate(KONST_VALUES):
            e[:, (sub * 10 + i) * 16:(sub * 10 + i + 1) * 16] = color_stage(
                HALF, tex, k, 128, OP_SUB if sub else OP_ADD, BIAS_ADDHALF if sub else BIAS_SUBHALF, SCALE_1, clamp=True)
    return e


def expected_div():
    e = np.zeros((4, 4 * len(PREV_VALUES), 3), np.int32)
    for i, v in enumerate(PREV_VALUES):
        h = color_stage(0, 0, 0, np.array(v), OP_ADD, BIAS_ADDHALF, SCALE_DIVIDE2)
        h = color_stage(0, 0, 0, h, OP_ADD, BIAS_ADDHALF, SCALE_DIVIDE2)
        h = color_stage(K_1_8, 0, 0, h, OP_ADD, BIAS_ADDHALF, SCALE_DIVIDE2, clamp=True)
        e[:, i * 4:i * 4 + 4] = to_efb(h)
    return e


def expected_blend():
    """Row 0: D, then + S, then - T. Row 1: D, then - S, then + T. Saturating after each step."""
    e = np.zeros((8, 4 * len(BLEND_VALUES), 3), np.int32)
    for i, (d, s_, t) in enumerate(BLEND_VALUES):
        d, s_, t = np.array(d), np.array(s_), np.array(t)
        e[0:4, i * 4:i * 4 + 4] = np.maximum(np.minimum(d + s_, 255) - t, 0)
        e[4:8, i * 4:i * 4 + 4] = np.minimum(np.maximum(d - s_, 0) + t, 255)
    return e


def blend_efb(w, x):
    """The blended classifier's framebuffer: one byte per row, 3 rows per pixel."""
    m = BlendMatrix(w)
    m.tev_sums(x[: m.n])
    rows = w.shape[0]
    pw, ph = region(rows)
    acc = np.full(pw * ph * 3, -1, np.int32)
    acc[:rows] = m.last_acc
    return acc.reshape(ph, pw, 3)


def compare(name, got, exp, show=8):
    mask = exp >= 0
    bad = (got != exp) & mask
    n = int(bad.sum())
    print(f"{name:8s} {'OK' if n == 0 else 'DIFFERENT'}: {n} of {int(mask.sum())} bytes differ")
    if n:
        ys, xs, cs = np.nonzero(bad)
        for y, x, c in list(zip(ys, xs, cs))[:show]:
            print(f"           at x={x} y={y} {'RGB'[c]}: got {got[y, x, c]}, expected {exp[y, x, c]}")
        d = (got - exp)[bad]
        vals, counts = np.unique(d, return_counts=True)
        top = sorted(zip(counts, vals), reverse=True)[:8]
        print("           got - expected, most common: " + ", ".join(f"{v:+d} x{c}" for c, v in top))
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("probe_bin")
    args = ap.parse_args()

    got = read_probe(args.probe_bin)
    simple = {"tex": expected_tex, "konst": expected_konst, "reg": expected_reg, "arith": expected_arith,
              "div": expected_div, "blend": expected_blend}
    for name, fn in simple.items():
        if name in got:
            compare(name, got[name], fn())

    W = Weights(args.checkpoint)
    n = max(W.dim, W.hidden)
    x = ((((np.arange(n) * 37) % 101) - 50) / 16.0).astype(np.float32)
    if "cls_bl" in got:
        exp = blend_efb(W.f32["cls"], x)
        if got["cls_bl"].shape != exp.shape:
            print(f"cls_bl   copy is {got['cls_bl'].shape[1]}x{got['cls_bl'].shape[0]}, expected {exp.shape[1]}x{exp.shape[0]}")
        else:
            compare("cls_bl", got["cls_bl"], exp, show=4)
    for name, key, mode in (("w1_c8", ("w1", 0), "c8"), ("cls_c8", "cls", "c8"), ("w1_e11", ("w1", 0), "e11")):
        if name not in got:
            continue
        exp, pw, ph, slots_x = matmul_efb(W.f32[key], x, mode)
        g = got[name]
        if g.shape != exp.shape:
            print(f"{name:8s} copy is {g.shape[1]}x{g.shape[0]}, expected {exp.shape[1]}x{exp.shape[0]}")
            continue
        if compare(name, g, exp, show=4):
            # which slots (draws) are wrong, and by how much
            per = []
            for s in range((exp >= 0).any(-1).sum() // (pw * ph)):
                y0, x0 = (s // slots_x) * ph, (s % slots_x) * pw
                e, gg = exp[y0:y0 + ph, x0:x0 + pw], g[y0:y0 + ph, x0:x0 + pw]
                per.append(f"{s}:{int((e != gg).sum())}")
            print("           wrong bytes per slot: " + " ".join(per))


if __name__ == "__main__":
    main()
