"""
Matrix-vector products on the Wii's TEV, simulated stage by stage with tev.py.

Layout (one draw = one "pass" over `cols` input columns):

  * Each framebuffer pixel holds 3 output rows, one per RGB channel. Every
    channel does the same math, so the simulator just treats rows as rows.
  * Weights live in an RGBA8 texture as offset-binary bytes, Wu = q + 128,
    where q is the same int8 (per-row, per-group-of-32 scale) the CPU uses.
  * The activations for the pass go in the 4 konst colors: 16 bytes, one per
    stage, picked with KCSEL K0_R..K3_A. Every row multiplies the same x, so x
    never needs to be a texture. Each pass gets its own x scale (its max |x|).
  * Stage k:  a = HALF, b = TEXC (Wu), c = KONST (|x_k|), d = PREV (stage 0:
              the constant HALF or ZERO, since a real Wii doesn't reset PREV
              between pixels),
              op = ADD if x_k >= 0 else SUB, bias = SUBHALF / ADDHALF.
    lerp(128, Wu, c) = 128 + q*c/256, and the bias cancels the 128, so each
    stage adds exactly +/- round(q * |x_k| / 256) to PREV.
  * The CPU reads the framebuffer bytes back and does the cross-pass sum,
    applying the per-row weight scale and per-pass x scale in float.

Output modes, i.e. how each pass's 11-bit PREV gets out through 8-bit pixels:

  clamp8   16 columns/pass, the sum starts at 128, last stage clamps to [0, 255].
           Partial sums outside [-128, 127] saturate. 1 draw per 16 columns.
  exact11  8 columns/pass, 2 draws: one writes PREV's low byte (u8 wrap), one
           runs 3 extra halving stages and writes floor(PREV/8) + 128. The CPU
           rebuilds the exact 11-bit sum. 2 draws per 8 columns.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from export_wii import quantize_q8
from tev import (
    BIAS_ADDHALF,
    BIAS_SUBHALF,
    HALF,
    K_1_8,
    OP_ADD,
    OP_SUB,
    SCALE_1,
    SCALE_DIVIDE2,
    color_stage,
    to_efb,
)

MODES = {"clamp8": 16, "exact11": 8}


def quantize_x(xs, headroom):
    """Per-pass activation quantization. xs: (passes, cols). Returns konst bytes c, effective c', scale."""
    xmax = np.abs(xs).max(axis=1)
    xmax = np.where(xmax == 0, 1.0, xmax).astype(np.float32)
    t = np.abs(xs) / xmax[:, None] * (256.0 * headroom)
    cp = np.clip(np.rint(t), 0, 256).astype(np.int32)
    # c' = c + (c >> 7) can't produce 128, so snap to the nearer of 127 / 129
    cp = np.where(cp == 128, np.where(t >= 128, 129, 127), cp)
    c = cp - (cp >= 129)
    return c, cp, xmax / headroom


class TevMatrix:
    def __init__(self, w_f32, mode="clamp8", headroom=1.0, cols=None, group_size=32):
        self.mode, self.cols, self.headroom = mode, cols or MODES[mode], headroom
        assert self.cols <= MODES[mode], f"{mode} fits at most {MODES[mode]} columns per pass"
        rows, n = w_f32.shape
        self.n = n
        self.gs = group_size
        assert self.gs % self.cols == 0
        n_pad = -(-n // self.gs) * self.gs
        wp = np.zeros((rows, n_pad), np.float32)
        wp[:, :n] = w_f32
        q, s, _ = quantize_q8(wp, self.gs)
        self.q = q.reshape(rows, n_pad)
        self.s = s.reshape(rows, n_pad // self.gs)
        self.wu = (self.q.astype(np.int32) + 128).reshape(rows, -1, self.cols)  # texture bytes
        self.passes = n_pad // self.cols
        self.stats = {"calls": 0, "partials": 0, "saturated": 0}

    def tev_partials(self, x):
        """Run the TEV program for every pass. Returns (partial sums (rows, passes), x scale per pass)."""
        xp = np.zeros(self.passes * self.cols, np.float32)
        xp[: self.n] = x
        xs = xp.reshape(self.passes, self.cols)
        c, cp, xscale = quantize_x(xs, self.headroom)
        op = np.where(xs < 0, OP_SUB, OP_ADD)
        bias = np.where(xs < 0, BIAS_ADDHALF, BIAS_SUBHALF)
        rows = self.wu.shape[0]

        last = self.cols - 1
        clamp8 = self.mode == "clamp8"
        v = np.full((rows, self.passes), 128 if clamp8 else 0, np.int32)  # stage 0's d: HALF or ZERO
        for k in range(self.cols):
            v = color_stage(HALF, self.wu[:, :, k], c[:, k], v, op[:, k], bias[:, k], SCALE_1,
                            clamp=clamp8 and k == last)
        if clamp8:
            partial = to_efb(v) - 128
        else:
            lo = to_efb(v)  # draw 1 ends here
            h = color_stage(0, 0, 0, v, OP_ADD, BIAS_ADDHALF, SCALE_DIVIDE2)
            h = color_stage(0, 0, 0, h, OP_ADD, BIAS_ADDHALF, SCALE_DIVIDE2)
            h = color_stage(K_1_8, 0, 0, h, OP_ADD, BIAS_ADDHALF, SCALE_DIVIDE2, clamp=True)
            hi = to_efb(h)  # draw 2
            partial = 8 * (hi - 128) + (lo & 7)

        # exact integer result the hardware would give with unlimited range, to count saturation
        sign = np.where(xs < 0, -1, 1)
        q3 = self.wu - 128
        prod = q3 * cp[None]
        exact = np.where(sign[None] > 0, (prod + 128) >> 8, -((prod + 127) >> 8)).sum(axis=2)
        self.stats["calls"] += 1
        self.stats["partials"] += partial.size
        self.stats["saturated"] += int((partial != exact).sum())
        return partial, xscale

    def __call__(self, x):
        partial, xscale = self.tev_partials(x)
        per_group = self.gs // self.cols
        # same float32 order as gx_matmul.c: y += (s * xscale) * partial, pass by pass
        y = np.zeros(partial.shape[0], np.float32)
        for p in range(self.passes):
            y += (self.s[:, p // per_group] * xscale[p]) * partial[:, p].astype(np.float32)
        return y


class GPU:
    """Every weight matmul on the TEV (attention, norms and SiLU stay on the CPU)."""

    def __init__(self, W, mode="clamp8", headroom=1.0, cols=None, rescore_k=0):
        self.mats = {k: TevMatrix(m, mode, headroom, cols) for k, m in W.f32.items() if k != "emb"}
        cols = self.mats["cls"].cols
        self.label = f"GPU only ({mode}/{cols} cols, headroom {headroom})" + (f" + CPU top-{rescore_k}" if rescore_k else "")
        self.rescore_k = rescore_k
        self.cls_q8 = W.dequant("cls") if rescore_k else None

    def __call__(self, name, x):
        y = self.mats[name](x)
        if name == "cls" and self.rescore_k:
            y = rescore(y, self.cls_q8, x, self.rescore_k)
        return y

    def saturation(self):
        out = {}
        for k, m in self.mats.items():
            key = k[0] if isinstance(k, tuple) else k
            a = out.setdefault(key, [0, 0])
            a[0] += m.stats["saturated"]
            a[1] += m.stats["partials"]
        return {k: s / max(n, 1) for k, (s, n) in out.items()}


class Split:
    """Transformer layers on the CPU (int8), classifier on the TEV with exact CPU rescoring of the top K."""

    def __init__(self, W, mode="clamp8", headroom=1.0, cols=None, k=64):
        self.q8 = {n: W.dequant(n) for n in W.q8}
        self.cls = TevMatrix(W.f32["cls"], mode, headroom, cols)
        self.label = f"split ({mode}/{self.cls.cols} cols, headroom {headroom}, CPU top-{k})"
        self.k = k

    def __call__(self, name, x):
        if name != "cls":
            return self.q8[name] @ x
        return rescore(self.cls(x), self.q8["cls"], x, self.k)


def rescore(coarse, w_exact, x, k):
    """Exact logits for the K best coarse candidates; the coarse values stand in for the rest."""
    top = np.argpartition(-coarse, k)[:k]
    out = coarse.copy()
    out[top] = w_exact[top] @ x
    return out
