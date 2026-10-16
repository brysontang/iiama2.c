"""
Matrix-vector products on the Wii's TEV, simulated stage by stage with tev.py.

Layout (one draw = one "pass" over `cols` input columns):

  * Each framebuffer pixel holds 3 output rows, one per RGB channel. Every
    channel does the same math, so the simulator just treats rows as rows.
  * Weights live in an RGBA8 texture as offset-binary bytes, Wu = q + 128,
    where q is the same int8 (per-row, per-group-of-32 scale) the CPU uses,
    or (row_scales) that int8 requantized with one scale per row.
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
           The GPU-fast engine uses it with row_scales: smaller weights saturate
           less (0.3% of partial sums instead of 1.8%), which more than pays
           for the coarser scale, and the readback gets cheaper.
  exact11  8 columns/pass, 2 draws: one writes PREV's low byte (u8 wrap), one
           runs 3 extra halving stages and writes floor(PREV/8) + 128. The CPU
           rebuilds the exact 11-bit sum. 2 draws per 8 columns.
  blend    (BlendMatrix) 16 columns/pass like clamp8, but every pass blends
           into the same pixels, so the framebuffer does the cross-pass sum
           and the CPU reads one byte per row. Coarse, so it's for the
           classifier, with the CPU recomputing the top candidates exactly.
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


def requantize_rows(w_f32, group_size):
    """The CPU's per-group int8, requantized with one scale per row (gx_matmul.c's wu()). Returns (q, s)."""
    rows, n = w_f32.shape
    q, s, _ = quantize_q8(w_f32, group_size)
    w_cpu = q.astype(np.float32).reshape(rows, -1, group_size) * s.reshape(rows, -1, 1)  # what the CPU has
    qr, sr, _ = quantize_q8(w_cpu.reshape(rows, n), n)
    return qr.reshape(rows, n), sr


def fma32(a, b, c):
    """float32 a * b + c with one rounding, like Broadway's fmadds and ps_madds0."""
    return (np.float64(a) * np.asarray(b, np.float64) + np.asarray(c, np.float64)).astype(np.float32)


def quantize_x(xs, headroom):
    """Per-pass activation quantization. xs: (passes, cols). Returns konst bytes c, effective c', scale."""
    xmax = np.abs(xs).max(axis=1)
    xmax = np.where(xmax == 0, 1.0, xmax).astype(np.float32)
    inv = np.float32(256.0 * headroom) / xmax  # gx_matmul.c's quantize: one division, float32 throughout
    t = np.abs(xs).astype(np.float32) * inv[:, None]
    cp = np.clip(np.rint(t), 0, 256).astype(np.int32)
    # c' = c + (c >> 7) can't produce 128, so snap to the nearer of 127 / 129
    cp = np.where(cp == 128, np.where(t >= 128, 129, 127), cp)
    c = cp - (cp >= 129)
    return c, cp, xmax / headroom


class TevMatrix:
    def __init__(self, w_f32, mode="clamp8", headroom=1.0, cols=None, group_size=32, row_scales=False):
        self.mode, self.cols, self.headroom = mode, cols or MODES[mode], headroom
        assert self.cols <= MODES[mode], f"{mode} fits at most {MODES[mode]} columns per pass"
        rows, n = w_f32.shape
        self.n = n
        self.gs = group_size
        assert self.gs % self.cols == 0
        n_pad = -(-n // self.gs) * self.gs
        wp = np.zeros((rows, n_pad), np.float32)
        wp[:, :n] = w_f32
        self.row_scales = row_scales
        if row_scales:
            self.q, self.srow = requantize_rows(wp, self.gs)
        else:
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
        if self.row_scales:
            # gx_matmul.c's sum_pixel: acc = fma(x scale, byte - 128, acc) pass by pass, then the row scale
            acc = np.zeros(partial.shape[0], np.float32)
            for p in range(self.passes):
                acc = fma32(xscale[p], partial[:, p], acc)
            return self.srow * acc
        per_group = self.gs // self.cols
        # gx_matmul.c's reduce_batch: y = fma(s * xscale, partial, y) pass by pass (GCC fuses it into fmadds)
        y = np.zeros(partial.shape[0], np.float32)
        for p in range(self.passes):
            y = fma32(self.s[:, p // per_group] * xscale[p], partial[:, p], y)
        return y


# The blended classifier's settings in source/gx_matmul.c, and how many candidates the CPU recomputes
# exactly (LlmModel.rescore_k). tools/tev_sim/blend_sweep.py picked them: with these, the app's sampler
# (temperature 0.8, top-p 0.9) matches fp32's as closely as the old 18-bytes-per-row classifier did.
BLEND = {"headroom": 0.25, "a0": 32, "c0": 32}
RESCORE_K = 256


class BlendMatrix:
    """
    Every pass blended into the same framebuffer pixels, so the CPU reads back
    one byte per row instead of one per row per pass.

    Blending saturates at 8 bits and the TEV can't write a negative number, so
    passes alternate. Pass 0 draws without blending and writes a0 + c0 + P
    (stage 0's d is C0 = a0 + c0). Odd passes flip every stage's sign and
    blend with SUBTRACT (dst - src), writing c0 - P. Even passes write c0 + P
    and blend with ADD. The c0's cancel in pairs, so the framebuffer ends at
    a0 + sum(P), plus c0 if the pass count is odd, as long as nothing
    saturates on the way.

    Summing across passes in integers needs one scale for each row and one
    for x: the weights are requantized per row (from the CPU's per-group
    int8), and x gets a single scale for all columns, with a small headroom so
    whole-row sums fit in the byte.
    """

    cols = 16

    def __init__(self, w_f32, headroom=BLEND["headroom"], a0=BLEND["a0"], c0=BLEND["c0"], group_size=32):
        rows, n = w_f32.shape
        assert n % self.cols == 0
        self.n, self.headroom, self.a0, self.c0 = n, headroom, a0, c0
        qr, self.s = requantize_rows(w_f32, group_size)  # self.s: (rows,)
        self.wu = (qr.astype(np.int32) + 128).reshape(rows, -1, self.cols)
        self.passes = n // self.cols
        self.stats = {"calls": 0, "partials": 0, "saturated": 0}

    def tev_sums(self, x):
        """The framebuffer's whole-row sums S (rows,), and the x scale."""
        xs = np.asarray(x, np.float32).reshape(1, self.n)
        c, _, xscale = quantize_x(xs, self.headroom)
        c, xs = c.reshape(self.passes, self.cols), xs.reshape(self.passes, self.cols)
        rows = self.wu.shape[0]
        last = self.cols - 1
        acc = None
        for p in range(self.passes):
            odd = p % 2 == 1
            neg = (xs[p] < 0) != odd
            v = np.full(rows, self.a0 + self.c0 if p == 0 else self.c0, np.int32)  # stage 0's d: C0
            for k in range(self.cols):
                v = color_stage(HALF, self.wu[:, p, k], c[p, k], v, OP_SUB if neg[k] else OP_ADD,
                                BIAS_ADDHALF if neg[k] else BIAS_SUBHALF, SCALE_1, clamp=k == last)
            out = to_efb(v)
            if p == 0:
                acc = out
            elif odd:
                acc = np.maximum(acc - out, 0)  # GX_BM_SUBTRACT
            else:
                acc = np.minimum(acc + out, 255)  # GX_BM_BLEND, ONE * src + ONE * dst
        self.last_acc = acc  # the framebuffer bytes, for check_probe.py
        S = acc - self.a0 - (self.c0 if self.passes % 2 else 0)
        self.stats["calls"] += 1
        self.stats["partials"] += rows
        self.stats["saturated"] += int(((acc == 0) | (acc == 255)).sum())
        return S, xscale[0]

    def tev_partials(self, x):
        """Same interface as TevMatrix's, for the selftest hash: one 'pass' holding the whole-row sum."""
        S, xscale = self.tev_sums(x)
        return S.reshape(-1, 1), np.array([xscale])

    def __call__(self, x):
        S, xscale = self.tev_sums(x)
        return (self.s * xscale) * S.astype(np.float32)  # gx_matmul.c's order


class GPU:
    """Every weight matmul on the TEV (attention, norms and SiLU stay on the CPU)."""

    def __init__(self, W, mode="clamp8", headroom=1.0, cols=None, rescore_k=0, blend_cls=False, row_scales=False):
        self.mats = {k: TevMatrix(m, mode, headroom, cols, row_scales=row_scales)
                     for k, m in W.f32.items() if k not in ("emb", "cls")}
        self.mats["cls"] = BlendMatrix(W.f32["cls"]) if blend_cls else TevMatrix(W.f32["cls"], mode, headroom, cols)
        cols = self.mats[("w1", 0)].cols
        self.label = (f"GPU only ({mode}/{cols} cols, headroom {headroom}" + (", row scales" if row_scales else "") + ")"
                      + (", blended classifier" if blend_cls else "")
                      + (f" + CPU top-{rescore_k}" if rescore_k else ""))
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
        if mode == "blend":
            self.cls = BlendMatrix(W.f32["cls"], headroom)
        else:
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
