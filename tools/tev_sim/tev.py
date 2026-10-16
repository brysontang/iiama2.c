"""
Bit-exact model of one TEV color combiner stage, transcribed from Dolphin's
software renderer (Source/Core/VideoBackends/Software/Tev.cpp,
DrawColorRegular + the clamps in Tev::Draw). Vectorized over numpy arrays so a
whole matmul's worth of pixels runs at once.

    out = clamp((d + bias) * scale  +/-  lerp(a, b, c) * scale)

Inputs a, b, c are 8-bit unsigned (Dolphin stores them in `unsigned : 8`
bitfields, so register values wrap). d is 11-bit signed. Registers saturate to
[-1024, 1023], or [0, 255] when the stage's clamp bit is set. The value written
to the framebuffer is the last stage's register cast to u8 (it wraps).

source/gx_probe.c confirmed all of this on a real Wii. The one place Dolphin
and the hardware differ is outside this function: Dolphin starts each pixel's
PREV from the GX_SetTevColor value, and the Wii doesn't reset it at all.
"""
import numpy as np

# GX enums, same order as the hardware fields
OP_ADD, OP_SUB = 0, 1
BIAS_ZERO, BIAS_ADDHALF, BIAS_SUBHALF = 0, 1, 2
SCALE_1, SCALE_2, SCALE_4, SCALE_DIVIDE2 = 0, 1, 2, 3

_BIAS = np.array([0, 128, -128], np.int32)  # indexed by BIAS_*
_LSHIFT = {SCALE_1: 0, SCALE_2: 1, SCALE_4: 2, SCALE_DIVIDE2: 0}
_RSHIFT = {SCALE_1: 0, SCALE_2: 0, SCALE_4: 0, SCALE_DIVIDE2: 1}

# KCSEL fixed constants
K_1, K_7_8, K_3_4, K_5_8, K_1_2, K_3_8, K_1_4, K_1_8 = 255, 223, 191, 159, 128, 96, 64, 32
HALF, ONE, ZERO = 128, 255, 0


def _s11(v):
    """Wrap into an 11-bit signed bitfield, like Dolphin's `signed d : 11`."""
    return ((np.asarray(v, np.int32) + 1024) & 2047) - 1024


def color_stage(a, b, c, d, op=OP_ADD, bias=BIAS_ZERO, scale=SCALE_1, clamp=False):
    """One stage, one channel. a, b, c, d, op and bias may be arrays (broadcast together)."""
    a = np.asarray(a, np.int32) & 255
    b = np.asarray(b, np.int32) & 255
    c = np.asarray(c, np.int32) & 255
    d = _s11(d)
    op = np.asarray(op, np.int32)
    sub = op == OP_SUB

    c = c + (c >> 7)
    temp = a * (256 - c) + b * c
    temp = temp << _LSHIFT[scale]
    if scale != SCALE_DIVIDE2:
        temp = temp + np.where(sub, 127, 128)
    temp = temp >> 8
    temp = np.where(sub, -temp, temp)
    result = ((d + _BIAS[np.asarray(bias)]) << _LSHIFT[scale]) + temp
    result = result >> _RSHIFT[scale]
    return np.clip(result, 0, 255) if clamp else np.clip(result, -1024, 1023)


def _color_stage_scalar(a, b, c, d, op, bias, scale, clamp):
    """Line-by-line transcription of DrawColorRegular for one channel, used to test color_stage."""
    a, b, c = a & 255, b & 255, c & 255
    d = ((d + 1024) & 2047) - 1024
    c = c + (c >> 7)
    temp = a * (256 - c) + b * c
    temp <<= _LSHIFT[scale]
    temp += 0 if scale == SCALE_DIVIDE2 else (127 if op == OP_SUB else 128)
    temp >>= 8  # Python >> on ints is arithmetic, like s32 >> in C
    temp = -temp if op == OP_SUB else temp
    result = ((d + int(_BIAS[bias])) << _LSHIFT[scale]) + temp
    result >>= _RSHIFT[scale]
    return min(max(result, 0), 255) if clamp else min(max(result, -1024), 1023)


def to_efb(reg):
    """Final register -> RGB8 framebuffer byte (u8 cast)."""
    return np.asarray(reg, np.int32) & 255


def self_test(n=200000, seed=1):
    rng = np.random.default_rng(seed)
    a, b, c = (rng.integers(0, 256, n) for _ in range(3))
    d = rng.integers(-1024, 1024, n)
    op = rng.integers(0, 2, n)
    for bias in (BIAS_ZERO, BIAS_ADDHALF, BIAS_SUBHALF):
        for scale in (SCALE_1, SCALE_2, SCALE_4, SCALE_DIVIDE2):
            for clamp in (False, True):
                got = color_stage(a, b, c, d, op, bias, scale, clamp)
                for i in range(0, n, 997):
                    want = _color_stage_scalar(int(a[i]), int(b[i]), int(c[i]), int(d[i]), int(op[i]), bias, scale, clamp)
                    assert got[i] == want, (a[i], b[i], c[i], d[i], op[i], bias, scale, clamp, got[i], want)
    return True


if __name__ == "__main__":
    self_test()
    print("tev.color_stage matches the scalar transcription")
