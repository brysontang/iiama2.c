#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
Does the real Wii build (run in Dolphin) compute what the simulator says?

    tools/dolphin_bench.sh sd/llama 16 > dolphin.txt
    uv run tools/tev_sim/check_dolphin.py models/stories15M.bin models/tokenizer.bin dolphin.txt

Two checks, both against the bench output of source/main.c:

  selftest  FNV-1a of every pass's integer sum for a fixed input, per matrix
            and output mode. These come straight off the GPU (no floats), so
            they must match bit for bit.
  stories   the greedy story from each engine. The CPU-side float math differs
            a little between numpy and C, so a late divergence is possible
            but they should mostly agree.
"""
import argparse
import re
import sys

import numpy as np
from gxmatmul import GPU, Split, TevMatrix
from llama import Q8, Tokenizer, Weights, generate

# must match ENGINES in source/main.c
ENGINES = [
    ("CPU", lambda W: Q8(W)),
    ("split (GPU classifier + CPU top-64)", lambda W: Split(W, "clamp8", 0.5, 16, 64)),
    ("GPU, fast (clamp8)", lambda W: GPU(W, "clamp8", 0.5, 16)),
    ("GPU, exact (exact11)", lambda W: GPU(W, "exact11", 1.0, 8)),
    ("GPU, exact + CPU top-64", lambda W: GPU(W, "exact11", 1.0, 8, rescore_k=64)),
]
CHECKS = [("clamp8", 0.5, 16), ("exact11", 1.0, 8)]


def fnv1a_partials(partial):
    h = 2166136261
    for b in partial.astype("<i2").tobytes():
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def selftest(W):
    n = max(W.dim, W.hidden)
    x = ((((np.arange(n) * 37) % 101) - 50) / 16.0).astype(np.float32)
    out = {}
    for mode, headroom, cols in CHECKS:
        for name, key in (("cls", "cls"), ("w1", ("w1", 0)), ("w2", ("w2", 0))):
            m = TevMatrix(W.f32[key], mode, headroom, cols)
            partial, _ = m.tev_partials(x[: m.n])
            out[(mode, name)] = f"{fnv1a_partials(partial):08x}"
    return out


def parse_bench(text):
    hashes, stories = {}, {}
    for line in text.splitlines():
        m = re.match(r"selftest (\w+) headroom [\d.]+: (.*)", line)
        if m:
            for name, h in re.findall(r"(\w+) ([0-9a-f]{8})", m.group(2)):
                hashes[(m.group(1), name)] = h
    for m in re.finditer(r"\n\[(.+?)\] \d+ tokens.*?\n(?:  GPU:.*?\n)?  (.*?)(?=\n\[|\ndone|\Z)", text, re.DOTALL):
        stories[m.group(1)] = m.group(2)
    return hashes, stories


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("tokenizer")
    ap.add_argument("bench_output", help="what tools/dolphin_bench.sh printed")
    args = ap.parse_args()

    with open(args.bench_output) as f:
        bench = f.read()
    got_hashes, got_stories = parse_bench(bench)
    steps = int(re.search(r"\] (\d+) tokens", bench).group(1)) if "] " in bench else 64

    W = Weights(args.checkpoint)
    tok = Tokenizer(args.tokenizer)
    ok = True

    print("selftest (integer TEV output, must match exactly)")
    for key, want in selftest(W).items():
        got = got_hashes.get(key)
        ok &= got == want
        print(f"  {key[0]:8s} {key[1]:4s} sim {want}  dolphin {got}  {'OK' if got == want else 'MISMATCH'}")

    print("\nstories")
    for name, make in ENGINES:
        if name not in got_stories:
            print(f"  [{name}] not in the Dolphin output")
            continue
        _, want = generate(W, make(W), tok, "Once upon a time", steps)
        got = got_stories[name].rstrip()
        same = got == want.rstrip()
        n = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        print(f"  [{name}] {'IDENTICAL' if same else f'differs after {n} chars'}")
        if not same:
            print(f"    sim:     {want!r}\n    dolphin: {got!r}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
