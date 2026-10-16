#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
Picks the blended classifier's settings (gxmatmul.BLEND) and how many
candidates the CPU recomputes.

    uv run tools/tev_sim/blend_sweep.py models/stories15M.bin models/tokenizer.bin

The blended classifier gives coarse logits for the whole vocab; the CPU then
recomputes the top K exactly. Settings: x headroom (precision vs saturating
the top tokens), a0 (where the framebuffer starts) and c0 (the TEV's offset
per pass). On the int8 CPU path's hidden states (the split engine), teacher
forced on fp32-sampled stories like eval.py, it reports:

  recall@K   how often the int8 CPU's best token is in the blended top K
             (ties counted against it)
  top-1      after the CPU recomputes the top K, against fp32
  KL         full distribution at temperature 1, against fp32 (the tail's
             coarse logits count here)
  sampler    total variation distance between the app's sampling
             distributions (temperature 0.8, top-p 0.9) and fp32's: how often
             the two would draw different tokens, at best
"""
import argparse
import os
import sys
import time

import numpy as np
from gxmatmul import BlendMatrix, TevMatrix
from llama import F32, Q8, Tokenizer, Transformer, Weights, generate, softmax

PROMPTS = ["Once upon a time", "The grumpy turtle king", "One day, a little girl named Lily",
           "Tom and his dog went to the park", "There was a brave little robot", ""]


def positions(W, tok, stories, steps):
    seqs = [generate(W, F32(W), tok, PROMPTS[i % len(PROMPTS)], steps, temperature=1.0, seed=100 + i)[0]
            for i in range(stories)]
    refs, xs = [], []

    class Ref(F32):
        def __call__(self, name, x):
            y = super().__call__(name, x)
            if name == "cls":
                refs.append(y)
            return y

    class Probe(Q8):
        def __call__(self, name, x):
            if name == "cls":
                xs.append(x.copy())
            return super().__call__(name, x)

    for toks in seqs:
        for backend in (Ref(W), Probe(W)):
            m = Transformer(W, backend)
            for pos, t in enumerate(toks[:-1]):
                m.forward(t, pos)
    return np.array(xs), np.array(refs)


def sampler(logits, temperature=0.8, topp=0.9):
    """The app's sampling distribution (llama.sample's truncation)."""
    p = softmax(logits.astype(np.float64) / temperature)
    order = np.argsort(-p, kind="stable")
    keep = order[: int(np.searchsorted(np.cumsum(p[order]), topp, side="right")) + 1]
    q = np.zeros_like(p)
    q[keep] = p[keep] / p[keep].sum()
    return q


def rescore(coarse, w_exact, x, k, cap):
    top = np.argpartition(-coarse, k)[:k]
    out = coarse.copy()
    out[top] = w_exact[top] @ x
    if cap:  # nothing outside the top K may outrank anything inside it
        rest = np.ones(len(out), bool)
        rest[top] = False
        out[rest] = np.minimum(out[rest], out[top].min())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("tokenizer")
    ap.add_argument("--stories", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cache", help="save/load the hidden states here (.npz)")
    args = ap.parse_args()

    W = Weights(args.checkpoint)
    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        xs, refs = z["xs"], z["refs"]
    else:
        xs, refs = positions(W, Tokenizer(args.tokenizer), args.stories, args.steps)
        if args.cache:
            np.savez(args.cache, xs=xs, refs=refs)
    n = len(xs)
    print(f"{n} positions", file=sys.stderr)

    cls_q8 = W.dequant("cls")
    exact = xs @ cls_q8.T
    ref_p = [softmax(r.astype(np.float64)) for r in refs]
    ref_s = [sampler(r) for r in refs]

    def metrics(outs):
        top1 = np.mean([np.argmax(o) == np.argmax(r) for o, r in zip(outs, refs)])
        kl = np.mean([np.sum(p * (np.log(p + 1e-30) - np.log(softmax(o.astype(np.float64)) + 1e-30)))
                      for o, p in zip(outs, ref_p)])
        tv = np.mean([0.5 * np.abs(sampler(o) - s).sum() for o, s in zip(outs, ref_s)])
        return f"{top1:.1%} | {kl:.4f} | {tv:.3f}"

    print("| classifier | recall@1 | @16 | @64 | CPU recomputes | top-1 vs fp32 | KL | sampler TV |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| int8 CPU, whole vocab | | | | all | {metrics(exact)} |", flush=True)

    def sweep(name, m):
        t0 = time.time()
        coarse = [m(x) for x in xs]
        ranks = np.array([int((c >= c[int(np.argmax(y))]).sum()) - 1 for c, y in zip(coarse, exact)])
        rec = " | ".join(f"{np.mean(ranks < k):.1%}" for k in (1, 16, 64))
        for k, cap in ((64, False), (64, True), (256, False), (256, True)):
            outs = [rescore(c, cls_q8, x, k, cap) for c, x in zip(coarse, xs)]
            print(f"| {name} | {rec} | top {k}{', tail capped' if cap else ''} | {metrics(outs)} |", flush=True)
        print(f"  ({time.time() - t0:.0f}s)", file=sys.stderr)

    sweep("clamp8, 18 bytes/row (before)", TevMatrix(W.f32["cls"], "clamp8", 0.5, 16))
    for h in (0.125, 0.1875, 0.25):
        for a0, c0 in ((32, 32), (16, 32), (64, 32)):
            sweep(f"blend h={h} a0={a0} c0={c0}", BlendMatrix(W.f32["cls"], h, a0, c0))


if __name__ == "__main__":
    main()
