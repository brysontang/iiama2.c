#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
How much does running the matmuls on the Wii's TEV hurt the model?

    uv run tools/tev_sim/eval.py models/stories15M.bin models/tokenizer.bin > report.md

The report is markdown on stdout (progress goes to stderr); --json saves the
raw numbers too.

Generates reference stories with fp32, then replays them (teacher forcing)
through each backend and compares next-token distributions against fp32:

  top-1   how often the backend's most likely token matches fp32's
  KL      mean KL(fp32 || backend), in nats
  ppl     perplexity of the reference text

Then shows greedy stories from each backend so you can read the damage.

Variants are kind:mode:cols:headroom[:k], e.g. split:clamp8:16:0.5:64 or
gpu:exact11:8:1.0. kind is split, gpu, or gpu+rescore.
"""
import argparse
import json
import sys
import time

import numpy as np
import tev
from gxmatmul import GPU, Split, TevMatrix
from llama import F32, Q8, Tokenizer, Transformer, Weights, generate, softmax

PROMPTS = ["Once upon a time", "The grumpy turtle king", "One day, a little girl named Lily",
           "Tom and his dog went to the park", "There was a brave little robot", ""]

DEFAULT_VARIANTS = ["split:clamp8:16:0.5:64", "split:exact11:8:1.0:64",
                    "gpu:clamp8:16:0.5", "gpu:clamp8:8:0.7", "gpu:exact11:8:1.0", "gpu+rescore:exact11:8:1.0:64"]


def make_backend(W, spec):
    if spec == "f32":
        return F32(W)
    if spec == "q8":
        return Q8(W)
    kind, mode, cols, headroom, *k = spec.split(":")
    cols, headroom = int(cols), float(headroom)
    if kind == "split":
        return Split(W, mode, headroom, cols, int(k[0]) if k else 64)
    if kind == "gpu":
        return GPU(W, mode, headroom, cols)
    if kind == "gpu+rescore":
        return GPU(W, mode, headroom, cols, rescore_k=int(k[0]) if k else 64)
    raise SystemExit(f"unknown variant {spec}")


def replay(W, backend, seqs):
    """Teacher-forced logits for every position of every sequence."""
    out = []
    for toks in seqs:
        m = Transformer(W, backend)
        out.append(np.stack([m.forward(t, pos) for pos, t in enumerate(toks[:-1])]))
    return out


def compare(ref_logits, logits, seqs):
    top1, kl, nll, n = 0, 0.0, 0.0, 0
    for R, L, toks in zip(ref_logits, logits, seqs):
        for pos in range(len(R)):
            p, q = softmax(R[pos].astype(np.float64)), softmax(L[pos].astype(np.float64))
            top1 += int(np.argmax(R[pos]) == np.argmax(L[pos]))
            kl += float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
            nll += -float(np.log(q[toks[pos + 1]] + 1e-30))
            n += 1
    return {"top1": top1 / n, "kl": kl / n, "ppl": float(np.exp(nll / n)), "positions": n}


def classifier_recall(W, seqs, specs, ks=(1, 4, 16, 64)):
    """On the int8 CPU path's hidden states: is the true best token inside the TEV's top K?"""
    q8 = Q8(W)

    def tev_cls(spec):
        mode, cols, headroom = spec.split(":")
        return TevMatrix(W.f32["cls"], mode, float(headroom), int(cols))

    tevs = {s: tev_cls(s) for s in specs}
    hits = {s: {k: 0 for k in ks} for s in specs}
    nucleus = {s: [0, 0] for s in specs}
    n = 0

    class Probe(Q8):
        def __call__(self, name, x):
            y = q8(name, x)
            if name == "cls":
                nonlocal n
                n += 1
                best = int(np.argmax(y))
                p = softmax(y / 0.8)  # the app's default temperature
                order = np.argsort(-p)
                nuc = order[: int(np.searchsorted(np.cumsum(p[order]), 0.9)) + 1]
                for s, m in tevs.items():
                    c = m(x)
                    rank = int((c > c[best]).sum())
                    for k in ks:
                        hits[s][k] += int(rank < k)
                    top64 = set(np.argpartition(-c, 64)[:64].tolist())
                    nucleus[s][0] += sum(int(t) in top64 for t in nuc)
                    nucleus[s][1] += len(nuc)
            return y

    for toks in seqs:
        m = Transformer(W, Probe(W))
        for pos, t in enumerate(toks[:-1]):
            m.forward(t, pos)
    return {s: {"recall": {k: hits[s][k] / n for k in ks}, "nucleus_in_top64": nucleus[s][0] / nucleus[s][1]}
            for s in specs}


def render(results):
    """The markdown report, from the same dict that goes to --json."""
    c, ref = results["config"], results["reference"]
    out = [f"# TEV accuracy: {results['model']}", "",
           (f"dim {c['dim']}, hidden {c['hidden']}, {c['layers']} layers, vocab {c['vocab']}, "
            f"int8 group size {c['gs']}. Reference: {ref['stories']} stories sampled from fp32 at temperature 1, "
            f"{ref['positions']} positions, replayed through each backend."), "",
           "## Classifier on the TEV, int8 hidden states", "",
           "recall@K is how often the int8 CPU's best token lands in the TEV's top K.", "",
           "| TEV classifier | recall@1 | @4 | @16 | @64 | top-p nucleus in top 64 |", "|---|---|---|---|---|---|"]
    for s, r in results["classifier"].items():
        rc = {str(k): v for k, v in r["recall"].items()}  # int keys in memory, str after JSON
        out.append(f"| {s} | {rc['1']:.1%} | {rc['4']:.1%} | {rc['16']:.1%} | {rc['64']:.1%} | "
                   f"{r['nucleus_in_top64']:.1%} |")
    out += ["", "## End to end, teacher-forced against fp32", "",
            "| backend | top-1 vs fp32 | KL (nats) | ppl | TEV saturation |", "|---|---|---|---|---|"]
    for r in results["variants"].values():
        sat = ", ".join(f"{k} {v:.2%}" for k, v in r["saturation"].items()) or "-"
        out.append(f"| {r['label']} | {r['top1']:.1%} | {r['kl']:.4f} | {r['ppl']:.2f} | {sat} |")
    out += ["", "## Greedy stories"]
    prompts = [g["prompt"] for g in next(iter(results["greedy"].values()))]
    for i, prompt in enumerate(prompts):
        out += ["", f'### "{prompt}"' if prompt else "### (no prompt)"]
        for spec, gens in results["greedy"].items():
            g, label = gens[i], results["variants"][spec]["label"]
            div = g["diverges_from_q8_at"]
            note = "" if spec in ("f32", "q8") else (" (same as int8 CPU)" if div is None
                                                       else f" (diverges from int8 CPU at token {div})")
            out += ["", f"**{label}**{note}", "", "\n>\n".join("> " + line for line in g["text"].strip().split("\n"))]
    return "\n".join(out) + "\n"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("tokenizer")
    ap.add_argument("--steps", type=int, default=200, help="tokens per reference story")
    ap.add_argument("--stories", type=int, default=6, help="number of reference stories")
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    ap.add_argument("--gen-steps", type=int, default=120, help="tokens per greedy sample story")
    ap.add_argument("--json", help="also write the raw results here")
    args = ap.parse_args()

    tev.self_test(20000)
    W = Weights(args.checkpoint)
    tok = Tokenizer(args.tokenizer)
    results = {"model": args.checkpoint, "config": {"dim": W.dim, "hidden": W.hidden, "layers": W.n_layers,
                                                    "vocab": W.vocab, "gs": W.gs}, "variants": {}}

    t0 = time.time()
    seqs = [generate(W, F32(W), tok, PROMPTS[i % len(PROMPTS)], args.steps, temperature=1.0, seed=100 + i)[0]
            for i in range(args.stories)]
    results["reference"] = {"stories": len(seqs), "positions": sum(len(s) - 1 for s in seqs)}
    log(f"reference stories: {time.time() - t0:.0f}s")
    ref = replay(W, F32(W), seqs)

    t0 = time.time()
    results["classifier"] = classifier_recall(W, seqs, ["clamp8:16:0.5", "clamp8:8:0.7", "exact11:8:1.0"])
    log(f"classifier recall: {time.time() - t0:.0f}s")

    backends = {}
    for spec in ["f32", "q8"] + args.variants:
        t0 = time.time()
        be = make_backend(W, spec)
        backends[spec] = be
        logits = ref if spec == "f32" else replay(W, be, seqs)
        r = compare(ref, logits, seqs)
        sat = getattr(be, "saturation", dict)()
        if isinstance(be, Split):
            sat = {"cls": be.cls.stats["saturated"] / max(be.cls.stats["partials"], 1)}
        r["saturation"] = sat
        r["label"] = getattr(be, "label", spec)
        results["variants"][spec] = r
        log(f"{r['label']}: {time.time() - t0:.0f}s")

    gens = {}
    for prompt in PROMPTS[:2]:
        base = None
        for spec, be in backends.items():
            toks, text = generate(W, be, tok, prompt, args.gen_steps)
            if spec == "q8":
                base = toks
            div = None
            if base is not None and spec != "q8":
                div = next((i for i, (a, b) in enumerate(zip(toks, base)) if a != b), None)
                if div is None and len(toks) != len(base):
                    div = min(len(toks), len(base))
            gens.setdefault(spec, []).append({"prompt": prompt, "text": text, "diverges_from_q8_at": div})
    results["greedy"] = gens

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    print(render(results), end="")


if __name__ == "__main__":
    main()
