#!/usr/bin/env python3
"""
Costs each engine in source/main.c on a real Wii, fitted to one.

    python3 tools/cost_model.py models/stories15M.bin
    python3 tools/cost_model.py models/stories15M.bin --pos 200 --sampling
    python3 tools/cost_model.py models/stories15M.bin --before   # before the blended classifier

Bottom-up: count the work each engine does per token, cost it with per-
operation numbers, and add memory traffic at an effective bandwidth. Dolphin
can't answer this: it doesn't model Broadway's pipeline, caches or bus, or GPU
timing at all.

Where the numbers come from:
  int8 matmul   from the compiled loop (devkitPPC -O3, powerpc-eabi-objdump):
                4 MACs per loop, 4 lbz + 4 lfsx (LUT) + 4 lfs (x) = 12 loads on
                Broadway's single load/store unit -> 3 cycles/MAC, plus about
                8 cycles of per-group overhead every 32 MACs.
  GX issue      ~3,600 instructions per 16-column pass (Dolphin's instruction
                count for the GX calls; libogc does read-modify-write on shadow
                registers). The Wii agreed within 10%.
  the rest      fitted to the bench on a real Wii (README, "Speed"): CPU memory
                bandwidth, the readback loop's real cost (cache misses on the
                framebuffer copy make it ~13 cycles per row and pass, not the
                ~3.3 its instructions suggest), TEV rate, copy rate, sync cost.
                Fitted totals land within 2% of the measured ones. gpu_bw never
                limits anything here, so it's a guess.

The fit was made before the blended classifier existed, and predicted its
speedups within 4% of what the Wii then measured (MEASURED vs --before).

Useful for asking what a change would buy before writing it. Stdlib only.
"""
import argparse
import struct

CPU_HZ, GPU_HZ = 729e6, 243e6

PARAMS = {
    "cpu_bw": 250e6,     # effective CPU streaming read bandwidth (the 60x bus peaks at 1.9 GB/s)
    "tev_rate": 486e6,   # TEV pixel-stages per second; each stage also fetches one texel
    "gpu_bw": 2.5e9,     # GPU texture fetch bandwidth from main memory (a guess; never the limit)
    "copy_rate": 972e6,  # framebuffer-to-RAM copy, pixels per second
    "sync_s": 40e-6,     # one GX_DrawDone round trip
    "ipc": 1.0,          # instructions per cycle for branchy glue code (norms, RoPE, GX calls)
}

# What the Wii measured (SD:/llama/bench_out.txt, stories15M, 64 greedy tokens, positions 0-63).
# The fit above was made on the "before" engines: the classifier as 18 clamp8 passes per row.
MEASURED_BEFORE = {"CPU": 7.41, "split (GPU classifier + CPU top-64)": 12.29, "GPU, fast (clamp8)": 20.55,
                   "GPU, exact (exact11)": 9.46, "GPU, exact + CPU top-64": 9.41}
MEASURED = {"CPU": 7.41, "split (GPU classifier + CPU top-256)": 15.34, "GPU, fast (clamp8)": 30.01,
            "GPU, exact (exact11)": 17.36}  # the current engines, blended classifier

RESCORE_K = 256      # classifier candidates the CPU recomputes exactly (LlmModel.rescore_k)

INT8_CYC_PER_MAC = 3.0 + 8 / 32
ATTN_CYC_PER_MAC = 3.5   # q.k has one dependent accumulator; att*v is load/load/store
READBACK_CYC = 13.0      # per row per pass in clamp8 mode, fitted (cache misses included)
READBACK_EXACT = 1.4     # exact11 reads two bytes per row and pass
ISSUE_INSTR = {"clamp8": 3600, "exact11": 2100}  # per pass
OTHER_INSTR = 2.5e6      # norms, RoPE trig, SiLU, residuals, embedding, greedy argmax
SAMPLING_INSTR = 5e6     # softmax over the vocab + top-p sort, when temperature > 0


def read_config(path):
    with open(path, "rb") as f:
        head = f.read(28)
    if head[:4] == b"WLLM":  # exported file
        with open(path, "rb") as f:
            h = struct.unpack(">4s11i", f.read(48))
        return dict(zip(["dim", "hidden", "layers", "heads", "kv_heads", "vocab", "seq_len"], h[4:11]))
    dim, hidden, layers, heads, kv, vocab, seq = struct.unpack("<7i", head)
    return {"dim": dim, "hidden": hidden, "layers": layers, "heads": heads, "kv_heads": kv, "vocab": abs(vocab),
            "seq_len": seq}


def matrices(c):
    kv = c["dim"] * c["kv_heads"] // c["heads"]
    layer = [(c["dim"], c["dim"]), (kv, c["dim"]), (kv, c["dim"]), (c["dim"], c["dim"]),
             (c["hidden"], c["dim"]), (c["dim"], c["hidden"]), (c["hidden"], c["dim"])]
    return layer * c["layers"], [(c["vocab"], c["dim"])]


def ceil_div(a, b):
    return -(-a // b)


def region(rows):
    """gx_matmul.c's pass region: smallest pw x ph (multiples of 4, pw <= 128) holding ceil(rows/3) pixels."""
    npx = ceil_div(rows, 3)
    best = None
    for ph in range(4, 481, 4):
        pw = ceil_div(ceil_div(npx, ph), 4) * 4
        if pw <= 128 and (best is None or pw * ph < best[0] * best[1]):
            best = (pw, ph)
    return best


def cpu_matmul_s(mats, p):
    macs = sum(r * n for r, n in mats)
    return macs * INT8_CYC_PER_MAC / CPU_HZ + macs / p["cpu_bw"]  # 1 byte of weight per MAC, stalls don't overlap


def gpu_matmul_s(mats, mode, p):
    """Returns (seconds, breakdown) for running `mats` on the TEV, as gx_matmul.c does: synchronously."""
    cols_per_pass, draws, stages = (8, 2, [8, 11]) if mode == "exact11" else (16, 1, [16])
    t = {"issue": 0.0, "gpu": 0.0, "sync+copy": 0.0, "readback": 0.0}
    for rows, n in mats:
        pw, ph = region(rows)
        passes = n // cols_per_pass
        per_batch = (640 // pw) * (480 // ph) // draws
        batches = ceil_div(passes, per_batch)
        px = pw * ph
        pixel_stages = passes * px * sum(stages)
        tex_bytes = passes * px * cols_per_pass * 4  # one RGBA8 texel per product stage
        issue = passes * ISSUE_INSTR["exact11" if mode == "exact11" else "clamp8"] / (p["ipc"] * CPU_HZ)
        gpu = max(pixel_stages / p["tev_rate"], tex_bytes / p["gpu_bw"])
        t["issue"] += issue
        t["gpu"] += max(gpu - issue, 0.0)  # the GPU works through the FIFO while the CPU is still issuing
        if mode == "blend":  # every pass blends into one region: one copy, one byte and one scale per row
            t["sync+copy"] += p["sync_s"] + px / p["copy_rate"]
            t["readback"] += rows * READBACK_CYC / CPU_HZ + (px * 4 + rows * 4) / p["cpu_bw"]
            continue
        t["sync+copy"] += batches * p["sync_s"] + passes * draws * px / p["copy_rate"]
        row_passes = rows * passes
        mem = passes * draws * px * 4 + rows * passes * 4 * cols_per_pass / 32  # framebuffer copy + scales
        t["readback"] += row_passes * READBACK_CYC * (1 if mode == "clamp8" else READBACK_EXACT) / CPU_HZ + mem / p["cpu_bw"]
    return sum(t.values()), t


def attention_s(c, pos, p):
    macs = c["layers"] * 2 * c["dim"] * (pos + 1)
    kv_bytes = c["layers"] * 2 * (pos + 1) * (c["dim"] * c["kv_heads"] // c["heads"]) * 4
    return macs * ATTN_CYC_PER_MAC / CPU_HZ + kv_bytes / p["cpu_bw"]


def rescore_s(c, k, p):
    """The CPU recomputing the classifier's top k: heap selection over the vocab, k int8 rows, the tail cap."""
    return (k * c["dim"] * INT8_CYC_PER_MAC + c["vocab"] * 9) / CPU_HZ + k * c["dim"] / p["cpu_bw"]


def engine_s(c, engine, p, pos, sampling, before=False):
    """before: the engines as first measured, with the classifier as 18 clamp8 passes per row."""
    layers, cls = matrices(c)
    base = attention_s(c, pos, p) + (OTHER_INSTR + (SAMPLING_INSTR if sampling else 0)) / (p["ipc"] * CPU_HZ)
    parts = {"attention+other": base}
    if engine == "CPU":
        parts["CPU matmuls"] = cpu_matmul_s(layers + cls, p)
        return sum(parts.values()), parts
    mode = "clamp8" if "fast" in engine or engine.startswith("split") else "exact11"
    cls_mode = (mode if not engine.startswith("split") else "clamp8") if before else "blend"
    k = (64 if engine.startswith("split") or "top-64" in engine else 0) if before else RESCORE_K
    if engine.startswith("split"):
        parts["CPU matmuls"] = cpu_matmul_s(layers, p)
        br = {}
    else:
        _, br = gpu_matmul_s(layers, mode, p)
    _, brc = gpu_matmul_s(cls, cls_mode, p)
    for key, v in brc.items():
        br[key] = br.get(key, 0.0) + v
    parts.update({"GPU " + key: v for key, v in br.items()})
    if k:
        parts[f"rescore top-{k}"] = rescore_s(c, k, p)
    return sum(parts.values()), parts


ENGINES = ["CPU", "split (GPU classifier + CPU top-256)", "GPU, fast (clamp8)", "GPU, exact (exact11)"]
ENGINES_BEFORE = ["CPU", "split (GPU classifier + CPU top-64)", "GPU, fast (clamp8)", "GPU, exact (exact11)",
                  "GPU, exact + CPU top-64"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="llama2.c checkpoint or exported model.bin (only the header is read)")
    ap.add_argument("--pos", type=int, default=32, help="context position to cost attention at (default 32, the bench's average)")
    ap.add_argument("--sampling", action="store_true", help="include temperature sampling (the app's default)")
    ap.add_argument("--before", action="store_true", help="the engines before the blended classifier (what the fit used)")
    args = ap.parse_args()

    c = read_config(args.model)
    engines, measured = (ENGINES_BEFORE, MEASURED_BEFORE) if args.before else (ENGINES, MEASURED)
    print(f"model: dim {c['dim']}, hidden {c['hidden']}, {c['layers']} layers, vocab {c['vocab']}; "
          f"position {args.pos}, {'sampling' if args.sampling else 'greedy'}\n")
    print("| engine | model tok/s | vs CPU | measured on a Wii (stories15M bench) |")
    print("|---|---|---|---|")
    cpu = 1 / engine_s(c, "CPU", PARAMS, args.pos, args.sampling)[0]
    for e in engines:
        tps = 1 / engine_s(c, e, PARAMS, args.pos, args.sampling, args.before)[0]
        m = measured.get(e)
        print(f"| {e} | {tps:.1f} | {tps / cpu:.1f}x | {f'{m:.2f}' if m else '-'} |")

    print("\nwhere a token's time goes (ms):\n")
    for e in engines:
        total, parts = engine_s(c, e, PARAMS, args.pos, args.sampling, args.before)
        print(f"  {e}: {total * 1e3:.1f} ms = " + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in parts.items() if v > 5e-5))


if __name__ == "__main__":
    main()
