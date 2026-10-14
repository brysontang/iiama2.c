#!/usr/bin/env python3
"""
Predicts tok/s on a real Wii for each engine in source/main.c, before anyone
runs it on one. Dolphin can't answer this: it doesn't model Broadway's
pipeline, caches or bus, or GPU timing at all.

    python3 tools/cost_model.py models/stories15M.bin
    python3 tools/cost_model.py models/stories15M.bin --pos 200 --sampling

Bottom-up: count the work each engine does per token, cost it with per-
operation numbers, and add memory traffic at an effective bandwidth. Numbers
that come from the compiled code are fixed; the ones nobody has measured get a
pessimistic / central / optimistic range.

Where the fixed numbers come from (devkitPPC -O3, powerpc-eabi-objdump):
  int8 matmul   4 MACs per loop: 4 lbz + 4 lfsx (LUT) + 4 lfs (x) = 12 loads on
                Broadway's single load/store unit -> 3 cycles/MAC, plus about
                8 cycles of per-group overhead every 32 MACs.
  GPU readback  per row and pass: lbz + lfsx (LUT) + lfs (scale) -> ~3.3 cycles.
  GX issue      ~3,600 instructions per 16-column pass (Dolphin's instruction
                count for the GX calls; libogc does read-modify-write on shadow
                registers).

The bench mode (SD:/llama/bench.txt) prints the issue / wait / readback split
for each engine on real hardware, which pins down the unknowns below.
Stdlib only.
"""
import argparse
import struct

CPU_HZ, GPU_HZ = 729e6, 243e6

# (pessimistic, central, optimistic)
UNKNOWN = {
    # effective CPU streaming read bandwidth with no prefetching (in-order core,
    # one cache line miss at a time); the 60x bus peak is 1.9 GB/s
    "cpu_bw": (150e6, 300e6, 600e6),
    # TEV pixel-stages per second; each stage also fetches one texel
    "tev_rate": (243e6, 486e6, 972e6),
    # GPU texture fetch bandwidth from main memory (weights stream, no reuse)
    "gpu_bw": (1.5e9, 2.5e9, 3.5e9),
    # framebuffer-to-RAM copy, pixels per second
    "copy_rate": (121e6, 243e6, 486e6),
    # one GX_DrawDone round trip
    "sync_s": (40e-6, 20e-6, 8e-6),
    # instructions per cycle for branchy glue code (norms, RoPE, GX calls)
    "ipc": (0.7, 1.0, 1.4),
}

INT8_CYC_PER_MAC = 3.0 + 8 / 32
ATTN_CYC_PER_MAC = 3.5   # q.k has one dependent accumulator; att*v is load/load/store
READBACK_CYC = 3.3       # per row per pass, before memory stalls
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
    cols_per_pass, draws, stages = (16, 1, [16]) if mode == "clamp8" else (8, 2, [8, 11])
    t = {"issue": 0.0, "gpu": 0.0, "sync+copy": 0.0, "readback": 0.0}
    for rows, n in mats:
        pw, ph = region(rows)
        passes = n // cols_per_pass
        per_batch = (640 // pw) * (480 // ph) // draws
        batches = ceil_div(passes, per_batch)
        px = pw * ph
        pixel_stages = passes * px * sum(stages)
        tex_bytes = passes * px * cols_per_pass * 4  # one RGBA8 texel per product stage
        issue = passes * ISSUE_INSTR[mode] / (p["ipc"] * CPU_HZ)
        gpu = max(pixel_stages / p["tev_rate"], tex_bytes / p["gpu_bw"])
        t["issue"] += issue
        t["gpu"] += max(gpu - issue, 0.0)  # the GPU works through the FIFO while the CPU is still issuing
        t["sync+copy"] += batches * p["sync_s"] + passes * draws * px / p["copy_rate"]
        row_passes = rows * passes
        mem = passes * draws * px * 4 + rows * passes * 4 * cols_per_pass / 32  # framebuffer copy + scales
        t["readback"] += row_passes * READBACK_CYC * (1 if mode == "clamp8" else 1.6) / CPU_HZ + mem / p["cpu_bw"]
    return sum(t.values()), t


def attention_s(c, pos, p):
    macs = c["layers"] * 2 * c["dim"] * (pos + 1)
    kv_bytes = c["layers"] * 2 * (pos + 1) * (c["dim"] * c["kv_heads"] // c["heads"]) * 4
    return macs * ATTN_CYC_PER_MAC / CPU_HZ + kv_bytes / p["cpu_bw"]


def engine_s(c, engine, p, pos, sampling):
    layers, cls = matrices(c)
    base = attention_s(c, pos, p) + (OTHER_INSTR + (SAMPLING_INSTR if sampling else 0)) / (p["ipc"] * CPU_HZ)
    parts = {"attention+other": base}
    if engine == "CPU":
        parts["CPU matmuls"] = cpu_matmul_s(layers + cls, p)
    elif engine.startswith("split"):
        parts["CPU matmuls"] = cpu_matmul_s(layers, p)
        _, br = gpu_matmul_s(cls, "clamp8", p)
        parts.update({"GPU " + k: v for k, v in br.items()})
        parts["rescore top-64"] = (64 * c["dim"] * INT8_CYC_PER_MAC + c["vocab"] * 6) / CPU_HZ
    else:
        mode = "clamp8" if "fast" in engine else "exact11"
        _, br = gpu_matmul_s(layers + cls, mode, p)
        parts.update({"GPU " + k: v for k, v in br.items()})
        if "top-64" in engine:
            parts["rescore top-64"] = (64 * c["dim"] * INT8_CYC_PER_MAC + c["vocab"] * 6) / CPU_HZ
    return sum(parts.values()), parts


ENGINES = ["CPU", "split (GPU classifier + CPU top-64)", "GPU, fast (clamp8)", "GPU, exact (exact11)",
           "GPU, exact + CPU top-64"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="llama2.c checkpoint or exported model.bin (only the header is read)")
    ap.add_argument("--pos", type=int, default=128, help="context position to cost attention at (default 128)")
    ap.add_argument("--sampling", action="store_true", help="include temperature sampling (the app's default)")
    args = ap.parse_args()

    c = read_config(args.model)
    names = ("pessimistic", "central", "optimistic")
    scen = [{k: v[i] for k, v in UNKNOWN.items()} for i in range(3)]
    print(f"model: dim {c['dim']}, hidden {c['hidden']}, {c['layers']} layers, vocab {c['vocab']}; "
          f"position {args.pos}, {'sampling' if args.sampling else 'greedy'}\n")
    print("| engine | " + " | ".join(f"{n} tok/s" for n in names) + " | vs CPU (central) |")
    print("|---|---|---|---|---|")
    cpu_central = 1 / engine_s(c, "CPU", scen[1], args.pos, args.sampling)[0]
    for e in ENGINES:
        tps = [1 / engine_s(c, e, s, args.pos, args.sampling)[0] for s in scen]
        print(f"| {e} | " + " | ".join(f"{t:.1f}" for t in tps) + f" | {tps[1] / cpu_central:.1f}x |")

    print("\nwhere a token's time goes (central, ms):\n")
    for e in ENGINES:
        total, parts = engine_s(c, e, scen[1], args.pos, args.sampling)
        print(f"  {e}: {total * 1e3:.1f} ms = " + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in parts.items() if v > 5e-5))


if __name__ == "__main__":
    main()
