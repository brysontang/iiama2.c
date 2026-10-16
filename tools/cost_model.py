#!/usr/bin/env python3
"""
Costs each engine in source/main.c on a real Wii, fitted to one.

    python3 tools/cost_model.py models/stories15M.bin
    python3 tools/cost_model.py models/stories15M.bin --pos 200 --sampling
    python3 tools/cost_model.py models/stories15M.bin --before   # the engines as first run on a Wii
    python3 tools/cost_model.py models/stories15M.bin --unstacked  # GPU engines before wq, wk, wv were stacked

Bottom-up: count the work each engine does per token, cost it with per-
operation numbers, and add memory traffic at an effective bandwidth. Dolphin
can't answer this: it doesn't model Broadway's pipeline, caches or bus, or GPU
timing at all.

Where the numbers come from:
  int8 matmul   from the compiled loop (devkitPPC -O3, powerpc-eabi-objdump):
                4 MACs per loop, 4 lbz + 4 lfsx (LUT) + 4 lfs (x) = 12 loads on
                Broadway's single load/store unit -> 3 cycles/MAC, plus about
                8 cycles of per-group overhead every 32 MACs.
  GPU           measured by the probe (gx_probe.c): the TEV does 951M pixel-
                stages/s when texture fetch keeps up, but fetching from MEM2
                holds it to 478M texels/s for RGBA8 and 716M for RGB565.
  issue         measured by the probe: cycles per pass, raw register writes
                now, libogc's setters before.
  readback      fitted to the bench, all in (instructions and cache misses):
                per pixel and pass for GPU fast, per row and pass for exact11,
                per row for the blended output layer.
  attention     from the compiled loops: q.k waits on one accumulator, att*v
                is four loads and stores a MAC; int8 keys and values add a table
                lookup. Quantizing each new key and value divides and calls rintf.
  the rest      fitted to the bench on a real Wii (README, "Speed"): CPU memory
                bandwidth, copy rate, sync cost, ~2.6 us of GPU time per draw in
                GPU fast's pipeline that the probes don't explain yet, and how
                fast the KV cache reads (mostly from L2 at the bench's positions).

The first fit (--before) predicted the blended classifier's speedups within 4%
of what the Wii then measured. The next round's prediction missed GPU fast by
17% (51 vs 43.5 tok/s), mostly on the readback's cache misses. The one after
missed by 3% (51 vs 49.4): it read the float KV cache at main-memory speed,
when at positions 0-63 it mostly sits in L2, so shrinking it to int8 saved less
than predicted. How the KV cache costs further into a story is a guess until a
long run measures it. MEASURED predates stacking wq, wk, wv and w1, w3 into one
GPU matrix each, so the fit is checked with --unstacked; the default is the
prediction for stacking.

Useful for asking what a change would buy before writing it. Stdlib only.
"""
import argparse
import struct

CPU_HZ, GPU_HZ = 729e6, 243e6

PARAMS = {
    "cpu_bw": 250e6,     # effective CPU streaming read bandwidth, as the int8 matmul sees it (MEM1)
    "tev_rate": 951e6,   # TEV pixel-stages per second when texture fetch keeps up (4 pixels a clock)
    "tex_rate": {"RGBA8": 478e6, "RGB565": 716e6},  # texels per second fetched from MEM2 (probe)
    "copy_rate": 972e6,  # framebuffer-to-RAM copy, pixels per second
    "sync_s": 40e-6,     # one GX_DrawDone round trip
    "draw_s": 2.6e-6,    # GPU time per small pipelined draw beyond its pixels (fitted to GPU fast; unexplained)
    "kv_bw": 0.84e9,     # KV cache reads, fitted: at the bench's positions it's mostly in L2
    "ipc": 1.0,          # instructions per cycle for branchy glue code (norms, RoPE, GX calls)
}

# What the Wii measured (SD:/llama/bench_out.txt, stories15M, 64 greedy tokens, positions 0-63): the
# engines as first run, and before the GPU engines stacked wq, wk, wv and w1, w3 (--unstacked).
MEASURED_BEFORE = {"CPU": 7.41, "split (GPU classifier + CPU top-64)": 12.29, "GPU, fast (clamp8)": 20.55,
                   "GPU, exact (exact11)": 9.46, "GPU, exact + CPU top-64": 9.41}
MEASURED = {"CPU": 7.44, "split (GPU classifier + CPU top-256)": 16.06, "GPU, fast (clamp8)": 49.41,
            "GPU, exact (exact11)": 21.91}

RESCORE_K = 256      # classifier candidates the CPU recomputes exactly (LlmModel.rescore_k)

INT8_CYC_PER_MAC = 3.0 + 8 / 32
ATTN_CYC_PER_MAC = {"f32": 3.5, "int8": 3.75}  # q.k waits on its accumulator; att*v is load/load/store (+ lookup)
QUANT_KV_CYC = 45        # per key or value element: a divide and a call to rintf
# Readback, all in, fitted to the Wii. Now: framebuffer copies in MEM1, per-group scales in MEM2.
ROWS_CYC_PER_PIXEL_PASS = 21.0  # GPU fast (sum_pixel): per pixel (3 rows) and pass; ~5 of it instructions
EXACT_CYC_PER_ROW_PASS = 24.8   # exact11: two bytes, two table lookups and a scale per row and pass
BLEND_CYC_PER_ROW = 32.4        # the blended output layer: a byte, a row scale and a logit per row
ROWS_CHUNK = 6                  # passes per framebuffer copy (gx_matmul.c's CHUNK)
HOT_BYTES = 128 * 1024          # the MEM1 copy buffer; exact11 batches are cut to fit it (gx_matmul.c's g_hot)
ISSUE_CYC = {"clamp8": 1270, "exact11": 940}  # per pass, raw register writes (probe)
# Before: copies in MEM2, libogc's setters
READBACK_CYC = 13.0      # per row and pass, clamp8, plus the copy and scales at cpu_bw
READBACK_EXACT = 1.4     # exact11 relative to clamp8
ISSUE_CYC_LIBOGC = {"clamp8": 3600, "exact11": 2100}
OTHER_INSTR = 3.1e6      # norms, SiLU, residuals, embedding, greedy argmax
ROPE_INSTR = {"per layer": 0.35e6, "per token": 0.01e6}  # powf/cosf/sinf, before and after hoisting them
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


def matrices(c, stacked=False):
    """(rows, cols) of every layer matrix, and the classifier's. stacked: as the GPU engines upload them,
    wq, wk, wv's rows in one matrix and w1, w3's in another."""
    kv = c["dim"] * c["kv_heads"] // c["heads"]
    if stacked:
        layer = [(c["dim"] + 2 * kv, c["dim"]), (c["dim"], c["dim"]), (2 * c["hidden"], c["dim"]), (c["dim"], c["hidden"])]
    else:
        layer = [(c["dim"], c["dim"]), (kv, c["dim"]), (kv, c["dim"]), (c["dim"], c["dim"]),
                 (c["hidden"], c["dim"]), (c["dim"], c["hidden"]), (c["hidden"], c["dim"])]
    return layer * c["layers"], [(c["vocab"], c["dim"])]


def ceil_div(a, b):
    return -(-a // b)


def region(rows, narrow=False):
    """
    gx_matmul.c's pass region: smallest pw x ph (multiples of 4, pw <= 128) holding ceil(rows/3) pixels, or
    for GPU fast's per-row textures (narrow), 4 pixels wide when that fits the framebuffer's 480 lines.
    """
    npx = ceil_div(rows, 3)
    if narrow and ceil_div(ceil_div(npx, 4), 4) * 4 <= 480:
        return 4, ceil_div(ceil_div(npx, 4), 4) * 4
    best = None
    for ph in range(4, 481, 4):
        pw = ceil_div(ceil_div(npx, ph), 4) * 4
        if pw <= 128 and (best is None or pw * ph < best[0] * best[1]):
            best = (pw, ph)
    return best


def cpu_matmul_s(mats, p):
    macs = sum(r * n for r, n in mats)
    return macs * INT8_CYC_PER_MAC / CPU_HZ + macs / p["cpu_bw"]  # 1 byte of weight per MAC, stalls don't overlap


def pipeline_s(chunks):
    """
    One matmul in chunks of (issue, gpu, sum) seconds, the way run_rows overlaps them: the CPU issues
    chunk c + 1, then waits for chunk c and sums it, while the GPU works through whatever is issued.
    """
    cpu, gpu, done = 0.0, 0.0, []
    for c in range(len(chunks) + 1):
        if c < len(chunks):
            cpu += chunks[c][0]
            gpu = max(gpu, cpu) + chunks[c][1]
            done.append(gpu)
        if c > 0:
            cpu = max(cpu, done[c - 1]) + chunks[c - 1][2]
    return cpu


def rows_matmul_s(mats, p):
    """(seconds, breakdown) for clamp8 with per-row scales, pipelined (run_rows)."""
    t = {"issue+sum": 0.0, "waiting": 0.0}
    for rows, n in mats:
        pw, ph = region(rows)
        passes, px = n // 16, pw * ph
        chunks = []
        for p0 in range(0, passes, ROWS_CHUNK):
            k = min(ROWS_CHUNK, passes - p0)
            texels = k * px * 16
            gpu = max(texels / p["tev_rate"], texels / p["tex_rate"]["RGBA8"]) + k * px / p["copy_rate"] + k * p["draw_s"]
            total_sum = ceil_div(rows, 3) * k * ROWS_CYC_PER_PIXEL_PASS / CPU_HZ
            chunks.append((k * ISSUE_CYC["clamp8"] / CPU_HZ, gpu, total_sum))
        total = pipeline_s(chunks)
        busy = sum(c[0] + c[2] for c in chunks)
        t["issue+sum"] += busy
        t["waiting"] += total - busy
    return sum(t.values()), t


def gpu_matmul_s(mats, mode, p, before=False):
    """(seconds, breakdown) for `mats` on the TEV, synchronously: exact11, the blended output layer, or before."""
    cols_per_pass, draws, stages = (8, 2, [8, 11]) if mode == "exact11" else (16, 1, [16])
    t = {"issue": 0.0, "gpu": 0.0, "sync+copy": 0.0, "readback": 0.0}
    for rows, n in mats:
        pw, ph = region(rows)
        passes = n // cols_per_pass
        per_batch = (640 // pw) * (480 // ph) // draws
        hot = HOT_BYTES // ((640 // pw) * pw * ph * 4) * (640 // pw) // draws  # passes whose copy fits in MEM1
        if not before and 0 < hot < per_batch:
            per_batch = hot
        batches = ceil_div(passes, per_batch)
        px = pw * ph
        pixel_stages = passes * px * sum(stages)
        texels = passes * px * cols_per_pass  # one texel per product stage
        fmt = "RGB565" if mode == "blend" and not before else "RGBA8"
        issue = passes * (ISSUE_CYC_LIBOGC if before else ISSUE_CYC)["exact11" if mode == "exact11" else "clamp8"] / CPU_HZ
        gpu = max(pixel_stages / p["tev_rate"], texels / p["tex_rate"][fmt])
        t["issue"] += issue
        t["gpu"] += max(gpu - issue, 0.0)  # the GPU works through the FIFO while the CPU is still issuing
        if mode == "blend":  # every pass blends into one region: one copy, one byte and one scale per row
            t["sync+copy"] += p["sync_s"] + px / p["copy_rate"]
            t["readback"] += rows * BLEND_CYC_PER_ROW / CPU_HZ
            continue
        t["sync+copy"] += batches * p["sync_s"] + passes * draws * px / p["copy_rate"]
        row_passes = rows * passes
        if not before:
            t["readback"] += row_passes * EXACT_CYC_PER_ROW_PASS / CPU_HZ
            continue
        mem = passes * draws * px * 4 + rows * passes * 4 * cols_per_pass / 32  # framebuffer copy + scales
        t["readback"] += row_passes * READBACK_CYC * (1 if mode == "clamp8" else READBACK_EXACT) / CPU_HZ + mem / p["cpu_bw"]
    return sum(t.values()), t


def attention_s(c, pos, p, kv="int8"):
    """Attention over positions 0..pos, KV cache as int8 (a scale per head vector) or f32, plus caching this token."""
    macs = c["layers"] * 2 * c["dim"] * (pos + 1)
    kv_dim = c["dim"] * c["kv_heads"] // c["heads"]
    vectors = c["layers"] * 2 * (pos + 1)
    kv_bytes = vectors * (kv_dim + 4 * c["kv_heads"]) if kv == "int8" else vectors * kv_dim * 4
    quantize = c["layers"] * 2 * kv_dim * QUANT_KV_CYC if kv == "int8" else 0
    return (macs * ATTN_CYC_PER_MAC[kv] + quantize) / CPU_HZ + kv_bytes / p["kv_bw"]


def rescore_s(c, k, p):
    """The CPU recomputing the classifier's top k: heap selection over the vocab, k int8 rows, the tail cap."""
    return (k * c["dim"] * INT8_CYC_PER_MAC + c["vocab"] * 9) / CPU_HZ + k * c["dim"] / p["cpu_bw"]


def engine_s(c, engine, p, pos, sampling, before=False, stacked=True):
    """before: the engines as first measured, with the classifier as 18 clamp8 passes per row.
    stacked: the GPU engines run wq, wk, wv as one matrix and w1, w3 as another (not before)."""
    layers, cls = matrices(c, stacked and not before and engine.startswith("GPU"))
    other = OTHER_INSTR + ROPE_INSTR["per layer" if before else "per token"] + (SAMPLING_INSTR if sampling else 0)
    base = attention_s(c, pos, p, "f32" if before else "int8") + other / (p["ipc"] * CPU_HZ)
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
    elif mode == "clamp8" and not before:
        _, br = rows_matmul_s(layers, p)
    else:
        _, br = gpu_matmul_s(layers, mode, p, before)
    _, brc = gpu_matmul_s(cls, cls_mode, p, before)
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
    ap.add_argument("--before", action="store_true", help="the engines as first run on a Wii (what the first fit used)")
    ap.add_argument("--unstacked", action="store_true",
                    help="the GPU engines with wq, wk, wv and w1, w3 as separate matrices (what MEASURED ran)")
    args = ap.parse_args()

    c = read_config(args.model)
    engines, measured = (ENGINES_BEFORE, MEASURED_BEFORE) if args.before else (ENGINES, MEASURED)
    print(f"model: dim {c['dim']}, hidden {c['hidden']}, {c['layers']} layers, vocab {c['vocab']}; "
          f"position {args.pos}, {'sampling' if args.sampling else 'greedy'}\n")
    print("| engine | model tok/s | vs CPU | measured on a Wii (stories15M bench) |")
    print("|---|---|---|---|")
    cpu = 1 / engine_s(c, "CPU", PARAMS, args.pos, args.sampling)[0]
    for e in engines:
        tps = 1 / engine_s(c, e, PARAMS, args.pos, args.sampling, args.before, not args.unstacked)[0]
        m = measured.get(e)
        print(f"| {e} | {tps:.1f} | {tps / cpu:.1f}x | {f'{m:.2f}' if m else '-'} |")

    print("\nwhere a token's time goes (ms):\n")
    for e in engines:
        total, parts = engine_s(c, e, PARAMS, args.pos, args.sampling, args.before, not args.unstacked)
        print(f"  {e}: {total * 1e3:.1f} ms = " + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in parts.items() if v > 5e-5))


if __name__ == "__main__":
    main()
