#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
Convert a llama2.c checkpoint (+ tokenizer.bin) into the big-endian format
that iiama2.c reads on the Wii.

    uv run tools/export_wii.py stories15M.bin tokenizer.bin sd/llama/
    uv run tools/export_wii.py stories42M.bin tokenizer.bin sd/llama/ --dtype q8

Outputs <outdir>/model.bin and <outdir>/tokenizer.bin. Copy that folder to the
root of your SD card as /llama/.

Why a converter at all: the Wii's Broadway CPU is big-endian PowerPC, and
llama2.c checkpoints are written little-endian. Reading them as-is gives you
"beautiful garbage". This script byteswaps once on your PC so the Wii never
has to, and can optionally int8-quantize the matrices to fit in 88 MB.

Only needs numpy.
"""
import argparse
import os
import struct
import sys

import numpy as np

MAGIC_MODEL = b"WLLM"
MAGIC_TOK = b"WTOK"
VERSION = 1
DTYPE_F32 = 0
DTYPE_Q8 = 1

BE_F32 = np.dtype(">f4")
BE_I32 = np.dtype(">i4")


def read_llama2c_checkpoint(path):
    """Parse a legacy (version 0) llama2.c .bin checkpoint."""
    with open(path, "rb") as f:
        header = f.read(7 * 4)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("<7i", header)
        data = np.frombuffer(f.read(), dtype="<f4")

    if dim <= 0 or n_layers <= 0 or n_heads <= 0 or dim % n_heads != 0:
        sys.exit(f"{path} doesn't look like a llama2.c checkpoint (header: {header!r})")

    shared = vocab_size > 0  # negative vocab = unshared classifier (llama2.c convention)
    vocab_size = abs(vocab_size)
    head_size = dim // n_heads
    kv_dim = dim * n_kv_heads // n_heads
    L = n_layers

    off = 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        if off + n > data.size:
            sys.exit(f"{path} is truncated (needed {n} more floats at offset {off})")
        t = data[off:off + n].reshape(shape)
        off += n
        return t

    w = {}
    w["emb"] = take(vocab_size, dim)
    w["rms_att"] = take(L, dim)
    w["wq"] = take(L, dim, dim)
    w["wk"] = take(L, kv_dim, dim)
    w["wv"] = take(L, kv_dim, dim)
    w["wo"] = take(L, dim, dim)
    w["rms_ffn"] = take(L, dim)
    w["w1"] = take(L, hidden_dim, dim)
    w["w2"] = take(L, dim, hidden_dim)
    w["w3"] = take(L, hidden_dim, dim)
    w["rms_final"] = take(dim)
    take(seq_len * head_size // 2)  # legacy freq_cis_real, unused
    take(seq_len * head_size // 2)  # legacy freq_cis_imag, unused
    w["cls"] = None if shared else take(vocab_size, dim)

    cfg = {"dim": dim, "hidden_dim": hidden_dim, "n_layers": n_layers, "n_heads": n_heads,
           "n_kv_heads": n_kv_heads, "vocab_size": vocab_size, "seq_len": seq_len, "shared": shared}
    return cfg, w


def pick_group_size(requested, dim, hidden_dim):
    """Largest power of two <= requested that divides every matmul input width."""
    gs = 1
    while gs * 2 <= requested and dim % (gs * 2) == 0 and hidden_dim % (gs * 2) == 0:
        gs *= 2
    return gs


def quantize_q8(m, gs):
    """Symmetric int8, one fp32 scale per `gs` consecutive weights along each row."""
    flat = m.astype(np.float32).reshape(-1, gs)
    scale = np.abs(flat).max(axis=1) / 127.0
    safe = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(flat / safe[:, None]), -127, 127).astype(np.int8)
    err = np.abs(q.astype(np.float32) * scale[:, None] - flat).max()
    return q.reshape(-1), scale.astype(np.float32), float(err)


def write_tensor(f, m, dtype, gs, stats):
    if dtype == DTYPE_F32:
        f.write(np.ascontiguousarray(m, dtype=np.float32).astype(BE_F32).tobytes())
        stats["bytes"] += m.size * 4
    else:
        q, s, err = quantize_q8(m, gs)
        f.write(q.tobytes())
        f.write(s.astype(BE_F32).tobytes())
        stats["bytes"] += q.size + s.size * 4
        stats["max_err"] = max(stats["max_err"], err)


def export_model(cfg, w, out_path, dtype, gs):
    stats = {"bytes": 0, "max_err": 0.0}
    with open(out_path, "wb") as f:
        f.write(MAGIC_MODEL)
        f.write(np.array([VERSION, dtype, gs, cfg["dim"], cfg["hidden_dim"], cfg["n_layers"],
                          cfg["n_heads"], cfg["n_kv_heads"], cfg["vocab_size"], cfg["seq_len"],
                          int(cfg["shared"])], dtype=BE_I32).tobytes())
        # norm weights stay fp32 (tiny, and precision-sensitive)
        for k in ("rms_att", "rms_ffn", "rms_final"):
            f.write(w[k].astype(BE_F32).tobytes())
            stats["bytes"] += w[k].size * 4
        # matrices, one at a time so the Wii can allocate (and place) each separately
        write_tensor(f, w["emb"], dtype, gs, stats)
        for l in range(cfg["n_layers"]):
            for k in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
                write_tensor(f, w[k][l], dtype, gs, stats)
        if w["cls"] is not None:
            write_tensor(f, w["cls"], dtype, gs, stats)
    return stats


def export_tokenizer(in_path, out_path):
    """llama2.c tokenizer.bin (LE, no count) -> WTOK (BE, with count)."""
    with open(in_path, "rb") as f:
        raw = f.read()
    (max_len,) = struct.unpack_from("<i", raw, 0)
    pos = 4
    entries = []
    while pos < len(raw):
        score, n = struct.unpack_from("<fi", raw, pos)
        pos += 8
        entries.append((score, raw[pos:pos + n]))
        pos += n
    with open(out_path, "wb") as f:
        f.write(MAGIC_TOK)
        f.write(struct.pack(">ii", len(entries), max_len))
        for score, piece in entries:
            f.write(struct.pack(">fi", score, len(piece)))
            f.write(piece)
    return len(entries)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="llama2.c model .bin (e.g. stories15M.bin)")
    ap.add_argument("tokenizer", help="llama2.c tokenizer.bin")
    ap.add_argument("outdir", help="output folder (copy it to SD:/llama/)")
    ap.add_argument("--dtype", choices=["q8", "f32"], default="q8",
                    help="q8 = int8 weights, ~4x smaller and faster on the Wii (default); f32 = exact")
    ap.add_argument("--group-size", type=int, default=32, help="q8 quantization group size (default 32)")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="max tokens per story (default 256, about one TV screen of text). The KV cache "
                         "grows with this, e.g. stories42M needs ~34 MB of cache at its native 1024.")
    args = ap.parse_args()

    cfg, w = read_llama2c_checkpoint(args.checkpoint)
    if args.seq_len > 0:
        cfg["seq_len"] = min(cfg["seq_len"], args.seq_len)  # shorter context = smaller KV cache
    dtype = DTYPE_Q8 if args.dtype == "q8" else DTYPE_F32
    gs = pick_group_size(args.group_size, cfg["dim"], cfg["hidden_dim"]) if dtype == DTYPE_Q8 else 0

    os.makedirs(args.outdir, exist_ok=True)
    stats = export_model(cfg, w, os.path.join(args.outdir, "model.bin"), dtype, gs)
    n_tok = export_tokenizer(args.tokenizer, os.path.join(args.outdir, "tokenizer.bin"))
    if n_tok != cfg["vocab_size"]:
        sys.exit(f"tokenizer has {n_tok} tokens but model vocab is {cfg['vocab_size']}: wrong tokenizer?")

    kv_dim = cfg["dim"] * cfg["n_kv_heads"] // cfg["n_heads"]
    kv_bytes = 2 * cfg["n_layers"] * cfg["seq_len"] * kv_dim * 4
    total_mb = (stats["bytes"] + kv_bytes) / 2**20
    print(f"config  : {cfg}")
    print(f"weights : {args.dtype}" + (f", group size {gs}, max abs error {stats['max_err']:.2e}" if gs else ""))
    print(f"memory  : ~{total_mb:.1f} MB on the Wii (weights + KV cache)")
    if total_mb > 70:
        print("warning : that's more than the Wii can realistically hold (88 MB total, ~70 MB usable)."
              + (" Try --dtype q8." if dtype == DTYPE_F32 else ""))
    print(f"wrote   : {args.outdir}/model.bin, {args.outdir}/tokenizer.bin")


if __name__ == "__main__":
    main()
