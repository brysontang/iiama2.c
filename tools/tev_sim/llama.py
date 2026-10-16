"""
numpy port of source/llm.c: same forward pass, tokenizer and int8 weights, but
every weight matmul goes through a pluggable backend so the GPU (TEV) versions
can be swapped in and compared against the CPU paths.

Weights are read from a llama2.c checkpoint and quantized with the exact
function export_wii.py uses, so the "q8" backend is the CPU path the Wii runs.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from export_wii import pick_group_size, quantize_q8, read_llama2c_checkpoint

MATS = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")


class Weights:
    """fp32 weights plus the int8 (q, per-group scale) copy the Wii uses."""

    def __init__(self, checkpoint, group_size=32):
        cfg, w = read_llama2c_checkpoint(checkpoint)
        self.cfg = cfg
        self.dim, self.hidden = cfg["dim"], cfg["hidden_dim"]
        self.n_layers, self.n_heads, self.n_kv_heads = cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"]
        self.vocab, self.seq_len = cfg["vocab_size"], cfg["seq_len"]
        self.gs = pick_group_size(group_size, self.dim, self.hidden)
        self.rms_att, self.rms_ffn, self.rms_final = w["rms_att"], w["rms_ffn"], w["rms_final"]

        self.f32 = {}  # name -> (rows, cols) float32
        self.q8 = {}   # name -> (q int8 (rows, cols), s float32 (rows, cols/gs))
        for l in range(self.n_layers):
            for k in MATS:
                self._add((k, l), w[k][l])
        self._add("emb", w["emb"])
        self._add("cls", w["emb"] if w["cls"] is None else w["cls"])

    def _add(self, name, m):
        m = np.ascontiguousarray(m, dtype=np.float32)
        q, s, _ = quantize_q8(m, self.gs)
        self.f32[name] = m
        self.q8[name] = (q.reshape(m.shape), s.reshape(m.shape[0], m.shape[1] // self.gs))

    def dequant(self, name):
        q, s = self.q8[name]
        return (q.astype(np.float32).reshape(q.shape[0], -1, self.gs) * s[:, :, None]).reshape(q.shape)


# ----------------------------------------------------------------------------
# backends: callables (name, x) -> y


class F32:
    label = "fp32 CPU"

    def __init__(self, W):
        self.W = W

    def __call__(self, name, x):
        return self.W.f32[name] @ x


class Q8:
    label = "int8 CPU (the Wii today)"

    def __init__(self, W):
        self.mats = {k: W.dequant(k) for k in W.q8}

    def __call__(self, name, x):
        return self.mats[name] @ x


# ----------------------------------------------------------------------------
# forward pass (mirrors llm_forward)


def rmsnorm(x, w):
    ss = np.float32(np.dot(x, x) / x.size + 1e-5)
    return w * (x / np.sqrt(ss))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def quantize_kv(v, hs):
    """One position's keys or values -> int8 per head, scale amax / 127 (llm.c's quantize_kv)."""
    v = v.reshape(-1, hs)
    s = (np.abs(v).max(axis=1) / np.float32(127)).astype(np.float32)
    q = np.clip(np.rint(v / np.where(s == 0, np.float32(1), s)[:, None]), -127, 127).astype(np.float32)
    return q.reshape(-1), s


class Transformer:
    def __init__(self, W, backend):
        self.W, self.mm = W, backend
        kv_dim = W.dim * W.n_kv_heads // W.n_heads
        self.k_cache = np.zeros((W.n_layers, W.seq_len, kv_dim), np.float32)
        self.v_cache = np.zeros_like(self.k_cache)
        # int8 models (everything but the fp32 reference) keep an int8 KV cache, one scale per head and position
        self.kv8 = not isinstance(backend, F32)
        self.k_scale = np.zeros((W.n_layers, W.seq_len, W.n_kv_heads), np.float32)
        self.v_scale = np.zeros_like(self.k_scale)
        hs = W.dim // W.n_heads
        freqs = 1.0 / np.power(10000.0, (np.arange(0, W.dim, 2) % hs) / hs)
        self.rope_freqs = freqs.astype(np.float32)
        # the Wii stores the embedding table as int8 too; only the fp32 reference uses fp32
        self.emb = W.f32["emb"] if isinstance(backend, F32) else W.dequant("emb")

    def rope(self, v, pos):
        n = v.size
        ang = pos * self.rope_freqs[: n // 2]
        c, s = np.cos(ang), np.sin(ang)
        v0, v1 = v[0::2].copy(), v[1::2].copy()
        v[0::2] = v0 * c - v1 * s
        v[1::2] = v0 * s + v1 * c

    def forward(self, token, pos):
        W, mm = self.W, self.mm
        dim, hs = W.dim, W.dim // W.n_heads
        kv_mul = W.n_heads // W.n_kv_heads
        x = self.emb[token].copy()
        for l in range(W.n_layers):
            xb = rmsnorm(x, W.rms_att[l])
            q = mm(("wq", l), xb)
            k = mm(("wk", l), xb)
            v = mm(("wv", l), xb)
            self.rope(q, pos)
            self.rope(k, pos)
            if self.kv8:
                (self.k_cache[l, pos], self.k_scale[l, pos]), (self.v_cache[l, pos], self.v_scale[l, pos]) = (
                    quantize_kv(k, hs), quantize_kv(v, hs))
            else:
                self.k_cache[l, pos], self.v_cache[l, pos] = k, v
            out = np.empty(dim, np.float32)
            for h in range(W.n_heads):
                kh = h // kv_mul
                K = self.k_cache[l, : pos + 1, kh * hs:(kh + 1) * hs]
                V = self.v_cache[l, : pos + 1, kh * hs:(kh + 1) * hs]
                scores = K @ q[h * hs:(h + 1) * hs]
                if self.kv8:
                    scores = scores * self.k_scale[l, : pos + 1, kh]
                att = softmax(scores / np.sqrt(hs))
                if self.kv8:
                    att = att * self.v_scale[l, : pos + 1, kh]
                out[h * hs:(h + 1) * hs] = att @ V
            x = x + mm(("wo", l), out)
            xb = rmsnorm(x, W.rms_ffn[l])
            h1, h3 = mm(("w1", l), xb), mm(("w3", l), xb)
            x = x + mm(("w2", l), (h1 / (1.0 + np.exp(-h1))) * h3)
        x = rmsnorm(x, W.rms_final)
        return mm("cls", x)


# ----------------------------------------------------------------------------
# tokenizer (llama2.c tokenizer.bin, same BPE as llm_encode)


class Tokenizer:
    def __init__(self, path):
        with open(path, "rb") as f:
            raw = f.read()
        self.max_len = int(np.frombuffer(raw, "<i4", 1, 0)[0])
        pos, self.vocab, self.scores = 4, [], []
        while pos < len(raw):
            score = float(np.frombuffer(raw, "<f4", 1, pos)[0])
            n = int(np.frombuffer(raw, "<i4", 1, pos + 4)[0])
            self.vocab.append(raw[pos + 8:pos + 8 + n])
            self.scores.append(score)
            pos += 8 + n
        self.lookup = {p: i for i, p in enumerate(self.vocab)}

    def encode(self, text, bos=True):
        toks = [1] if bos else []
        if text:
            toks.append(self.lookup[b" "])
        data = text.encode()
        i = 0
        while i < len(data):  # one token per UTF-8 codepoint, else byte fallback
            j = i + 1
            while j < len(data) and (data[j] & 0xC0) == 0x80 and j - i < 4:
                j += 1
            piece = data[i:j]
            if piece in self.lookup:
                toks.append(self.lookup[piece])
            else:
                toks.extend(b + 3 for b in piece)
            i = j
        while True:
            best, best_id, best_i = -1e10, -1, -1
            for i in range(len(toks) - 1):
                pid = self.lookup.get(self.vocab[toks[i]] + self.vocab[toks[i + 1]])
                if pid is not None and self.scores[pid] > best:
                    best, best_id, best_i = self.scores[pid], pid, i
            if best_i < 0:
                break
            toks[best_i:best_i + 2] = [best_id]
        return toks

    def decode(self, prev, tok):
        piece = self.vocab[tok]
        if prev == 1 and piece.startswith(b" "):
            piece = piece[1:]
        if piece.startswith(b"<0x") and piece.endswith(b">") and len(piece) == 6:
            piece = bytes([int(piece[3:5], 16)])
        return piece


# ----------------------------------------------------------------------------
# generation


def sample(logits, temperature, topp, rng):
    if temperature == 0:
        return int(np.argmax(logits))
    p = softmax(logits / temperature)
    if 0 < topp < 1:
        order = np.argsort(-p, kind="stable")
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, topp, side="right")) + 1]
        q = p[keep] / p[keep].sum()
        return int(keep[np.searchsorted(np.cumsum(q), rng.random() * 0.9999999, side="right")])
    return int(np.searchsorted(np.cumsum(p), rng.random() * 0.9999999, side="right"))


def generate(W, backend, tok, prompt, steps, temperature=0.0, topp=0.9, seed=0, on_logits=None):
    """Returns (token list incl. BOS, decoded text)."""
    model = Transformer(W, backend)
    rng = np.random.default_rng(seed)
    ptoks = tok.encode(prompt)
    toks, out = [ptoks[0]], b""
    for pos in range(min(steps, W.seq_len)):
        logits = model.forward(toks[-1], pos)
        if on_logits:
            on_logits(pos, logits)
        nxt = ptoks[pos + 1] if pos + 1 < len(ptoks) else sample(logits, temperature, topp, rng)
        if nxt == 1:
            break
        out += tok.decode(toks[-1], nxt)
        toks.append(nxt)
    return toks, out.decode("utf-8", errors="replace")
