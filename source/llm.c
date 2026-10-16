// llm.c: transformer inference for the Wii, ported from Karpathy's llama2.c (MIT).
// See llm.h for what changed and why.

#include "llm.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ----------------------------------------------------------------------------
// errors

static char g_err[256];

static int fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
    return -1;
}

const char *llm_error(void) { return g_err; }

int llm_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
    return -1;
}

// ----------------------------------------------------------------------------
// big-endian file reading
//
// Files from export_wii.py are big-endian. On the Wii (big-endian PowerPC) the
// bytes go straight into memory. On a little-endian PC we swap after reading,
// which lets you test the exact same files on your desktop.

static int host_is_little_endian(void) {
    const uint16_t one = 1;
    return *(const uint8_t *)&one == 1;
}

static void swap32(void *buf, size_t count) {
    uint8_t *p = (uint8_t *)buf;
    for (size_t i = 0; i < count; i++, p += 4) {
        uint8_t a = p[0], b = p[1];
        p[0] = p[3]; p[1] = p[2]; p[2] = b; p[3] = a;
    }
}

typedef struct {
    FILE *f;
    size_t done, total;
    LlmProgressFn progress;
    void *user;
} Reader;

#define READ_CHUNK (256 * 1024)

static int read_bytes(Reader *r, void *dst, size_t n) {
    uint8_t *p = (uint8_t *)dst;
    while (n > 0) {
        size_t want = n < READ_CHUNK ? n : READ_CHUNK;
        if (fread(p, 1, want, r->f) != want) return fail("model file is truncated or unreadable");
        p += want; n -= want; r->done += want;
        if (r->progress) r->progress(r->done, r->total, r->user);
    }
    return 0;
}

static int read_be32(Reader *r, void *dst, size_t count) {
    if (read_bytes(r, dst, count * 4)) return -1;
    if (host_is_little_endian()) swap32(dst, count);
    return 0;
}

// ----------------------------------------------------------------------------
// model loading

static void *track_alloc(LlmModel *m, size_t bytes, const char *what) {
    void *p = calloc(1, bytes ? bytes : 1);
    if (!p) {
        fail("out of memory allocating %s (%lu KB, %lu KB already used)", what,
             (unsigned long)(bytes / 1024), (unsigned long)(m->bytes / 1024));
        return NULL;
    }
    m->bytes += bytes;
    return p;
}

static int alloc_matrix(LlmModel *m, LlmMatrix *mat, size_t rows, size_t cols, const char *what) {
    size_t n = rows * cols;
    if (m->dtype == LLM_DTYPE_F32) {
        mat->f = (float *)track_alloc(m, n * sizeof(float), what);
        return mat->f ? 0 : -1;
    }
    mat->q = (int8_t *)track_alloc(m, n, what);
    if (!mat->q) return -1;
    mat->s = (float *)track_alloc(m, (n / m->gs) * sizeof(float), what);
    return mat->s ? 0 : -1;
}

static int read_matrix(LlmModel *m, Reader *r, LlmMatrix *mat, size_t rows, size_t cols) {
    size_t n = rows * cols;
    if (m->dtype == LLM_DTYPE_F32) return read_be32(r, mat->f, n);
    if (read_bytes(r, mat->q, n)) return -1;
    return read_be32(r, mat->s, n / m->gs);
}

static void free_matrix(LlmMatrix *mat) {
    free(mat->f);
    free(mat->q);
    free(mat->s);
    memset(mat, 0, sizeof(*mat));
}

void llm_free_model(LlmModel *m) {
    if (m->layers) {
        for (int l = 0; l < m->c.n_layers; l++) {
            LlmLayer *L = &m->layers[l];
            free_matrix(&L->wq); free_matrix(&L->wk); free_matrix(&L->wv); free_matrix(&L->wo);
            free_matrix(&L->w1); free_matrix(&L->w2); free_matrix(&L->w3);
        }
        free(m->layers);
    }
    if (!m->shared) free_matrix(&m->cls);
    free_matrix(&m->emb);
    free(m->rms_att); free(m->rms_ffn); free(m->rms_final);
    free(m->x); free(m->xb); free(m->xb2); free(m->hb); free(m->q); // hb2, k and v are in hb's and q's blocks
    free(m->att); free(m->logits); free(m->key_cache); free(m->value_cache);
    free(m->key_cache8); free(m->value_cache8); free(m->key_scale); free(m->value_scale);
    free(m->rope);
    memset(m, 0, sizeof(*m));
}

int llm_load_model(LlmModel *m, const char *path, LlmProgressFn progress, void *user) {
    memset(m, 0, sizeof(*m));
    Reader r = {0};
    r.progress = progress;
    r.user = user;
    r.f = fopen(path, "rb");
    if (!r.f) return fail("couldn't open %s", path);
    fseek(r.f, 0, SEEK_END);
    r.total = (size_t)ftell(r.f);
    fseek(r.f, 0, SEEK_SET);

    int rc = -1;
    char magic[4];
    int32_t h[11];
    if (read_bytes(&r, magic, 4) || memcmp(magic, "WLLM", 4) != 0) {
        fail("%s isn't an iiama2.c model (run tools/export_wii.py on it first)", path);
        goto done;
    }
    if (read_be32(&r, h, 11)) goto done;
    if (h[0] != 1) { fail("unsupported model file version %d", (int)h[0]); goto done; }
    m->dtype = h[1];
    m->gs = h[2];
    LlmConfig *c = &m->c;
    c->dim = h[3]; c->hidden_dim = h[4]; c->n_layers = h[5]; c->n_heads = h[6];
    c->n_kv_heads = h[7]; c->vocab_size = h[8]; c->seq_len = h[9];
    m->shared = h[10];
    if (c->dim <= 0 || c->n_heads <= 0 || c->n_kv_heads <= 0 || c->n_layers <= 0 ||
        c->dim % c->n_heads || c->n_heads % c->n_kv_heads ||
        (m->dtype == LLM_DTYPE_Q8 && (m->gs <= 0 || c->dim % m->gs || c->hidden_dim % m->gs)) ||
        (m->dtype != LLM_DTYPE_F32 && m->dtype != LLM_DTYPE_Q8)) {
        fail("model header looks corrupt");
        goto done;
    }

    size_t dim = c->dim, hid = c->hidden_dim, L = c->n_layers;
    size_t kv_dim = dim * c->n_kv_heads / c->n_heads;

    // Allocation order matters on the Wii: malloc fills MEM1 (24 MB) first and
    // moves to MEM2 (64 MB) once a request doesn't fit. Allocating the many small
    // things first packs MEM1 tightly; the huge embedding table goes last, into MEM2.
#define ALLOC(ptr, n, what) if (!((ptr) = (float *)track_alloc(m, (n) * sizeof(float), what))) goto done
    ALLOC(m->x, dim, "activations");
    ALLOC(m->xb, dim, "activations");
    ALLOC(m->xb2, dim, "activations");
    // hb then hb2, and for int8 models q, k, v, in one block each: the GPU's stacked matmuls write them in one go
    ALLOC(m->hb, 2 * hid, "activations");
    m->hb2 = m->hb + hid;
    ALLOC(m->q, dim + (m->dtype == LLM_DTYPE_Q8 ? 2 * kv_dim : 0), "activations");
    ALLOC(m->att, (size_t)c->n_heads * c->seq_len, "attention buffer");
    ALLOC(m->logits, c->vocab_size, "logits");
    if (m->dtype == LLM_DTYPE_Q8) {
        size_t n = L * c->seq_len * kv_dim;
        if (!(m->key_cache8 = (int8_t *)track_alloc(m, n, "KV cache")) ||
            !(m->value_cache8 = (int8_t *)track_alloc(m, n, "KV cache")))
            goto done;
        ALLOC(m->key_scale, L * c->seq_len * c->n_kv_heads, "KV cache");
        ALLOC(m->value_scale, L * c->seq_len * c->n_kv_heads, "KV cache");
        m->k = m->q + dim;
        m->v = m->k + kv_dim;
    } else {
        ALLOC(m->key_cache, L * c->seq_len * kv_dim, "KV cache");
        ALLOC(m->value_cache, L * c->seq_len * kv_dim, "KV cache");
    }
    ALLOC(m->rope, dim / c->n_heads, "activations");
    ALLOC(m->rms_att, L * dim, "norm weights");
    ALLOC(m->rms_ffn, L * dim, "norm weights");
    ALLOC(m->rms_final, dim, "norm weights");
#undef ALLOC
    m->layers = (LlmLayer *)track_alloc(m, L * sizeof(LlmLayer), "layer table");
    if (!m->layers) goto done;
    for (size_t l = 0; l < L; l++) {
        LlmLayer *Ly = &m->layers[l];
        if (alloc_matrix(m, &Ly->wq, dim, dim, "wq") || alloc_matrix(m, &Ly->wk, kv_dim, dim, "wk") ||
            alloc_matrix(m, &Ly->wv, kv_dim, dim, "wv") || alloc_matrix(m, &Ly->wo, dim, dim, "wo") ||
            alloc_matrix(m, &Ly->w1, hid, dim, "w1") || alloc_matrix(m, &Ly->w2, dim, hid, "w2") ||
            alloc_matrix(m, &Ly->w3, hid, dim, "w3"))
            goto done;
    }
    if (!m->shared && alloc_matrix(m, &m->cls, c->vocab_size, dim, "classifier")) goto done;
    if (alloc_matrix(m, &m->emb, c->vocab_size, dim, "token embeddings")) goto done;
    if (m->shared) m->cls = m->emb;

    // now read everything in file order
    if (read_be32(&r, m->rms_att, L * dim) || read_be32(&r, m->rms_ffn, L * dim) ||
        read_be32(&r, m->rms_final, dim))
        goto done;
    if (read_matrix(m, &r, &m->emb, c->vocab_size, dim)) goto done;
    for (size_t l = 0; l < L; l++) {
        LlmLayer *Ly = &m->layers[l];
        if (read_matrix(m, &r, &Ly->wq, dim, dim) || read_matrix(m, &r, &Ly->wk, kv_dim, dim) ||
            read_matrix(m, &r, &Ly->wv, kv_dim, dim) || read_matrix(m, &r, &Ly->wo, dim, dim) ||
            read_matrix(m, &r, &Ly->w1, hid, dim) || read_matrix(m, &r, &Ly->w2, dim, hid) ||
            read_matrix(m, &r, &Ly->w3, hid, dim))
            goto done;
    }
    if (!m->shared && read_matrix(m, &r, &m->cls, c->vocab_size, dim)) goto done;
    rc = 0;

done:
    fclose(r.f);
    if (rc != 0) llm_free_model(m);
    return rc;
}

// ----------------------------------------------------------------------------
// per-token profile (LlmModel.prof)
//
// Each part of a token ends with a lap, which charges it the time since the previous lap. One counter
// read separates two parts, and the parts add up to the whole token.

static void prof_start(LlmProfile *pf) {
    if (pf) pf->last = pf->now();
}

static void prof_lap(LlmProfile *pf, int part) {
    if (!pf) return;
    uint32_t now = pf->now();
    pf->ticks[part] += now - pf->last;
    pf->last = now;
}

// ----------------------------------------------------------------------------
// math
//
// Broadway (PowerPC 750CL) has no fast int->float instruction, so converting
// each int8 weight with a cast costs a store/load round trip. A 256-entry
// lookup table turns it into a single indexed load.

static float g_i8_to_f32[256];
static int g_lut_ready = 0;

static void init_lut(void) {
    if (g_lut_ready) return;
    for (int i = 0; i < 256; i++) g_i8_to_f32[i] = (float)(int8_t)(uint8_t)i;
    g_lut_ready = 1;
}

#define I8F(v) g_i8_to_f32[(uint8_t)(v)]

// One head's key or value vector -> int8, scale amax / 127. Must match quantize_kv in tools/tev_sim/llama.py.
static void quantize_kv(const float *x, int n, int8_t *q, float *scale) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++)
        if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
    float s = amax / 127.0f, d = s == 0.0f ? 1.0f : s;
    *scale = s;
    for (int i = 0; i < n; i++) {
        float r = (x[i] / d + 12582912.0f) - 12582912.0f; // rintf for |r| < 2^22 (round half to even), no call
        q[i] = (int8_t)(r > 127.0f ? 127 : r < -127.0f ? -127 : r);
    }
}

static void matmul_f32(float *xout, const float *x, const float *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    for (int i = 0; i < d; i++) {
        const float *row = w + (size_t)i * n;
#ifdef LLM_EXACT_MATMUL
        // same summation order as upstream llama2.c, for bit-exact comparisons
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += row[j] * x[j];
        xout[i] = val;
#else
        // 4 independent accumulators keep the 750's FPU pipeline busy
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            a0 += row[j] * x[j];
            a1 += row[j + 1] * x[j + 1];
            a2 += row[j + 2] * x[j + 2];
            a3 += row[j + 3] * x[j + 3];
        }
        for (; j < n; j++) a0 += row[j] * x[j];
        xout[i] = (a0 + a1) + (a2 + a3);
#endif
    }
}

static void matmul_q8_scalar(float *xout, const float *x, const int8_t *w, const float *s, int n, int d, int gs) {
    int groups = n / gs;
    for (int i = 0; i < d; i++) {
        const int8_t *row = w + (size_t)i * n;
        const float *rs = s + (size_t)i * groups;
        float val = 0.0f;
        for (int g = 0; g < groups; g++) {
            const int8_t *r = row + g * gs;
            const float *xx = x + g * gs;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            int j = 0;
            for (; j + 4 <= gs; j += 4) {
                a0 += I8F(r[j]) * xx[j];
                a1 += I8F(r[j + 1]) * xx[j + 1];
                a2 += I8F(r[j + 2]) * xx[j + 2];
                a3 += I8F(r[j + 3]) * xx[j + 3];
            }
            for (; j < gs; j++) a0 += I8F(r[j]) * xx[j];
            val += ((a0 + a1) + (a2 + a3)) * rs[g];
        }
        xout[i] = val;
    }
}

#ifdef GEKKO
#define GQR_S8 0x00060006u // load and store type s8, scale 0

// One weight group of two rows with Broadway's paired singles: psq_l turns two int8 weights into floats
// (GQR7 is set to s8) or loads two floats of x (GQR0, plain floats), and each ps_madd is two of the scalar
// loop's fmadds, lane for lane. acc gets each row's four accumulators, a0..a3, for the caller to combine
// the way matmul_q8_scalar does, so the results are the same floats with a third of the loads.
static inline void q8_group2(const int8_t *r0, const int8_t *r1, const float *xx, int quads, float *acc) {
    static const float zero[2] __attribute__((aligned(8))) = {0.0f, 0.0f};
    double a01, a23, b01, b23, x01, x23, w01, w23, v01, v23;
    __asm__ volatile("psq_l %[a01], 0(%[zero]), 0, 0\n\t"
                     "ps_mr %[a23], %[a01]\n\t"
                     "ps_mr %[b01], %[a01]\n\t"
                     "ps_mr %[b23], %[a01]\n\t"
                     "mtctr %[quads]\n"
                     "1:\n\t"
                     "psq_l %[x01], 0(%[xx]), 0, 0\n\t"
                     "psq_l %[x23], 8(%[xx]), 0, 0\n\t"
                     "psq_l %[w01], 0(%[r0]), 0, 7\n\t"
                     "psq_l %[w23], 2(%[r0]), 0, 7\n\t"
                     "psq_l %[v01], 0(%[r1]), 0, 7\n\t"
                     "psq_l %[v23], 2(%[r1]), 0, 7\n\t"
                     "addi %[xx], %[xx], 16\n\t"
                     "addi %[r0], %[r0], 4\n\t"
                     "addi %[r1], %[r1], 4\n\t"
                     "ps_madd %[a01], %[w01], %[x01], %[a01]\n\t"
                     "ps_madd %[a23], %[w23], %[x23], %[a23]\n\t"
                     "ps_madd %[b01], %[v01], %[x01], %[b01]\n\t"
                     "ps_madd %[b23], %[v23], %[x23], %[b23]\n\t"
                     "bdnz 1b\n\t"
                     "psq_st %[a01], 0(%[acc]), 0, 0\n\t"
                     "psq_st %[a23], 8(%[acc]), 0, 0\n\t"
                     "psq_st %[b01], 16(%[acc]), 0, 0\n\t"
                     "psq_st %[b23], 24(%[acc]), 0, 0\n\t"
                     : [a01] "=&f"(a01), [a23] "=&f"(a23), [b01] "=&f"(b01), [b23] "=&f"(b23), [x01] "=&f"(x01),
                       [x23] "=&f"(x23), [w01] "=&f"(w01), [w23] "=&f"(w23), [v01] "=&f"(v01), [v23] "=&f"(v23),
                       [xx] "+b"(xx), [r0] "+b"(r0), [r1] "+b"(r1)
                     : [zero] "b"(zero), [quads] "r"(quads), [acc] "b"(acc)
                     : "ctr", "memory");
}

// Four attention scores at once, for positions t..t+3 of an int8 key cache: each lane still adds
// q[i] * k[i] for i = 0, 1, 2... in order, the scalar loop's fmadds one for one. psq_l loads two keys of
// one position (GQR7 = s8), and merges pair up the positions. Raw dot products go to out[0..3].
static inline void qk4_s8(const float *q, const int8_t *k, int kv_dim, int pairs, float *out) {
    static const float zero[2] __attribute__((aligned(8))) = {0.0f, 0.0f};
    double s01, s23, qq, k0, k1, k2, k3, m0, m1, m2, m3;
    int o1 = kv_dim, o2 = 2 * kv_dim, o3 = 3 * kv_dim;
    __asm__ volatile("psq_l %[s01], 0(%[zero]), 0, 0\n\t"
                     "ps_mr %[s23], %[s01]\n\t"
                     "mtctr %[pairs]\n"
                     "1:\n\t"
                     "psq_l %[qq], 0(%[q]), 0, 0\n\t"
                     "psq_l %[k0], 0(%[k]), 0, 7\n\t"
                     "psq_lx %[k1], %[k], %[o1], 0, 7\n\t"
                     "psq_lx %[k2], %[k], %[o2], 0, 7\n\t"
                     "psq_lx %[k3], %[k], %[o3], 0, 7\n\t"
                     "addi %[q], %[q], 8\n\t"
                     "addi %[k], %[k], 2\n\t"
                     "ps_merge00 %[m0], %[k0], %[k1]\n\t"
                     "ps_merge00 %[m2], %[k2], %[k3]\n\t"
                     "ps_merge11 %[m1], %[k0], %[k1]\n\t"
                     "ps_merge11 %[m3], %[k2], %[k3]\n\t"
                     "ps_madds0 %[s01], %[m0], %[qq], %[s01]\n\t"
                     "ps_madds0 %[s23], %[m2], %[qq], %[s23]\n\t"
                     "ps_madds1 %[s01], %[m1], %[qq], %[s01]\n\t"
                     "ps_madds1 %[s23], %[m3], %[qq], %[s23]\n\t"
                     "bdnz 1b\n\t"
                     "psq_st %[s01], 0(%[out]), 0, 0\n\t"
                     "psq_st %[s23], 8(%[out]), 0, 0\n\t"
                     : [s01] "=&f"(s01), [s23] "=&f"(s23), [qq] "=&f"(qq), [k0] "=&f"(k0), [k1] "=&f"(k1),
                       [k2] "=&f"(k2), [k3] "=&f"(k3), [m0] "=&f"(m0), [m1] "=&f"(m1), [m2] "=&f"(m2),
                       [m3] "=&f"(m3), [q] "+b"(q), [k] "+b"(k)
                     : [zero] "b"(zero), [pairs] "r"(pairs), [o1] "r"(o1), [o2] "r"(o2), [o3] "r"(o3),
                       [out] "b"(out)
                     : "ctr", "memory");
}

// xb[i] += a * v[i] for i < 8 * octets, eight at a time in four pairs: the scalar loop's fmadds lane
// for lane (GQR7 = s8).
static inline void axpy_s8(float *xb, const int8_t *v, float a, int octets) {
    double x0, x1, x2, x3, v0, v1, v2, v3;
    __asm__ volatile("mtctr %[octets]\n"
                     "1:\n\t"
                     "psq_l %[x0], 0(%[xb]), 0, 0\n\t"
                     "psq_l %[x1], 8(%[xb]), 0, 0\n\t"
                     "psq_l %[x2], 16(%[xb]), 0, 0\n\t"
                     "psq_l %[x3], 24(%[xb]), 0, 0\n\t"
                     "psq_l %[v0], 0(%[v]), 0, 7\n\t"
                     "psq_l %[v1], 2(%[v]), 0, 7\n\t"
                     "psq_l %[v2], 4(%[v]), 0, 7\n\t"
                     "psq_l %[v3], 6(%[v]), 0, 7\n\t"
                     "ps_madds0 %[x0], %[v0], %[a], %[x0]\n\t"
                     "ps_madds0 %[x1], %[v1], %[a], %[x1]\n\t"
                     "ps_madds0 %[x2], %[v2], %[a], %[x2]\n\t"
                     "ps_madds0 %[x3], %[v3], %[a], %[x3]\n\t"
                     "psq_st %[x0], 0(%[xb]), 0, 0\n\t"
                     "psq_st %[x1], 8(%[xb]), 0, 0\n\t"
                     "psq_st %[x2], 16(%[xb]), 0, 0\n\t"
                     "psq_st %[x3], 24(%[xb]), 0, 0\n\t"
                     "addi %[xb], %[xb], 32\n\t"
                     "addi %[v], %[v], 8\n\t"
                     "bdnz 1b\n\t"
                     : [x0] "=&f"(x0), [x1] "=&f"(x1), [x2] "=&f"(x2), [x3] "=&f"(x3), [v0] "=&f"(v0),
                       [v1] "=&f"(v1), [v2] "=&f"(v2), [v3] "=&f"(v3), [xb] "+b"(xb), [v] "+b"(v)
                     : [a] "f"(a), [octets] "r"(octets)
                     : "ctr", "memory");
}
#endif

static void matmul_q8(float *xout, const float *x, const int8_t *w, const float *s, int n, int d, int gs) {
#ifdef GEKKO
    // paired loads of x want 8-byte alignment; malloc gives it, and groups keep it when gs % 4 == 0
    if (gs % 4 == 0 && ((uintptr_t)x & 7) == 0) {
        int groups = n / gs;
        float acc[8] __attribute__((aligned(8)));
        __asm__ volatile("mtspr 919, %0" : : "r"(GQR_S8)); // GQR7: load s8, unscaled
        for (int i = 0; i < d; i += 2) {
            int i1 = i + 1 < d ? i + 1 : i; // an odd last row goes through twice
            const int8_t *r0 = w + (size_t)i * n, *r1 = w + (size_t)i1 * n;
            const float *s0 = s + (size_t)i * groups, *s1 = s + (size_t)i1 * groups;
            float val0 = 0.0f, val1 = 0.0f;
            for (int g = 0; g < groups; g++) {
                q8_group2(r0 + g * gs, r1 + g * gs, x + g * gs, gs / 4, acc);
                val0 += ((acc[0] + acc[1]) + (acc[2] + acc[3])) * s0[g];
                val1 += ((acc[4] + acc[5]) + (acc[6] + acc[7])) * s1[g];
            }
            xout[i] = val0;
            xout[i1] = val1;
        }
        return;
    }
#endif
    matmul_q8_scalar(xout, x, w, s, n, d, gs);
}

int llm_check_paired_singles(const LlmModel *m, int *total) {
    *total = 0;
#ifdef GEKKO
    if (m->dtype != LLM_DTYPE_Q8) return 0;
    init_lut(); // the scalar loop's int8 -> float table, normally built by the first forward pass
    const LlmConfig *c = &m->c;
    int n = c->dim > c->hidden_dim ? c->dim : c->hidden_dim, bad = 0;
    float *x = (float *)malloc(n * sizeof(float)), *a = (float *)malloc(2 * 257 * sizeof(float)), *b = a + 257;
    if (!x || !a) {
        free(x);
        free(a);
        return -1;
    }
    for (int j = 0; j < n; j++) x[j] = sinf(j * 0.37f) * (1.0f + j % 7);
    const LlmMatrix *mats[] = {&m->layers[0].wq, &m->layers[0].w2, &m->cls};
    const int cols[] = {c->dim, c->hidden_dim, c->dim}, all[] = {c->dim, c->dim, c->vocab_size};
    for (int k = 0; k < 3; k++) {
        int rows = all[k] < 257 ? all[k] : 257;
        rows -= rows % 2 == 0; // odd, so the last row takes the single-row path
        matmul_q8(a, x, mats[k]->q, mats[k]->s, cols[k], rows, m->gs);
        matmul_q8_scalar(b, x, mats[k]->q, mats[k]->s, cols[k], rows, m->gs);
        for (int i = 0; i < rows; i++) bad += memcmp(&a[i], &b[i], sizeof(float)) != 0;
        *total += rows;
    }
    // attention: layer 0's wk bytes stand in for an int8 key/value cache of 8 positions
    int hs = c->dim / c->n_heads, kv_dim = c->dim * c->n_kv_heads / c->n_heads;
    if (hs % 8 == 0 && hs <= 257) { // wk has kv_dim rows of dim >= 8 bytes, and a holds hs floats
        __asm__ volatile("mtspr 919, %0" : : "r"(GQR_S8));
        const int8_t *kc = m->layers[0].wk.q;
        float s4[4] __attribute__((aligned(8)));
        for (int t = 0; t < 8; t += 4) { // q.k, four positions at a time
            qk4_s8(x, kc + (size_t)t * kv_dim, kv_dim, hs / 2, s4);
            for (int j = 0; j < 4; j++) {
                const int8_t *kk = kc + (size_t)(t + j) * kv_dim;
                float score = 0.0f;
                for (int i = 0; i < hs; i++) score += x[i] * I8F(kk[i]);
                bad += memcmp(&s4[j], &score, sizeof(float)) != 0;
                (*total)++;
            }
        }
        for (int i = 0; i < hs; i++) a[i] = b[i] = x[n - 1 - i]; // att*v over 8 positions
        for (int t = 0; t < 8; t++) {
            const int8_t *vv = kc + (size_t)t * kv_dim;
            float w = 0.71f - t * 0.13f;
            axpy_s8(a, vv, w, hs / 8);
            for (int i = 0; i < hs; i++) b[i] += w * I8F(vv[i]);
        }
        for (int i = 0; i < hs; i++) bad += memcmp(&a[i], &b[i], sizeof(float)) != 0;
        *total += hs;
    }
    free(x);
    free(a);
    return bad;
#else
    (void)m;
    return 0;
#endif
}

static void matmul(LlmModel *m, float *xout, const float *x, const LlmMatrix *w, int n, int d) {
    if (m->mode == LLM_MODE_GPU && w->gpu) m->gpu_matmul(w->gpu, xout, x);
    else if (m->dtype == LLM_DTYPE_F32) matmul_f32(xout, x, w->f, n, d);
    else matmul_q8(xout, x, w->q, w->s, n, d, m->gs);
}

// one row of W @ x on the CPU
static float row_dot(LlmModel *m, const LlmMatrix *w, int row, const float *x, int n) {
    float out;
    if (m->dtype == LLM_DTYPE_F32) matmul_f32(&out, x, w->f + (size_t)row * n, n, 1);
    else matmul_q8(&out, x, w->q + (size_t)row * n, w->s + (size_t)row * (n / m->gs), n, 1, m->gs);
    return out;
}

static void sift_down(int *h, int k, const float *v, int i) {
    // min-heap of indices into v
    for (;;) {
        int l = 2 * i + 1, r = l + 1, s = i;
        if (l < k && v[h[l]] < v[h[s]]) s = l;
        if (r < k && v[h[r]] < v[h[s]]) s = r;
        if (s == i) return;
        int t = h[i]; h[i] = h[s]; h[s] = t;
        i = s;
    }
}

// The GPU's logits are coarse. Find its top K and recompute those exactly on
// the CPU; the coarse values stand in for the long tail.
static void rescore(LlmModel *m, const float *x, LlmProfile *pf) {
    int V = m->c.vocab_size, k = m->rescore_k, *h = m->topk;
    if (k > V) k = V;
    if (k > LLM_MAX_RESCORE) k = LLM_MAX_RESCORE;
    const float *v = m->logits;
    for (int i = 0; i < k; i++) h[i] = i;
    for (int i = k / 2 - 1; i >= 0; i--) sift_down(h, k, v, i);
    for (int i = k; i < V; i++)
        if (v[i] > v[h[0]]) { h[0] = i; sift_down(h, k, v, 0); }
    prof_lap(pf, LLM_PROF_TOPK);
    for (int i = 0; i < k; i++) m->logits[h[i]] = row_dot(m, &m->cls, h[i], x, m->c.dim);
    prof_lap(pf, LLM_PROF_RECOMPUTE);
}

static void rmsnorm(float *o, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

// Separately rounded float operations, no fused multiply-adds: GCC takes an attribute, clang a pragma.
#ifdef __clang__
#define NO_FMA
#define NO_FMA_BODY _Pragma("STDC FP_CONTRACT OFF")
#else
#define NO_FMA __attribute__((optimize("fp-contract=off")))
#define NO_FMA_BODY
#endif

// e^x from float adds and multiplies only, so it gives the same floats on the Wii, on a desktop and in
// tools/tev_sim/llama.py (fast_exp), which copies it operation for operation. newlib's expf costs about
// 150 cycles a call on the Wii. Cody-Waite reduction x = n ln2 + r, |r| <= ln2/2, then Cephes expf's
// polynomial for e^r (about 1 ulp), times 2^n built from its exponent bits.
NO_FMA static float exp_f(float x) {
    NO_FMA_BODY
    if (x < -87.0f) return 0.0f; // e^-87 is just above FLT_MIN, so no denormals below
    if (x > 88.0f) return INFINITY;
    float n = (x * 1.44269504f + 12582912.0f) - 12582912.0f; // x / ln2, rounded to nearest even
    float r = x - n * 0.693359375f;                          // ln2 in two parts; this product is exact
    r = r - n * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    float y = p * (r * r) + r + 1.0f;
    union { uint32_t u; float f; } scale = {(uint32_t)((int)n + 127) << 23};
    return y * scale.f;
}

uint32_t llm_check_exp(void) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < 25600; i++) { // x = -100 .. 100 in steps of 1/128, every one exact
        union { float f; uint32_t u; } y = {exp_f((float)(i - 12800) / 128.0f)};
        for (int b = 0; b < 32; b += 8) h = (h ^ ((y.u >> b) & 0xff)) * 16777619u;
    }
    return h;
}

#ifdef LLM_EXACT_MATMUL
#define EXPF expf // upstream llama2.c's floats
#else
#define EXPF exp_f
#endif

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = EXPF(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

float *llm_forward(LlmModel *m, int token, int pos) {
    init_lut();
    LlmConfig *p = &m->c;
    float *x = m->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    LlmProfile *pf = pos > 0 ? m->prof : NULL; // like gen_ms, the profile skips the first token
    if (pf) pf->tokens++;
    prof_start(pf);

    // token embedding -> x
    if (m->dtype == LLM_DTYPE_F32) {
        memcpy(x, m->emb.f + (size_t)token * dim, dim * sizeof(float));
    } else {
        const int8_t *row = m->emb.q + (size_t)token * dim;
        const float *rs = m->emb.s + (size_t)token * (dim / m->gs);
        for (int j = 0; j < dim; j++) x[j] = I8F(row[j]) * rs[j / m->gs];
    }
    prof_lap(pf, LLM_PROF_EMBED);

    // RoPE's rotations depend only on the position and the dimension within a head, so they're computed
    // once per token instead of once per layer and head (the same operations, so the same floats)
    for (int i = 0; i < head_size; i += 2) {
        float freq = 1.0f / powf(10000.0f, i / (float)head_size);
        float val = pos * freq;
        m->rope[i] = cosf(val);
        m->rope[i + 1] = sinf(val);
    }
    prof_lap(pf, LLM_PROF_ROPE_TABLE);

    int kv8 = m->key_cache8 != NULL;
#ifdef GEKKO
    // paired-single attention: int8 KV, whole octets per head, and 8-byte aligned q and xb for psq_l
    int ps_attn = kv8 && head_size % 8 == 0 && ((uintptr_t)m->q & 7) == 0 && ((uintptr_t)m->xb & 7) == 0;
    if (ps_attn) __asm__ volatile("mtspr 919, %0" : : "r"(GQR_S8)); // GQR7: load s8, unscaled
#endif
    for (int l = 0; l < p->n_layers; l++) {
        LlmLayer *w = &m->layers[l];

        rmsnorm(m->xb, x, m->rms_att + (size_t)l * dim, dim);
        prof_lap(pf, LLM_PROF_RMSNORM);

        size_t loff = (size_t)l * p->seq_len * kv_dim;
        size_t soff = (size_t)l * p->seq_len * p->n_kv_heads;
        float *k = kv8 ? m->k : m->key_cache + loff + (size_t)pos * kv_dim;
        float *v = kv8 ? m->v : m->value_cache + loff + (size_t)pos * kv_dim;

        if (m->mode == LLM_MODE_GPU && w->gpu_qkv) { // stacked (int8 only): writes q, then k and v right after it
            m->gpu_matmul(w->gpu_qkv, m->q, m->xb);
        } else {
            matmul(m, m->q, m->xb, &w->wq, dim, dim);
            matmul(m, k, m->xb, &w->wk, dim, kv_dim);
            matmul(m, v, m->xb, &w->wv, dim, kv_dim);
        }
        prof_lap(pf, LLM_PROF_QKV);

        // RoPE
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float fcr = m->rope[head_dim];
            float fci = m->rope[head_dim + 1];
            int rotn = i < kv_dim ? 2 : 1;
            for (int r = 0; r < rotn; r++) {
                float *vec = r == 0 ? m->q : k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        prof_lap(pf, LLM_PROF_ROPE);

        uint32_t att_start = pf ? pf->last : 0; // for the profile's attention by position
        if (kv8) {
            for (int kh = 0; kh < p->n_kv_heads; kh++) {
                size_t at = loff + (size_t)pos * kv_dim + kh * head_size, st = soff + (size_t)pos * p->n_kv_heads + kh;
                quantize_kv(k + kh * head_size, head_size, m->key_cache8 + at, m->key_scale + st);
                quantize_kv(v + kh * head_size, head_size, m->value_cache8 + at, m->value_scale + st);
            }
        }
        prof_lap(pf, LLM_PROF_KV_QUANT);

        // multihead attention (sqrtf is a software loop on the Wii, and errno keeps GCC from hoisting it)
        float sqrt_hs = sqrtf(head_size);
        for (int h = 0; h < p->n_heads; h++) {
            int kh = h / kv_mul;
            float *q = m->q + h * head_size;
            float *att = m->att + h * p->seq_len;
            int t = 0;
#ifdef GEKKO
            if (ps_attn) {
                float s4[4] __attribute__((aligned(8)));
                for (; t + 4 <= pos + 1; t += 4) {
                    qk4_s8(q, m->key_cache8 + loff + (size_t)t * kv_dim + kh * head_size, kv_dim, head_size / 2, s4);
                    for (int j = 0; j < 4; j++) { // the scalar loop's scale and divide, per score
                        float score = s4[j] * m->key_scale[soff + (size_t)(t + j) * p->n_kv_heads + kh];
                        att[t + j] = score / sqrt_hs;
                    }
                }
            }
#endif
            for (; t <= pos; t++) {
                size_t at = loff + (size_t)t * kv_dim + kh * head_size;
                float score = 0.0f;
                if (kv8) {
                    const int8_t *kk = m->key_cache8 + at;
                    for (int i = 0; i < head_size; i++) score += q[i] * I8F(kk[i]);
                    score *= m->key_scale[soff + (size_t)t * p->n_kv_heads + kh];
                } else {
                    const float *kk = m->key_cache + at;
                    for (int i = 0; i < head_size; i++) score += q[i] * kk[i];
                }
                score /= sqrt_hs;
                att[t] = score;
            }
            prof_lap(pf, LLM_PROF_SCORES);
            softmax(att, pos + 1);
            prof_lap(pf, LLM_PROF_SOFTMAX);
            float *xb = m->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                size_t at = loff + (size_t)t * kv_dim + kh * head_size;
                float a = att[t];
                if (kv8) {
                    const int8_t *vv = m->value_cache8 + at;
                    a *= m->value_scale[soff + (size_t)t * p->n_kv_heads + kh];
#ifdef GEKKO
                    if (ps_attn) {
                        axpy_s8(xb, vv, a, head_size / 8);
                        continue;
                    }
#endif
                    for (int i = 0; i < head_size; i++) xb[i] += a * I8F(vv[i]);
                } else {
                    const float *vv = m->value_cache + at;
                    for (int i = 0; i < head_size; i++) xb[i] += a * vv[i];
                }
            }
            prof_lap(pf, LLM_PROF_MIX);
        }
        if (pf && pos < LLM_PROF_MAX_POS) pf->attention[pos] += pf->last - att_start;

        matmul(m, m->xb2, m->xb, &w->wo, dim, dim);
        prof_lap(pf, LLM_PROF_WO);
        for (int i = 0; i < dim; i++) x[i] += m->xb2[i];
        prof_lap(pf, LLM_PROF_RESIDUAL);

        // FFN: w2(silu(w1(x)) * w3(x))
        rmsnorm(m->xb, x, m->rms_ffn + (size_t)l * dim, dim);
        prof_lap(pf, LLM_PROF_RMSNORM);
        if (m->mode == LLM_MODE_GPU && w->gpu_w13) { // stacked: writes hb, then hb2 right after it
            m->gpu_matmul(w->gpu_w13, m->hb, m->xb);
        } else {
            matmul(m, m->hb, m->xb, &w->w1, dim, hidden_dim);
            matmul(m, m->hb2, m->xb, &w->w3, dim, hidden_dim);
        }
        prof_lap(pf, LLM_PROF_W1W3);
        for (int i = 0; i < hidden_dim; i++) {
            float val = m->hb[i];
            val *= (1.0f / (1.0f + EXPF(-val)));
            val *= m->hb2[i];
            m->hb[i] = val;
        }
        prof_lap(pf, LLM_PROF_SILU);
        matmul(m, m->xb, m->hb, &w->w2, hidden_dim, dim);
        prof_lap(pf, LLM_PROF_W2);
        for (int i = 0; i < dim; i++) x[i] += m->xb[i];
        prof_lap(pf, LLM_PROF_RESIDUAL);
    }

    rmsnorm(x, x, m->rms_final, dim);
    prof_lap(pf, LLM_PROF_RMSNORM);
    if (m->mode != LLM_MODE_CPU && m->cls.gpu) {
        m->gpu_matmul(m->cls.gpu, m->logits, x);
        prof_lap(pf, LLM_PROF_CLS);
        if (m->rescore_k > 0) rescore(m, x, pf);
    } else {
        matmul(m, m->logits, x, &m->cls, p->dim, p->vocab_size);
        prof_lap(pf, LLM_PROF_CLS);
    }
    return m->logits;
}

// ----------------------------------------------------------------------------
// tokenizer (sentencepiece BPE, same algorithm as llama2.c)

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((const LlmTokenIndex *)a)->str, ((const LlmTokenIndex *)b)->str);
}

void llm_free_tokenizer(LlmTokenizer *t) {
    if (t->vocab)
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->scores);
    free(t->sorted);
    memset(t, 0, sizeof(*t));
}

int llm_load_tokenizer(LlmTokenizer *t, const char *path, int vocab_size) {
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *f = fopen(path, "rb");
    if (!f) return fail("couldn't open %s", path);
    Reader r = {0};
    r.f = f;
    int rc = -1;
    char magic[4];
    int32_t h[2];
    if (read_bytes(&r, magic, 4) || memcmp(magic, "WTOK", 4) != 0) {
        fail("%s isn't an iiama2.c tokenizer (run tools/export_wii.py)", path);
        goto done;
    }
    if (read_be32(&r, h, 2)) goto done;
    if (h[0] != vocab_size) {
        fail("tokenizer has %d tokens, model expects %d", (int)h[0], vocab_size);
        goto done;
    }
    t->vocab_size = h[0];
    t->max_token_length = h[1];
    t->vocab = (char **)calloc(t->vocab_size, sizeof(char *));
    t->scores = (float *)malloc(t->vocab_size * sizeof(float));
    t->sorted = (LlmTokenIndex *)malloc(t->vocab_size * sizeof(LlmTokenIndex));
    if (!t->vocab || !t->scores || !t->sorted) { fail("out of memory loading tokenizer"); goto done; }
    for (int i = 0; i < t->vocab_size; i++) {
        int32_t len;
        if (read_be32(&r, &t->scores[i], 1) || read_be32(&r, &len, 1)) goto done;
        if (len < 0 || len > 4096) { fail("tokenizer file looks corrupt"); goto done; }
        t->vocab[i] = (char *)malloc(len + 1);
        if (!t->vocab[i]) { fail("out of memory loading tokenizer"); goto done; }
        if (len && read_bytes(&r, t->vocab[i], len)) goto done;
        t->vocab[i][len] = '\0';
    }
    // sort up front (upstream does it lazily); avoids a hitch on the first prompt
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted[i].str = t->vocab[i];
        t->sorted[i].id = i;
    }
    qsort(t->sorted, t->vocab_size, sizeof(LlmTokenIndex), compare_tokens);
    rc = 0;
done:
    fclose(f);
    if (rc != 0) llm_free_tokenizer(t);
    return rc;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

const char *llm_decode(LlmTokenizer *t, int prev_token, int token) {
    const char *piece = t->vocab[token];
    // after BOS (1), sentencepiece strips a leading space
    if (prev_token == 1 && piece[0] == ' ') piece++;
    // raw byte tokens look like "<0x0A>"
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
        int hi = hexval(piece[3]);
        if (hi >= 0) {
            int lo = hexval(piece[4]);
            int byte_val = lo >= 0 ? hi * 16 + lo : hi;
            piece = (const char *)t->byte_pieces + byte_val * 2;
        }
    }
    return piece;
}

static int str_lookup(const char *str, LlmTokenIndex *sorted, int vocab_size) {
    LlmTokenIndex tok;
    tok.str = (char *)str;
    LlmTokenIndex *res = (LlmTokenIndex *)bsearch(&tok, sorted, vocab_size, sizeof(LlmTokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

int llm_encode(LlmTokenizer *t, const char *text, int bos, int eos, int *tokens, int *n_tokens) {
    char *str_buffer = (char *)malloc(t->max_token_length * 2 + 1 + 2);
    if (!str_buffer) return fail("out of memory in tokenizer");
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') tokens[(*n_tokens)++] = str_lookup(" ", t->sorted, t->vocab_size);

    for (const char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // byte fallback: first 3 ids are <unk>, <s>, </s>
            for (size_t i = 0; i < str_len; i++) tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }

    // greedily merge the best-scoring adjacent pair until none remain
    while (1) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            snprintf(str_buffer, t->max_token_length * 2 + 3, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted, t->vocab_size);
            if (id != -1 && t->scores[id] > best_score) {
                best_score = t->scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
    return 0;
}

// ----------------------------------------------------------------------------
// sampler

int llm_init_sampler(LlmSampler *s, int vocab_size, float temperature, float topp, unsigned long long seed) {
    s->vocab_size = vocab_size;
    s->temperature = temperature;
    s->topp = topp;
    s->rng_state = seed ? seed : 0x9E3779B97F4A7C15ull;
    s->probindex = (LlmProbIndex *)malloc(vocab_size * sizeof(LlmProbIndex));
    return s->probindex ? 0 : fail("out of memory for sampler");
}

void llm_free_sampler(LlmSampler *s) {
    free(s->probindex);
    s->probindex = NULL;
}

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (unsigned int)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample_argmax(const float *p, int n) {
    int max_i = 0;
    float max_p = p[0];
    for (int i = 1; i < n; i++)
        if (p[i] > max_p) { max_i = i; max_p = p[i]; }
    return max_i;
}

static int sample_mult(const float *p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += p[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

static int compare_prob(const void *a, const void *b) {
    const LlmProbIndex *a_ = (const LlmProbIndex *)a, *b_ = (const LlmProbIndex *)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

static int sample_topp(const float *p, int n, float topp, LlmProbIndex *pi, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (p[i] >= cutoff) {
            pi[n0].index = i;
            pi[n0].prob = p[i];
            n0++;
        }
    }
    qsort(pi, n0, sizeof(LlmProbIndex), compare_prob);
    float cumulative = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative += pi[i].prob;
        if (cumulative > topp) { last_idx = i; break; }
    }
    float r = coin * cumulative;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += pi[i].prob;
        if (r < cdf) return pi[i].index;
    }
    return pi[last_idx].index;
}

int llm_sample(LlmSampler *s, float *logits) {
    if (s->temperature == 0.0f) return sample_argmax(logits, s->vocab_size);
    for (int q = 0; q < s->vocab_size; q++) logits[q] /= s->temperature;
    softmax(logits, s->vocab_size);
    float coin = random_f32(&s->rng_state);
    if (s->topp <= 0 || s->topp >= 1) return sample_mult(logits, s->vocab_size, coin);
    return sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
}

// ----------------------------------------------------------------------------
// generation loop

static int printable(const char *piece) {
    if (piece == NULL || piece[0] == '\0') return 0;
    if (piece[1] == '\0') {
        unsigned char b = (unsigned char)piece[0];
        if (!(isprint(b) || isspace(b))) return 0; // stray control byte
    }
    return 1;
}

int llm_generate(LlmModel *m, LlmTokenizer *t, LlmSampler *s, const char *prompt, int steps,
                 LlmTokenFn on_token, void *user, LlmClockFn clock_ms, unsigned long *gen_ms) {
    if (prompt == NULL) prompt = "";
    if (steps <= 0 || steps > m->c.seq_len) steps = m->c.seq_len;

    int n_prompt = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
    if (!prompt_tokens) { fail("out of memory for prompt"); return -1; }
    if (llm_encode(t, prompt, 1, 0, prompt_tokens, &n_prompt) || n_prompt < 1) {
        free(prompt_tokens);
        if (n_prompt < 1) fail("prompt encoded to zero tokens");
        return -1;
    }

    unsigned long start = 0;
    int started = 0;
    int next, token = prompt_tokens[0], pos = 0;
    while (pos < steps) {
        float *logits = llm_forward(m, token, pos);
        next = pos < n_prompt - 1 ? prompt_tokens[pos + 1] : llm_sample(s, logits);
        prof_lap(pos > 0 ? m->prof : NULL, LLM_PROF_SAMPLE); // since llm_forward's last lap, at the logits
        pos++;
        if (next == 1) break; // BOS delimits sequences
        const char *piece = llm_decode(t, token, next);
        if (printable(piece) && on_token && on_token(piece, user)) break;
        token = next;
        if (!started && clock_ms) { start = clock_ms(); started = 1; }
    }
    if (gen_ms) *gen_ms = (started && clock_ms) ? clock_ms() - start : 0;
    free(prompt_tokens);
    return pos;
}
