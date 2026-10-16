// llm.h: a llama2.c-style transformer, reworked for the Wii.
//
// Differences from upstream llama2.c run.c:
//   * reads the big-endian files made by tools/export_wii.py (byteswaps on
//     little-endian hosts, so the same files also work on your PC for testing)
//   * no mmap: every matrix is its own malloc, so weights can be split across
//     the Wii's two memory banks (24 MB MEM1 + 64 MB MEM2)
//   * optional int8 weights (q8) with per-group scales
//   * no exit() on errors: functions return nonzero and llm_error() says why,
//     so the Wii can show the message instead of black-screening
//   * a hook for running weight matmuls elsewhere (LlmModel.mode/gpu_matmul),
//     which gx_matmul.c uses to put them on the GPU
#ifndef WII_LLM_H
#define WII_LLM_H

#include <stddef.h>
#include <stdint.h>

#define LLM_DTYPE_F32 0
#define LLM_DTYPE_Q8 1

// Where the weight matmuls run. The GPU side lives in gx_matmul.c (Wii only).
#define LLM_MODE_CPU 0   // everything on the CPU
#define LLM_MODE_SPLIT 1 // classifier on the GPU; transformer layers on the CPU
#define LLM_MODE_GPU 2   // every weight matmul on the GPU

#define LLM_MAX_RESCORE 256

// xout = W @ x for a matrix uploaded to the GPU (LlmMatrix.gpu).
typedef void (*LlmGpuMatmulFn)(void *gpu, float *xout, const float *x);

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} LlmConfig;

// A weight matrix (rows, cols), stored either as fp32 or as int8 + group scales.
typedef struct {
    float *f;    // fp32 weights (LLM_DTYPE_F32)
    int8_t *q;   // int8 weights (LLM_DTYPE_Q8)
    float *s;    // one scale per `gs` weights (LLM_DTYPE_Q8)
    void *gpu;   // GPU copy, or NULL (owned by whoever uploaded it)
} LlmMatrix;

typedef struct {
    LlmMatrix wq, wk, wv, wo, w1, w2, w3;
    // GPU copies of wq, wk, wv stacked into one matrix (their rows in that order), and of w1, w3, so each
    // group runs as one matmul (or NULL). The matrices in a stack have no GPU copy of their own.
    void *gpu_qkv, *gpu_w13;
} LlmLayer;

typedef struct {
    LlmConfig c;
    int dtype, gs, shared;
    float *rms_att, *rms_ffn, *rms_final;
    LlmMatrix emb, cls;
    LlmLayer *layers;
    // activations + KV cache. int8 models keep the cache in int8 with one scale per head and position
    // (key_cache8 etc.), a quarter of the bytes for attention to read; fp32 models keep it in float.
    // hb2 follows hb in memory, and in int8 models k and v follow q, so a stacked GPU matmul writes them all.
    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits, *key_cache, *value_cache;
    int8_t *key_cache8, *value_cache8;
    float *key_scale, *value_scale; // [layer][position][kv head]
    float *k, *v;                   // this position's key and value, before they're quantized
    float *rope;                    // this position's RoPE cos/sin pairs, one per two head dimensions
    size_t bytes; // total bytes allocated, for the stats screen
    // GPU offload (all zero = CPU only)
    int mode;                  // LLM_MODE_*
    int rescore_k;             // when the classifier runs on the GPU, recompute its top K logits exactly on the CPU
    LlmGpuMatmulFn gpu_matmul; // set by whoever uploaded the matrices
    int topk[LLM_MAX_RESCORE];
} LlmModel;

typedef struct {
    char *str;
    int id;
} LlmTokenIndex;

typedef struct {
    char **vocab;
    float *scores;
    LlmTokenIndex *sorted;
    int vocab_size;
    int max_token_length;
    unsigned char byte_pieces[512];
} LlmTokenizer;

typedef struct {
    float prob;
    int index;
} LlmProbIndex;

typedef struct {
    int vocab_size;
    LlmProbIndex *probindex;
    float temperature, topp;
    unsigned long long rng_state;
} LlmSampler;

// Called while loading with (bytes_done, bytes_total).
typedef void (*LlmProgressFn)(size_t done, size_t total, void *user);
// Called for each generated piece of text. Return nonzero to stop early.
typedef int (*LlmTokenFn)(const char *piece, void *user);

const char *llm_error(void);
int llm_set_error(const char *fmt, ...); // for code outside llm.c (gx_matmul.c); always returns -1

int llm_load_model(LlmModel *m, const char *path, LlmProgressFn progress, void *user);
void llm_free_model(LlmModel *m);
int llm_load_tokenizer(LlmTokenizer *t, const char *path, int vocab_size);
void llm_free_tokenizer(LlmTokenizer *t);
int llm_init_sampler(LlmSampler *s, int vocab_size, float temperature, float topp, unsigned long long seed);
void llm_free_sampler(LlmSampler *s);

float *llm_forward(LlmModel *m, int token, int pos);
int llm_encode(LlmTokenizer *t, const char *text, int bos, int eos, int *tokens, int *n_tokens);
const char *llm_decode(LlmTokenizer *t, int prev_token, int token);
int llm_sample(LlmSampler *s, float *logits);

// Runs a full generation. Returns number of tokens generated (>= 0) or -1 on error.
// gen_ms (optional) receives the time spent after the first token, for tok/s.
typedef unsigned long (*LlmClockFn)(void); // milliseconds
int llm_generate(LlmModel *m, LlmTokenizer *t, LlmSampler *s, const char *prompt, int steps,
                 LlmTokenFn on_token, void *user, LlmClockFn clock_ms, unsigned long *gen_ms);

#endif
