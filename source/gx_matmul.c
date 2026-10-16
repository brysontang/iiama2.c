// gx_matmul.c: weight matmuls on the Wii's GPU, for LLM_MODE_SPLIT and LLM_MODE_GPU.
//
// This is tools/tev_sim/gxmatmul.py written in GX calls. The simulator
// explains the math and measures what each output mode costs in accuracy.
// In short:
//   * one draw = one pass over 16 input columns (8 in exact11 mode)
//   * each framebuffer pixel holds 3 output rows, one per RGB channel
//   * weights: RGBA8 textures of q + 128, the same int8 values the CPU uses
//   * activations: the 4 konst colors, one byte per TEV stage
//   * stage k: PREV += +/- round(q * x_k / 256), computed as
//     lerp(HALF, TEXC, KONST) with the stage bias cancelling the HALF
//   * the CPU copies the framebuffer back and sums the passes in float
// clamp8 runs on a twin of each layer matrix with one weight scale per row, which
// saturates less and lets the readback skip the scales (run_rows, sum_pixel).
// The classifier is different: its passes all blend into the same pixels, so the
// framebuffer does the summing and the CPU reads one byte per row (run_blend).

#include "gx_matmul.h"

#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <gccore.h>
#include <ogc/cast.h>
#include <ogc/lwp_watchdog.h>

#define FIFO_SIZE (256 * 1024)
#define EFB_W 640
#define EFB_H 480
#define PLANES 8        // input columns per texture, side by side
#define PLANE_W_MAX 128 // so a texture is at most 8 * 128 = 1024 texels wide
#define MAX_PASSES 128  // up to 1024 input columns
#define LAYOUT_BLEND 2  // load_layout()'s third setup, next to GXM_CLAMP8 and GXM_EXACT11

// The blended classifier: x's headroom, the framebuffer's starting value, and the TEV's offset per pass.
// Must match BLEND in tools/tev_sim/gxmatmul.py.
#define BLEND_HEADROOM 0.25f
#define BLEND_A0 32
#define BLEND_C0 32

// How a matrix's texture bytes are scaled
enum {
    SCALES_GROUP, // the CPU's int8 as is, one scale per row and group of gs columns (st)
    SCALES_ROW,   // requantized with one scale per row (srow)
    SCALES_BLEND, // SCALES_ROW, with all passes summed in the framebuffer
};

typedef struct GxMat {
    int rows, cols, gs;
    int npx;            // output pixels per pass: ceil(rows / 3)
    int pw, ph;         // one pass's framebuffer region, both multiples of 4
    int ntex;           // one texture per 8 input columns
    uint32_t tex_bytes; // per texture
    uint8_t *tex;
    GXTexObj *obj;
    float *st;          // SCALES_GROUP: weight scales transposed to (cols / gs, 3 * npx) so the readback streams them
    int kind;           // SCALES_*
    float *srow;        // SCALES_ROW and SCALES_BLEND: per-row scales (3 * npx, zero past the last row)
    struct GxMat *per_row; // a layer matrix's SCALES_ROW twin, which clamp8 runs on
} GxMat;

static int g_ready;
static int g_mode = GXM_CLAMP8;
static float g_headroom = 0.5f;
static int g_layout = -1;      // which static TEV setup is loaded
// Framebuffer copies land in g_hot when they fit. It's allocated before the model (gxm_reserve), so it's
// in MEM1, which the CPU streams ~2.8x faster than MEM2 (408 vs 147 MB/s on a Wii); g_readbuf is MEM2.
#define HOT_BYTES (128 * 1024)
static uint8_t *g_hot, *g_readbuf;
static uint8_t *g_copy;        // the buffer the latest copy went to
static float g_xscale[MAX_PASSES]; // per-pass activation scale
static uint16_t g_token;       // the latest draw-sync token (put_token)
static float *g_acc;           // run_rows' per-row sums
static int g_acc_n;
static int16_t *g_partials;    // only while hashing
static const uint8_t *g_last_copy; // the latest framebuffer copy (gx_probe.c reads it)
static int g_last_cw, g_last_ch;
static size_t g_bytes;
// framebuffer byte -> pass sum, without PowerPC's slow int-to-float conversion
static float g_lut_c8[256], g_lut_hi[256], g_lut_lo[256], g_lut_bl[2][256];
static GxmStats g_stats;

int gxm_reserve(void) {
    if (!g_hot) g_hot = (uint8_t *)memalign(32, HOT_BYTES);
    return g_hot ? 0 : -1;
}

static uint8_t *copy_buf(size_t bytes) { return g_hot && bytes <= HOT_BYTES ? g_hot : g_readbuf; }

int gxm_init(void) {
    if (g_ready) return 0;
    void *fifo = memalign(32, FIFO_SIZE);
    g_readbuf = (uint8_t *)memalign(32, EFB_W * EFB_H * 4);
    if (!fifo || !g_readbuf) return llm_set_error("out of memory setting up the GPU");
    memset(fifo, 0, FIFO_SIZE);
    GX_Init(fifo, FIFO_SIZE);
    g_token = GX_GetDrawSync();
    for (int b = 0; b < 256; b++) {
        g_lut_c8[b] = (float)(b - 128);
        g_lut_hi[b] = (float)(8 * (b - 128));
        g_lut_lo[b] = (float)(b & 7);
        g_lut_bl[0][b] = (float)(b - BLEND_A0); // even pass count
        g_lut_bl[1][b] = (float)(b - BLEND_A0 - BLEND_C0);
    }

    GXColor black = {0, 0, 0, 0};
    GX_SetCopyClear(black, GX_MAX_Z24);
    GX_SetPixelFmt(GX_PF_RGB8_Z24, GX_ZC_LINEAR);
    GX_SetViewport(0, 0, EFB_W, EFB_H, 0, 1);
    GX_SetScissor(0, 0, EFB_W, EFB_H);
    Mtx44 proj;
    guOrtho(proj, 0, EFB_H, 0, EFB_W, 0, 1); // 1 unit = 1 pixel, y down
    GX_LoadProjectionMtx(proj, GX_ORTHOGRAPHIC);
    Mtx id;
    guMtxIdentity(id);
    GX_LoadPosMtxImm(id, GX_PNMTX0);
    GX_SetCurrentMtx(GX_PNMTX0);

    // plain integer pixels out: no culling, depth, blending, dither, fog, or copy filtering
    GX_SetCullMode(GX_CULL_NONE);
    GX_SetClipMode(GX_CLIP_DISABLE);
    GX_SetZMode(GX_FALSE, GX_ALWAYS, GX_FALSE);
    GX_SetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR);
    GX_SetAlphaCompare(GX_ALWAYS, 0, GX_AOP_AND, GX_ALWAYS, 0);
    GX_SetColorUpdate(GX_TRUE);
    GX_SetAlphaUpdate(GX_FALSE);
    GX_SetDither(GX_FALSE);
    GX_SetFog(GX_FOG_NONE, 0, 1, 0, 1, black);
    GX_SetCopyFilter(GX_FALSE, NULL, GX_FALSE, NULL);
    GX_SetNumChans(1);
    GX_SetChanCtrl(GX_COLOR0A0, GX_DISABLE, GX_SRC_REG, GX_SRC_REG, 0, GX_DF_NONE, GX_AF_NONE);
    GX_SetNumIndStages(0);
    for (int k = 0; k < 16; k++) GX_SetTevDirect(GX_TEVSTAGE0 + k);

    GX_ClearVtxDesc();
    GX_SetVtxDesc(GX_VA_POS, GX_DIRECT);
    GX_SetVtxDesc(GX_VA_TEX0, GX_DIRECT);
    GX_SetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XY, GX_F32, 0);
    GX_SetVtxAttrFmt(GX_VTXFMT0, GX_VA_TEX0, GX_TEX_ST, GX_F32, 0);

    // texcoord j samples plane j: the quad's s runs over [0, 1/8], shifted by j/8
    GX_SetNumTexGens(PLANES);
    for (int j = 0; j < PLANES; j++) {
        Mtx t;
        guMtxTrans(t, j / (float)PLANES, 0, 0);
        GX_LoadTexMtxImm(t, GX_TEXMTX0 + 3 * j, GX_MTX2x4);
        GX_SetTexCoordGen(GX_TEXCOORD0 + j, GX_TG_MTX2x4, GX_TG_TEX0, GX_TEXMTX0 + 3 * j);
    }
    g_ready = 1;
    return 0;
}

void gxm_set_mode(int out_mode, float headroom) {
    g_mode = out_mode;
    g_headroom = headroom;
}

// The parts of the TEV setup that don't change from pass to pass.
static void load_layout(int mode) {
    if (g_layout == mode) return;
    int product = mode == GXM_EXACT11 ? 8 : 16;
    for (int k = 0; k < product; k++) {
        u8 st = GX_TEVSTAGE0 + k;
        GX_SetTevOrder(st, GX_TEXCOORD0 + (k & 7), GX_TEXMAP0 + (k >> 3), GX_COLORNULL);
        // Stage 0 starts the sum from a constant: real hardware doesn't reset PREV for each pixel
        // (it holds whatever an earlier pixel left there), so no stage may read PREV before writing it.
        // The blended classifier's start changes between passes, so it comes from C0.
        u8 d = k > 0 ? GX_CC_CPREV : mode == GXM_CLAMP8 ? GX_CC_HALF : mode == LAYOUT_BLEND ? GX_CC_C0 : GX_CC_ZERO;
        GX_SetTevColorIn(st, GX_CC_HALF, GX_CC_TEXC, GX_CC_KONST, d);
        GX_SetTevKColorSel(st, GX_TEV_KCSEL_K0_R + 4 * (k & 3) + (k >> 2)); // x_k = kcolor[k/4].rgba[k%4]
        GX_SetTevAlphaIn(st, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO);
        GX_SetTevAlphaOp(st, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
        GX_SetTevSwapMode(st, GX_TEV_SWAP0, GX_TEV_SWAP0);
    }
    if (mode == GXM_EXACT11) {
        // draw 2's extra stages: PREV -> floor(PREV / 8) + 128, via three halvings
        // (+128, +128, then +128 + 32 from the 1/8 konst)
        for (int k = 8; k < 11; k++) {
            u8 st = GX_TEVSTAGE0 + k;
            GX_SetTevOrder(st, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLORNULL);
            GX_SetTevColorIn(st, k == 10 ? GX_CC_KONST : GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_CPREV);
            GX_SetTevKColorSel(st, GX_TEV_KCSEL_1_8);
            GX_SetTevColorOp(st, GX_TEV_ADD, GX_TB_ADDHALF, GX_CS_DIVIDE_2, k == 10, GX_TEVPREV);
            GX_SetTevAlphaIn(st, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO);
            GX_SetTevAlphaOp(st, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
            GX_SetTevSwapMode(st, GX_TEV_SWAP0, GX_TEV_SWAP0);
        }
    }
    g_layout = mode;
}

// Texture byte for (row, col): the CPU's int8 weight + 128, or the weight requantized with one scale
// per row. Must match requantize_rows in tools/tev_sim/gxmatmul.py exactly.
static uint8_t wu(const GxMat *g, const LlmMatrix *w, int row, int col) {
    if (row >= g->rows) return 128;
    size_t i = (size_t)row * g->cols + col;
    if (g->kind == SCALES_GROUP) return (uint8_t)(w->q[i] + 128);
    float v = (float)w->q[i] * w->s[i / g->gs];
    float s = g->srow[row] == 0.0f ? 1.0f : g->srow[row];
    int q = (int)rintf(v / s);
    q = q > 127 ? 127 : q < -127 ? -127 : q;
    return (uint8_t)(q + 128);
}

static GxMat *upload(const LlmModel *m, const LlmMatrix *w, int rows, int cols, int kind) {
    if (m->dtype != LLM_DTYPE_Q8 || m->gs % 16 || cols % 16 || cols / 8 > MAX_PASSES) {
        llm_set_error("the GPU path needs int8 weights with a group size that's a multiple of 16");
        return NULL;
    }
    GxMat *g = (GxMat *)calloc(1, sizeof(GxMat));
    if (!g) { llm_set_error("out of memory for GPU matrix"); return NULL; }
    g->rows = rows;
    g->cols = cols;
    g->gs = m->gs;
    g->kind = kind;
    g->npx = (rows + 2) / 3;
    // smallest region (both sides multiples of 4, the RGBA8 tile size) that holds npx pixels
    g->pw = g->ph = 0;
    for (int ph = 4; ph <= EFB_H; ph += 4) {
        int pw = ((g->npx + ph - 1) / ph + 3) & ~3;
        if (pw <= PLANE_W_MAX && (g->pw == 0 || pw * ph < g->pw * g->ph)) { g->pw = pw; g->ph = ph; }
    }
    g->ntex = cols / PLANES;
    int tw = PLANES * g->pw;
    g->tex_bytes = (uint32_t)tw * g->ph * 4;
    if (g->pw == 0) {
        free(g);
        llm_set_error("matrix too tall for the framebuffer (%d rows)", rows);
        return NULL;
    }
    int groups = cols / g->gs, padded_rows = 3 * g->npx, per_row = kind != SCALES_GROUP;
    if (kind == SCALES_ROW && padded_rows > g_acc_n) {
        free(g_acc);
        g_acc = (float *)memalign(32, padded_rows * sizeof(float));
        g_acc_n = g_acc ? padded_rows : 0;
    }
    g->tex = (uint8_t *)memalign(32, (size_t)g->tex_bytes * g->ntex);
    g->obj = (GXTexObj *)malloc(sizeof(GXTexObj) * g->ntex);
    size_t nscales = per_row ? (size_t)padded_rows : (size_t)groups * padded_rows;
    float *scales = (float *)calloc(nscales, sizeof(float));
    if (per_row) g->srow = scales;
    else g->st = scales;
    if (!g->tex || !g->obj || !scales || (kind == SCALES_ROW && !g_acc)) {
        llm_set_error("out of memory for GPU textures (%lu KB, %lu KB already uploaded)",
                      (unsigned long)((size_t)g->tex_bytes * g->ntex / 1024), (unsigned long)(g_bytes / 1024));
        free(g->tex);
        free(g->obj);
        free(scales);
        free(g);
        return NULL;
    }
    if (per_row) {
        for (int r = 0; r < rows; r++) {
            float amax = 0.0f;
            for (int c = 0; c < cols; c++) {
                size_t i = (size_t)r * cols + c;
                float v = fabsf((float)w->q[i] * w->s[i / g->gs]);
                if (v > amax) amax = v;
            }
            g->srow[r] = amax / 127.0f;
        }
    } else {
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < groups; k++) g->st[(size_t)k * padded_rows + r] = w->s[(size_t)r * groups + k];
    }

    // RGBA8 textures are 4x4 tiles of 64 bytes: 16 AR pairs, then 16 GB pairs
    for (int t = 0; t < g->ntex; t++) {
        uint8_t *base = g->tex + (size_t)t * g->tex_bytes;
        for (int j = 0; j < PLANES; j++) {
            int col = t * PLANES + j;
            for (int py = 0; py < g->ph; py++) {
                for (int px = 0; px < g->pw; px++) {
                    int tx = j * g->pw + px, row = 3 * (py * g->pw + px);
                    uint8_t *tile = base + ((((py >> 2) * (tw >> 2)) + (tx >> 2)) << 6);
                    int i = ((py & 3) << 2) | (tx & 3);
                    tile[2 * i] = 0;
                    tile[2 * i + 1] = wu(g, w, row, col);
                    tile[32 + 2 * i] = wu(g, w, row + 1, col);
                    tile[32 + 2 * i + 1] = wu(g, w, row + 2, col);
                }
            }
        }
        DCFlushRange(base, g->tex_bytes);
        GX_InitTexObj(&g->obj[t], base, tw, g->ph, GX_TF_RGBA8, GX_CLAMP, GX_CLAMP, GX_FALSE);
        GX_InitTexObjFilterMode(&g->obj[t], GX_NEAR, GX_NEAR);
    }
    GX_InvalidateTexAll();
    g_bytes += (size_t)g->tex_bytes * g->ntex + nscales * sizeof(float);
    return g;
}

void *gxm_upload(const LlmModel *m, const LlmMatrix *w, int rows, int cols) {
    return upload(m, w, rows, cols, SCALES_GROUP);
}

// A layer matrix: per-group textures for exact11, and a per-row twin for clamp8. Without memory for
// the twin, clamp8 runs on the per-group textures instead.
static GxMat *upload_layer(const LlmModel *m, const LlmMatrix *w, int rows, int cols) {
    GxMat *g = upload(m, w, rows, cols, SCALES_GROUP);
    if (g) g->per_row = upload(m, w, rows, cols, SCALES_ROW);
    return g;
}

int gxm_upload_model(LlmModel *m, int layers, size_t *bytes) {
    LlmConfig *c = &m->c;
    int dim = c->dim, hid = c->hidden_dim, kv = c->dim * c->n_kv_heads / c->n_heads;
    m->gpu_matmul = gxm_matmul;
    int rc = 0;
    if (!m->cls.gpu && !(m->cls.gpu = upload(m, &m->cls, c->vocab_size, dim, SCALES_BLEND))) rc = -1;
    for (int l = 0; rc == 0 && layers && l < c->n_layers; l++) {
        LlmLayer *L = &m->layers[l];
        struct { LlmMatrix *w; int rows, cols; } mats[7] = {
            {&L->wq, dim, dim}, {&L->wk, kv, dim}, {&L->wv, kv, dim}, {&L->wo, dim, dim},
            {&L->w1, hid, dim}, {&L->w2, dim, hid}, {&L->w3, hid, dim}};
        for (int i = 0; i < 7 && rc == 0; i++)
            if (!mats[i].w->gpu && !(mats[i].w->gpu = upload_layer(m, mats[i].w, mats[i].rows, mats[i].cols))) rc = -1;
    }
    if (bytes) *bytes = g_bytes;
    return rc;
}

// Activation quantization for n columns with one scale. Must match quantize_x in gxmatmul.py exactly:
// one division per call, and rounding to nearest-even by adding and subtracting 1.5 * 2^23 (exact for
// 0 <= t < 2^22) instead of calling rintf, all in separately rounded float operations.
__attribute__((optimize("fp-contract=off")))
static void quantize(const float *x, int n, float headroom, uint8_t *kb, uint8_t *neg, float *xscale) {
    float xmax = 0.0f;
    for (int k = 0; k < n; k++)
        if (fabsf(x[k]) > xmax) xmax = fabsf(x[k]);
    if (xmax == 0.0f) xmax = 1.0f;
    float inv = 256.0f * headroom / xmax;
    for (int k = 0; k < n; k++) {
        float v = x[k];
        float t = fabsf(v) * inv;
        int cp = (int)((t + 12582912.0f) - 12582912.0f);
        if (cp > 256) cp = 256;
        if (cp == 128) cp = t >= 128.0f ? 129 : 127; // c' = c + (c >> 7) never equals 128
        kb[k] = (uint8_t)(cp - (cp >= 129));
        neg[k] = v < 0.0f;
    }
    *xscale = xmax / headroom;
}

static void draw_quad(int x0, int y0, int w, int h) {
    GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
    GX_Position2f32(x0, y0);
    GX_TexCoord2f32(0.0f, 0.0f);
    GX_Position2f32(x0 + w, y0);
    GX_TexCoord2f32(1.0f / PLANES, 0.0f);
    GX_Position2f32(x0 + w, y0 + h);
    GX_TexCoord2f32(1.0f / PLANES, 1.0f);
    GX_Position2f32(x0, y0 + h);
    GX_TexCoord2f32(0.0f, 1.0f);
    GX_End();
}

// One write to a GPU (BP) register through the write-gather pipe, as libogc's GX_LOAD_BP_REG does.
static inline void bp(uint32_t v) {
    wgPipe->U8 = 0x61;
    __asm__ volatile("" ::: "memory");
    wgPipe->U32 = v;
    __asm__ volatile("" ::: "memory");
}

// Per-pass state, written straight to the GPU's registers (g_raw) or through libogc. libogc's setters
// update shadow copies first, and GX_LoadTexObj makes the next draw redo the texture-size setup, which
// added up to ~3,600 instructions a pass. Only the konst colors, the product stages' ops, and the
// texture addresses change from pass to pass.
static int g_raw = 1;

// x's bytes into the konst colors, and each product stage's op: subtract when x_k < 0 (flipped by
// `flip`). d0 is stage 0's starting value and must be what load_layout() set.
static void set_pass(const uint8_t *kb, const uint8_t *neg, int n, int flip, int d0, int clamp_last) {
    for (int kc = 0; kc < n / 4; kc++) {
        const uint8_t *c = kb + 4 * kc; // r, g, b, a
        if (!g_raw) {
            GXColor col = {c[0], c[1], c[2], c[3]};
            GX_SetTevKColor(GX_KCOLOR0 + kc, col);
            continue;
        }
        bp((uint32_t)(0xe0 + 2 * kc) << 24 | 1u << 23 | (uint32_t)c[3] << 12 | c[0]);
        uint32_t bg = (uint32_t)(0xe1 + 2 * kc) << 24 | 1u << 23 | (uint32_t)c[1] << 12 | c[2];
        bp(bg); // three times, like GX_SetTevKColor
        bp(bg);
        bp(bg);
    }
    for (int k = 0; k < n; k++) {
        int sub = neg[k] ^ flip, clamp = clamp_last && k == n - 1;
        int op = sub ? GX_TEV_SUB : GX_TEV_ADD, bias = sub ? GX_TB_ADDHALF : GX_TB_SUBHALF;
        if (!g_raw) {
            GX_SetTevColorOp(GX_TEVSTAGE0 + k, op, bias, GX_CS_SCALE_1, clamp, GX_TEVPREV);
            continue;
        }
        bp((uint32_t)(0xc0 + 2 * k) << 24 | GX_TEVPREV << 22 | GX_CS_SCALE_1 << 20 | (uint32_t)clamp << 19 |
           (uint32_t)op << 18 | (uint32_t)bias << 16 | GX_CC_HALF << 12 | GX_CC_TEXC << 8 | GX_CC_KONST << 4 |
           (k > 0 ? GX_CC_CPREV : d0));
    }
}

// Textures t0 .. t0 + n - 1 of g into texmaps 0 .. n - 1. The first pass of a matmul loads them fully;
// later passes have the same size and format, so only the address registers change.
static void set_textures(const GxMat *g, int t0, int n, int first) {
    for (int t = 0; t < n; t++) {
        if (first || !g_raw) {
            GX_LoadTexObj(&g->obj[t0 + t], GX_TEXMAP0 + t);
            continue;
        }
        uint32_t addr = MEM_VIRTUAL_TO_PHYSICAL(g->tex + (size_t)(t0 + t) * g->tex_bytes) >> 5;
        bp((uint32_t)(t < 4 ? 0x94 + t : 0xb4 + t - 4) << 24 | (addr & 0xffffff));
    }
}

// Draw-sync tokens: the GPU sets its token register when it reaches one in the command stream, after
// everything before it, framebuffer copies included, is done.
static uint16_t put_token(void) {
    GX_SetDrawSync(++g_token); // also flushes the write-gather pipe
    return g_token;
}

// If a token never shows up (100 ms), fall back to GX_DrawDone for good: correct, just without overlap.
static int g_tokens_failed;

static void wait_token(uint16_t t) {
    if (!g_tokens_failed) {
        uint64_t t0 = gettime();
        while ((int16_t)(GX_GetDrawSync() - t) < 0)
            if (ticks_to_millisecs(gettime() - t0) > 100) {
                g_tokens_failed = 1;
                break;
            }
        if (!g_tokens_failed) return;
    }
    GX_DrawDone();
}

// pixel i (0..15) of a 4x4 RGBA8 tile: R at 2i+1, G at 32+2i, B at 33+2i
static inline void read_rgb(const uint8_t *buf, int width, int x, int y, int *rgb) {
    const uint8_t *tile = buf + ((((y >> 2) * (width >> 2)) + (x >> 2)) << 6);
    int i = ((y & 3) << 2) | (x & 3);
    rgb[0] = tile[2 * i + 1];
    rgb[1] = tile[32 + 2 * i];
    rgb[2] = tile[32 + 2 * i + 1];
}

// framebuffer slot -> its first tile in the copy
static const uint8_t *slot_base(const GxMat *g, int slot, int slots_x, int cw) {
    return g_copy + ((size_t)((slot / slots_x) * g->ph / 4) * (cw / 4) + (slot % slots_x) * g->pw / 4) * 64;
}

// Adds one batch of passes into xout, tile by tile, so the framebuffer copy,
// the scales and xout are all read in order. Per row, the float operations are
// the same as gxmatmul.py's: y += (s * xscale) * partial, pass by pass.
static void reduce_batch(const GxMat *g, float *xout, int p0, int np, int pc, int draws, int slots_x, int cw) {
    const uint8_t *lo[MAX_PASSES], *hi[MAX_PASSES];
    const float *sp[MAX_PASSES];
    int padded = 3 * g->npx, clamp8 = draws == 1;
    for (int q = 0; q < np; q++) {
        lo[q] = slot_base(g, q * draws, slots_x, cw);
        hi[q] = slot_base(g, q * draws + 1, slots_x, cw);
        sp[q] = g->st + (size_t)(((p0 + q) * pc) / g->gs) * padded;
    }
    const float *xs = g_xscale + p0;
    for (int ty = 0; ty < g->ph / 4; ty++) {
        for (int tx = 0; tx < g->pw / 4; tx++) {
            size_t off = ((size_t)ty * (cw / 4) + tx) * 64;
            for (int i = 0; i < 16; i++) {
                int pix = (ty * 4 + (i >> 2)) * g->pw + tx * 4 + (i & 3);
                if (pix >= g->npx) continue;
                int row = 3 * pix, n = g->rows - row < 3 ? g->rows - row : 3;
                float a0 = xout[row], a1 = n > 1 ? xout[row + 1] : 0.0f, a2 = n > 2 ? xout[row + 2] : 0.0f;
                for (int q = 0; q < np; q++) {
                    const uint8_t *t = lo[q] + off;
                    float v0, v1, v2;
                    if (clamp8) {
                        v0 = g_lut_c8[t[2 * i + 1]];
                        v1 = g_lut_c8[t[32 + 2 * i]];
                        v2 = g_lut_c8[t[33 + 2 * i]];
                    } else {
                        const uint8_t *h = hi[q] + off;
                        v0 = g_lut_hi[h[2 * i + 1]] + g_lut_lo[t[2 * i + 1]];
                        v1 = g_lut_hi[h[32 + 2 * i]] + g_lut_lo[t[32 + 2 * i]];
                        v2 = g_lut_hi[h[33 + 2 * i]] + g_lut_lo[t[33 + 2 * i]];
                    }
                    const float *s = sp[q] + row; // padded, so s[1] and s[2] exist
                    a0 += (s[0] * xs[q]) * v0;
                    a1 += (s[1] * xs[q]) * v1;
                    a2 += (s[2] * xs[q]) * v2;
                }
                xout[row] = a0;
                if (n > 1) xout[row + 1] = a1;
                if (n > 2) xout[row + 2] = a2;
            }
        }
    }
}

// Integer pass sums for the selftest hash.
static void collect_partials(const GxMat *g, int p0, int np, int passes, int draws, int slots_x, int cw) {
    for (int q = 0; q < np; q++) {
        int ls = q * draws, hs = ls + 1;
        for (int pix = 0; pix < g->npx; pix++) {
            int px = pix % g->pw, py = pix / g->pw, lo[3], hi[3];
            read_rgb(g_copy, cw, (ls % slots_x) * g->pw + px, (ls / slots_x) * g->ph + py, lo);
            read_rgb(g_copy, cw, (hs % slots_x) * g->pw + px, (hs / slots_x) * g->ph + py, hi);
            for (int c = 0; c < 3 && 3 * pix + c < g->rows; c++)
                g_partials[(size_t)(3 * pix + c) * passes + p0 + q] =
                    (int16_t)(draws == 1 ? lo[c] - 128 : 8 * (hi[c] - 128) + (lo[c] & 7));
        }
    }
}

// The blended classifier (BlendMatrix in tools/tev_sim/gxmatmul.py has the details). Every pass draws
// over the same pixels: pass 0 replaces them with A0 + C0 + sum, odd passes flip every stage's sign and
// subtract (C0 - sum), even passes add (C0 + sum). The C0s cancel in pairs, so the framebuffer ends
// with A0 + the whole row's sum, and the CPU reads one byte per row.
static void run_blend(GxMat *g, float *xout, const float *x) {
    static uint8_t kb[MAX_PASSES * PLANES], neg[MAX_PASSES * PLANES];
    int passes = g->cols / 16, base = BLEND_A0 + (passes & 1 ? BLEND_C0 : 0);
    const float *lut = g_lut_bl[passes & 1];
    float xscale;
    uint64_t ta = gettime();
    load_layout(LAYOUT_BLEND);
    quantize(x, g->cols, BLEND_HEADROOM, kb, neg, &xscale);
    GX_SetNumTevStages(16);
    for (int p = 0; p < passes; p++) {
        int odd = p & 1;
        set_pass(kb + 16 * p, neg + 16 * p, 16, odd, GX_CC_C0, 1);
        if (p < 2) {
            s16 c = p == 0 ? BLEND_A0 + BLEND_C0 : BLEND_C0;
            GXColorS10 c0 = {c, c, c, c};
            GX_SetTevColorS10(GX_TEVREG0, c0);
        }
        GX_SetBlendMode(p == 0 ? GX_BM_NONE : odd ? GX_BM_SUBTRACT : GX_BM_BLEND, GX_BL_ONE, GX_BL_ONE, GX_LO_CLEAR);
        set_textures(g, 2 * p, 2, p == 0);
        draw_quad(0, 0, g->pw, g->ph);
    }
    GX_SetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR);
    GX_SetTexCopySrc(0, 0, g->pw, g->ph);
    GX_SetTexCopyDst(g->pw, g->ph, GX_TF_RGBA8, GX_FALSE);
    g_copy = copy_buf(g->pw * g->ph * 4);
    GX_CopyTex(g_copy, GX_FALSE);
    uint64_t tb = gettime();
    GX_DrawDone();
    uint64_t tc = gettime();
    DCInvalidateRange(g_copy, g->pw * g->ph * 4);
    g_last_copy = g_copy;
    g_last_cw = g->pw;
    g_last_ch = g->ph;

    for (int ty = 0; ty < g->ph / 4; ty++) {
        for (int tx = 0; tx < g->pw / 4; tx++) {
            const uint8_t *t = g_copy + ((size_t)ty * (g->pw / 4) + tx) * 64;
            for (int i = 0; i < 16; i++) {
                int pix = (ty * 4 + (i >> 2)) * g->pw + tx * 4 + (i & 3);
                if (pix >= g->npx) continue;
                int row = 3 * pix, n = g->rows - row < 3 ? g->rows - row : 3;
                uint8_t b[3] = {t[2 * i + 1], t[32 + 2 * i], t[33 + 2 * i]};
                for (int c = 0; c < n; c++) {
                    if (xout) xout[row + c] = (g->srow[row + c] * xscale) * lut[b[c]];
                    if (g_partials) g_partials[row + c] = (int16_t)(b[c] - base);
                }
            }
        }
    }
    uint64_t td = gettime();
    g_stats.draw_ticks += tb - ta;
    g_stats.wait_ticks += tc - tb;
    g_stats.reduce_ticks += td - tc;
    g_stats.draws += passes;
    g_stats.matmuls++;
}

// acc[0..2] += xs[q] * (byte - 128) for q < n (n >= 1), where pass q's bytes are R at r + q * stride and
// G, B at gb + q * stride. psq_l turns bytes into floats (GQR6 is set to u8), and ps_madds0 updates two rows
// with one rounding, like fmadds. TevMatrix.__call__ in tools/tev_sim/gxmatmul.py replays these operations.
// The float accumulators load and store with lfs/stfs and merges: acc + 1 is only 4-byte aligned.
static inline void sum_pixel(const uint8_t *r, const uint8_t *gb, int stride, const float *xs, int n, float *acc) {
    static const float k128 = 128.0f;
    double a0, a12, a2, vr, vgb, xq, k;
    int off = 0;
    __asm__ volatile("lfs %[a0], 0(%[acc])\n\t"
                     "lfs %[a12], 4(%[acc])\n\t"
                     "lfs %[a2], 8(%[acc])\n\t"
                     "ps_merge00 %[a12], %[a12], %[a2]\n\t"
                     "lfs %[k], 0(%[k128])\n\t"
                     "ps_merge00 %[k], %[k], %[k]\n\t"
                     "mtctr %[n]\n"
                     "1:\n\t"
                     "psq_lx %[vr], %[r], %[off], 1, 6\n\t" // R (the other half is 1.0, unused)
                     "psq_lx %[vgb], %[gb], %[off], 0, 6\n\t"
                     "lfs %[xq], 0(%[xs])\n\t"
                     "ps_sub %[vr], %[vr], %[k]\n\t"
                     "ps_sub %[vgb], %[vgb], %[k]\n\t"
                     "addi %[xs], %[xs], 4\n\t"
                     "add %[off], %[off], %[stride]\n\t"
                     "ps_madds0 %[a0], %[vr], %[xq], %[a0]\n\t"
                     "ps_madds0 %[a12], %[vgb], %[xq], %[a12]\n\t"
                     "bdnz 1b\n\t"
                     "stfs %[a0], 0(%[acc])\n\t"
                     "stfs %[a12], 4(%[acc])\n\t"
                     "ps_merge11 %[a2], %[a12], %[a12]\n\t"
                     "stfs %[a2], 8(%[acc])\n\t"
                     : [a0] "=&f"(a0), [a12] "=&f"(a12), [a2] "=&f"(a2), [vr] "=&f"(vr), [vgb] "=&f"(vgb),
                       [xq] "=&f"(xq), [k] "=&f"(k), [xs] "+b"(xs), [off] "+r"(off)
                     : [r] "b"(r), [gb] "b"(gb), [stride] "r"(stride), [n] "r"(n), [acc] "b"(acc), [k128] "b"(&k128)
                     : "ctr", "memory");
}

// clamp8 with one weight scale per row, in chunks of CHUNK passes so the CPU sums one chunk while the
// GPU draws the next. Chunks alternate between two framebuffer columns and two halves of a copy buffer.
// Pass q of a chunk draws at (column, q * ph), so its part of the chunk's copy is one block, stride
// bytes after the previous pass's, and sum_pixel walks the passes for one pixel at a time. The sums
// stay unscaled in g_acc until the end applies the row scales.
#define CHUNK 6 // must match CHUNK in tools/tev_sim/check_probe.py

static void issue_chunk(GxMat *g, const float *x, int p0, int np, int col, uint8_t *dst) {
    uint8_t kb[16], neg[16];
    for (int i = 0; i < np; i++) {
        int p = p0 + i;
        quantize(x + 16 * p, 16, g_headroom, kb, neg, &g_xscale[p]);
        set_pass(kb, neg, 16, 0, GX_CC_HALF, 1);
        set_textures(g, 2 * p, 2, p == 0);
        draw_quad(col * g->pw, i * g->ph, g->pw, g->ph);
    }
    GX_SetTexCopySrc(col * g->pw, 0, g->pw, np * g->ph);
    GX_SetTexCopyDst(g->pw, np * g->ph, GX_TF_RGBA8, GX_FALSE);
    GX_CopyTex(dst, GX_FALSE);
}

static void sum_chunk(const GxMat *g, const uint8_t *buf, int p0, int np, int passes, int stride) {
    if (!g_partials) {
        CAST_SetGQR6(GQR_TYPE_U8, 0);
        for (int ty = 0; ty < g->ph / 4; ty++) {
            for (int tx = 0; tx < g->pw / 4; tx++) {
                const uint8_t *t = buf + ((size_t)ty * (g->pw / 4) + tx) * 64;
                for (int i = 0; i < 16; i++) {
                    int pix = (ty * 4 + (i >> 2)) * g->pw + tx * 4 + (i & 3);
                    if (pix < g->npx) sum_pixel(t + 2 * i + 1, t + 32 + 2 * i, stride, g_xscale + p0, np, g_acc + 3 * pix);
                }
            }
        }
        return;
    }
    for (int q = 0; q < np; q++) {
        for (int pix = 0; pix < g->npx; pix++) {
            int rgb[3];
            read_rgb(buf, g->pw, pix % g->pw, q * g->ph + pix / g->pw, rgb);
            for (int c = 0; c < 3 && 3 * pix + c < g->rows; c++)
                g_partials[(size_t)(3 * pix + c) * passes + p0 + q] = (int16_t)(rgb[c] - 128);
        }
    }
}

static void run_rows(GxMat *g, float *xout, const float *x) {
    int passes = g->cols / 16, stride = g->pw * g->ph * 4;
    int chunk = CHUNK < EFB_H / g->ph ? CHUNK : EFB_H / g->ph, nchunks = (passes + chunk - 1) / chunk;
    uint8_t *base = copy_buf((size_t)2 * chunk * stride), *buf[2] = {base, base + (size_t)chunk * stride};
    uint16_t tok[2] = {0, 0};
    load_layout(GXM_CLAMP8);
    GX_SetNumTevStages(16);
    memset(g_acc, 0, 3 * g->npx * sizeof(float));

    for (int c = 0; c <= nchunks; c++) {
        uint64_t ta = gettime();
        if (c < nchunks) {
            int p0 = c * chunk, np = passes - p0 < chunk ? passes - p0 : chunk;
            issue_chunk(g, x, p0, np, c & 1, buf[c & 1]);
            tok[c & 1] = put_token();
            g_stats.draws += np;
        }
        uint64_t tb = gettime(), tc = tb;
        if (c > 0) { // the previous chunk, while the GPU works on this one
            int p0 = (c - 1) * chunk, np = passes - p0 < chunk ? passes - p0 : chunk;
            uint8_t *b = buf[(c - 1) & 1];
            wait_token(tok[(c - 1) & 1]);
            tc = gettime();
            DCInvalidateRange(b, stride * np);
            sum_chunk(g, b, p0, np, passes, stride);
            g_last_copy = b;
            g_last_cw = g->pw;
            g_last_ch = np * g->ph;
        }
        uint64_t td = gettime();
        g_stats.draw_ticks += tb - ta;
        g_stats.wait_ticks += tc - tb;
        g_stats.reduce_ticks += td - tc;
    }
    if (xout)
        for (int r = 0; r < g->rows; r++) xout[r] = g->srow[r] * g_acc[r];
    g_stats.matmuls++;
}

static void run(GxMat *g, float *xout, const float *x) {
    if (g->kind == SCALES_BLEND) {
        run_blend(g, xout, x);
        return;
    }
    if (g_mode == GXM_CLAMP8 && g->per_row) {
        run_rows(g->per_row, xout, x);
        return;
    }
    int clamp8 = g_mode == GXM_CLAMP8;
    int pc = clamp8 ? 16 : 8, draws = clamp8 ? 1 : 2;
    int passes = g->cols / pc, tex_per_pass = pc / PLANES;
    int slots_x = EFB_W / g->pw, slots = slots_x * (EFB_H / g->ph);
    int per_batch = slots / draws;

    load_layout(g_mode);
    if (xout) memset(xout, 0, g->rows * sizeof(float));

    uint8_t kb[16], neg[16];
    for (int p0 = 0; p0 < passes; p0 += per_batch) {
        int np = passes - p0 < per_batch ? passes - p0 : per_batch;
        uint64_t ta = gettime();
        for (int i = 0; i < np; i++) {
            int p = p0 + i;
            quantize(x + p * pc, pc, g_headroom, kb, neg, &g_xscale[p]);
            set_pass(kb, neg, pc, 0, clamp8 ? GX_CC_HALF : GX_CC_ZERO, clamp8);
            set_textures(g, p * tex_per_pass, tex_per_pass, p == 0);
            int slot = i * draws;
            if (clamp8) {
                GX_SetNumTevStages(16);
                draw_quad((slot % slots_x) * g->pw, (slot / slots_x) * g->ph, g->pw, g->ph);
            } else {
                GX_SetNumTevStages(8); // low byte of the exact sum
                draw_quad((slot % slots_x) * g->pw, (slot / slots_x) * g->ph, g->pw, g->ph);
                GX_SetNumTevStages(11); // floor(sum / 8) + 128
                draw_quad(((slot + 1) % slots_x) * g->pw, ((slot + 1) / slots_x) * g->ph, g->pw, g->ph);
            }
        }
        int used = np * draws;
        int cw = (used < slots_x ? used : slots_x) * g->pw;
        int ch = ((used + slots_x - 1) / slots_x) * g->ph;
        GX_SetTexCopySrc(0, 0, cw, ch);
        GX_SetTexCopyDst(cw, ch, GX_TF_RGBA8, GX_FALSE);
        g_copy = copy_buf((size_t)cw * ch * 4);
        GX_CopyTex(g_copy, GX_FALSE);
        uint64_t tb = gettime();
        GX_DrawDone();
        uint64_t tc = gettime();
        DCInvalidateRange(g_copy, cw * ch * 4);
        g_last_copy = g_copy;
        g_last_cw = cw;
        g_last_ch = ch;
        if (xout) reduce_batch(g, xout, p0, np, pc, draws, slots_x, cw);
        if (g_partials) collect_partials(g, p0, np, passes, draws, slots_x, cw);
        uint64_t td = gettime();
        g_stats.draw_ticks += tb - ta;
        g_stats.wait_ticks += tc - tb;
        g_stats.reduce_ticks += td - tc;
        g_stats.draws += used;
    }
    g_stats.matmuls++;
}

void gxm_matmul(void *gpu, float *xout, const float *x) { run((GxMat *)gpu, xout, x); }

uint32_t gxm_partials_hash(void *gpu, const float *x) {
    GxMat *g = (GxMat *)gpu;
    int passes = g->kind == SCALES_BLEND ? 1 : g->cols / (g_mode == GXM_CLAMP8 ? 16 : 8); // blended: one sum per row
    g_partials = (int16_t *)calloc((size_t)g->rows * passes, sizeof(int16_t));
    if (!g_partials) return 0;
    run(g, NULL, x);
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < (size_t)g->rows * passes; i++) {
        uint16_t v = (uint16_t)g_partials[i];
        h = (h ^ (v & 0xff)) * 16777619u;
        h = (h ^ (v >> 8)) * 16777619u;
    }
    free(g_partials);
    g_partials = NULL;
    return h;
}

const uint8_t *gxm_last_copy(int *w, int *h) {
    *w = g_last_cw;
    *h = g_last_ch;
    return g_last_copy;
}

void gxm_set_raw_issue(int on) { g_raw = on; }

int gxm_tokens_failed(void) { return g_tokens_failed; }

const void *gxm_texture_memory(void *gpu, size_t *bytes) {
    GxMat *g = (GxMat *)gpu;
    *bytes = (size_t)g->tex_bytes * g->ntex;
    return g->tex;
}

float gxm_time_draws(void *gpu, int stages, int fmt, int tiny) {
    static uint8_t small[32 * 4 * 4] __attribute__((aligned(32))); // 32 x 4 RGBA8, sampled over and over
    GxMat *g = (GxMat *)gpu;
    int passes = g->cols / 16, tw = PLANES * g->pw;
    uint8_t kb[16], neg[16];
    memset(kb, 100, sizeof(kb));
    memset(neg, 0, sizeof(neg));
    g_layout = -1;
    load_layout(LAYOUT_BLEND);
    if (fmt < 0)
        for (int k = 0; k < 16; k++) GX_SetTevOrder(GX_TEVSTAGE0 + k, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLORNULL);
    GXColorS10 c0 = {64, 64, 64, 64};
    GX_SetTevColorS10(GX_TEVREG0, c0);
    GX_SetNumTevStages(stages);
    GXTexObj obj[2];
    if (tiny) {
        memset(small, 77, sizeof(small));
        DCFlushRange(small, sizeof(small));
        GX_InitTexObj(&obj[0], small, 32, 4, GX_TF_RGBA8, GX_REPEAT, GX_REPEAT, GX_FALSE);
        GX_InitTexObjFilterMode(&obj[0], GX_NEAR, GX_NEAR);
        GX_LoadTexObj(&obj[0], GX_TEXMAP0);
        GX_LoadTexObj(&obj[0], GX_TEXMAP1);
    }
    GX_DrawDone();
    uint64_t t0 = gettime();
    for (int p = 0; p < passes; p++) {
        int odd = p & 1;
        set_pass(kb, neg, 16, odd, GX_CC_C0, 1);
        GX_SetBlendMode(p == 0 ? GX_BM_NONE : odd ? GX_BM_SUBTRACT : GX_BM_BLEND, GX_BL_ONE, GX_BL_ONE, GX_LO_CLEAR);
        if (!tiny && fmt == GX_TF_RGBA8) {
            set_textures(g, 2 * p, 2, p == 0);
        } else if (!tiny && fmt >= 0) { // the same memory read as a smaller texel format
            for (int t = 0; t < 2; t++) {
                GX_InitTexObj(&obj[t], g->tex + (size_t)(2 * p + t) * g->tex_bytes, tw, g->ph, fmt, GX_CLAMP, GX_CLAMP,
                              GX_FALSE);
                GX_InitTexObjFilterMode(&obj[t], GX_NEAR, GX_NEAR);
                GX_LoadTexObj(&obj[t], GX_TEXMAP0 + t);
            }
        }
        draw_quad(0, 0, g->pw, g->ph);
    }
    GX_DrawDone();
    uint64_t t1 = gettime();
    GX_SetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR);
    g_layout = -1;
    return ticks_to_microsecs(t1 - t0) / 1000.0f;
}

void gxm_forget_layout(void) { g_layout = -1; }

void gxm_get_stats(GxmStats *out) { *out = g_stats; }
void gxm_reset_stats(void) { memset(&g_stats, 0, sizeof(g_stats)); }
