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

#include "gx_matmul.h"

#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <gccore.h>
#include <ogc/lwp_watchdog.h>

#define FIFO_SIZE (256 * 1024)
#define EFB_W 640
#define EFB_H 480
#define PLANES 8        // input columns per texture, side by side
#define PLANE_W_MAX 128 // so a texture is at most 8 * 128 = 1024 texels wide
#define MAX_PASSES 128  // up to 1024 input columns

typedef struct {
    int rows, cols, gs;
    int npx;            // output pixels per pass: ceil(rows / 3)
    int pw, ph;         // one pass's framebuffer region, both multiples of 4
    int ntex;           // one texture per 8 input columns
    uint32_t tex_bytes; // per texture
    uint8_t *tex;
    GXTexObj *obj;
    float *st;          // weight scales transposed to (cols / gs, 3 * npx) so the readback streams them
} GxMat;

static int g_ready;
static int g_mode = GXM_CLAMP8;
static float g_headroom = 0.5f;
static int g_layout = -1;      // which static TEV setup is loaded
static uint8_t *g_readbuf;     // framebuffer copies land here
static float g_xscale[MAX_PASSES]; // per-pass activation scale
static int16_t *g_partials;    // only while hashing
static int g_last_cw, g_last_ch; // size of the latest framebuffer copy (gx_probe.c reads it)
static size_t g_bytes;
// framebuffer byte -> pass sum, without PowerPC's slow int-to-float conversion
static float g_lut_c8[256], g_lut_hi[256], g_lut_lo[256];
static GxmStats g_stats;

int gxm_init(void) {
    if (g_ready) return 0;
    void *fifo = memalign(32, FIFO_SIZE);
    g_readbuf = (uint8_t *)memalign(32, EFB_W * EFB_H * 4);
    if (!fifo || !g_readbuf) return llm_set_error("out of memory setting up the GPU");
    memset(fifo, 0, FIFO_SIZE);
    GX_Init(fifo, FIFO_SIZE);
    for (int b = 0; b < 256; b++) {
        g_lut_c8[b] = (float)(b - 128);
        g_lut_hi[b] = (float)(8 * (b - 128));
        g_lut_lo[b] = (float)(b & 7);
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
    int product = mode == GXM_CLAMP8 ? 16 : 8;
    for (int k = 0; k < product; k++) {
        u8 st = GX_TEVSTAGE0 + k;
        GX_SetTevOrder(st, GX_TEXCOORD0 + (k & 7), GX_TEXMAP0 + (k >> 3), GX_COLORNULL);
        // Stage 0 starts the sum from a constant: real hardware doesn't reset PREV for each pixel
        // (it holds whatever an earlier pixel left there), so no stage may read PREV before writing it.
        u8 d = k > 0 ? GX_CC_CPREV : mode == GXM_CLAMP8 ? GX_CC_HALF : GX_CC_ZERO;
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

static uint8_t wu(const LlmMatrix *w, int rows, int cols, int row, int col) {
    return row < rows ? (uint8_t)(w->q[(size_t)row * cols + col] + 128) : 128;
}

void *gxm_upload(const LlmModel *m, const LlmMatrix *w, int rows, int cols) {
    if (m->dtype != LLM_DTYPE_Q8 || m->gs % 16 || cols % 16 || cols / 8 > MAX_PASSES) {
        llm_set_error("the GPU path needs int8 weights with a group size that's a multiple of 16");
        return NULL;
    }
    GxMat *g = (GxMat *)calloc(1, sizeof(GxMat));
    if (!g) { llm_set_error("out of memory for GPU matrix"); return NULL; }
    g->rows = rows;
    g->cols = cols;
    g->gs = m->gs;
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
    int groups = cols / g->gs, padded_rows = 3 * g->npx;
    g->tex = (uint8_t *)memalign(32, (size_t)g->tex_bytes * g->ntex);
    g->obj = (GXTexObj *)malloc(sizeof(GXTexObj) * g->ntex);
    g->st = (float *)calloc((size_t)groups * padded_rows, sizeof(float));
    if (!g->tex || !g->obj || !g->st) {
        llm_set_error("out of memory for GPU textures (%lu KB, %lu KB already uploaded)",
                      (unsigned long)((size_t)g->tex_bytes * g->ntex / 1024), (unsigned long)(g_bytes / 1024));
        free(g->tex);
        free(g->obj);
        free(g->st);
        free(g);
        return NULL;
    }
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < groups; k++) g->st[(size_t)k * padded_rows + r] = w->s[(size_t)r * groups + k];

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
                    tile[2 * i + 1] = wu(w, rows, cols, row, col);
                    tile[32 + 2 * i] = wu(w, rows, cols, row + 1, col);
                    tile[32 + 2 * i + 1] = wu(w, rows, cols, row + 2, col);
                }
            }
        }
        DCFlushRange(base, g->tex_bytes);
        GX_InitTexObj(&g->obj[t], base, tw, g->ph, GX_TF_RGBA8, GX_CLAMP, GX_CLAMP, GX_FALSE);
        GX_InitTexObjFilterMode(&g->obj[t], GX_NEAR, GX_NEAR);
    }
    GX_InvalidateTexAll();
    g_bytes += (size_t)g->tex_bytes * g->ntex + (size_t)groups * padded_rows * sizeof(float);
    return g;
}

int gxm_upload_model(LlmModel *m, int layers, size_t *bytes) {
    LlmConfig *c = &m->c;
    int dim = c->dim, hid = c->hidden_dim, kv = c->dim * c->n_kv_heads / c->n_heads;
    m->gpu_matmul = gxm_matmul;
    int rc = 0;
    if (!m->cls.gpu && !(m->cls.gpu = gxm_upload(m, &m->cls, c->vocab_size, dim))) rc = -1;
    for (int l = 0; rc == 0 && layers && l < c->n_layers; l++) {
        LlmLayer *L = &m->layers[l];
        struct { LlmMatrix *w; int rows, cols; } mats[7] = {
            {&L->wq, dim, dim}, {&L->wk, kv, dim}, {&L->wv, kv, dim}, {&L->wo, dim, dim},
            {&L->w1, hid, dim}, {&L->w2, dim, hid}, {&L->w3, hid, dim}};
        for (int i = 0; i < 7 && rc == 0; i++)
            if (!mats[i].w->gpu && !(mats[i].w->gpu = gxm_upload(m, mats[i].w, mats[i].rows, mats[i].cols))) rc = -1;
    }
    if (bytes) *bytes = g_bytes;
    return rc;
}

// Per-pass activation quantization. Must match quantize_x in gxmatmul.py exactly.
static void quantize_pass(const float *x, int base, int pc, uint8_t *kb, uint8_t *neg, float *xscale) {
    float xmax = 0.0f;
    for (int k = 0; k < pc; k++)
        if (fabsf(x[base + k]) > xmax) xmax = fabsf(x[base + k]);
    if (xmax == 0.0f) xmax = 1.0f;
    for (int k = 0; k < pc; k++) {
        float v = x[base + k];
        float t = fabsf(v) / xmax * (256.0f * g_headroom);
        int cp = (int)rintf(t);
        if (cp > 256) cp = 256;
        if (cp == 128) cp = t >= 128.0f ? 129 : 127; // c' = c + (c >> 7) never equals 128
        kb[k] = (uint8_t)(cp - (cp >= 129));
        neg[k] = v < 0.0f;
    }
    *xscale = xmax / g_headroom;
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
    return g_readbuf + ((size_t)((slot / slots_x) * g->ph / 4) * (cw / 4) + (slot % slots_x) * g->pw / 4) * 64;
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
            read_rgb(g_readbuf, cw, (ls % slots_x) * g->pw + px, (ls / slots_x) * g->ph + py, lo);
            read_rgb(g_readbuf, cw, (hs % slots_x) * g->pw + px, (hs / slots_x) * g->ph + py, hi);
            for (int c = 0; c < 3 && 3 * pix + c < g->rows; c++)
                g_partials[(size_t)(3 * pix + c) * passes + p0 + q] =
                    (int16_t)(draws == 1 ? lo[c] - 128 : 8 * (hi[c] - 128) + (lo[c] & 7));
        }
    }
}

static void run(GxMat *g, float *xout, const float *x) {
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
            quantize_pass(x, p * pc, pc, kb, neg, &g_xscale[p]);
            for (int kc = 0; kc < pc / 4; kc++) {
                GXColor col = {kb[4 * kc], kb[4 * kc + 1], kb[4 * kc + 2], kb[4 * kc + 3]};
                GX_SetTevKColor(GX_KCOLOR0 + kc, col);
            }
            for (int k = 0; k < pc; k++)
                GX_SetTevColorOp(GX_TEVSTAGE0 + k, neg[k] ? GX_TEV_SUB : GX_TEV_ADD,
                                 neg[k] ? GX_TB_ADDHALF : GX_TB_SUBHALF, GX_CS_SCALE_1, clamp8 && k == pc - 1,
                                 GX_TEVPREV);
            for (int t = 0; t < tex_per_pass; t++) GX_LoadTexObj(&g->obj[p * tex_per_pass + t], GX_TEXMAP0 + t);
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
        GX_CopyTex(g_readbuf, GX_FALSE);
        uint64_t tb = gettime();
        GX_DrawDone();
        uint64_t tc = gettime();
        DCInvalidateRange(g_readbuf, cw * ch * 4);
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
    int passes = g->cols / (g_mode == GXM_CLAMP8 ? 16 : 8);
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
    return g_readbuf;
}

void gxm_forget_layout(void) { g_layout = -1; }

void gxm_get_stats(GxmStats *out) { *out = g_stats; }
void gxm_reset_stats(void) { memset(&g_stats, 0, sizeof(g_stats)); }
