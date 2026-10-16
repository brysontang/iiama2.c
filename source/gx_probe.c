// gx_probe.c: hardware probes for the GPU path, for when the Wii disagrees with Dolphin.
//
// Runs when SD:/llama/probe.txt exists. Each probe draws something whose
// framebuffer bytes are known in advance and dumps the raw copy to
// SD:/llama/probe.bin; tools/tev_sim/check_probe.py says what came out
// differently. The probes split the matmul into its parts: texture sampling,
// the konst registers, a color register's value and how the framebuffer write
// wraps, one multiply stage, and the exact11 halving stages. The real matmuls
// from the selftest are dumped too, and hashed more than once to see if the
// GPU is even deterministic.
#include "gx_probe.h"

#include <malloc.h>
#include <stdio.h>
#include <string.h>

#include <gccore.h>

#include "gx_matmul.h"

static uint8_t *g_buf; // framebuffer copies for the probes' own draws
static FILE *g_out;
static GxpLogFn g_log;

static void section(const char *name, const uint8_t *data, int w, int h) {
    char tag[8] = {0};
    size_t len = strlen(name);
    memcpy(tag, name, len < sizeof(tag) ? len : sizeof(tag));
    uint32_t wh[2] = {(uint32_t)w, (uint32_t)h};
    fwrite(tag, 1, sizeof(tag), g_out);
    fwrite(wh, sizeof(uint32_t), 2, g_out); // big-endian on the Wii
    fwrite(data, 1, (size_t)w * h * 4, g_out);
    g_log("probe %-8s %dx%d\n", name, w, h);
}

static void copy_out(const char *name, int w, int h) {
    GX_SetTexCopySrc(0, 0, w, h);
    GX_SetTexCopyDst(w, h, GX_TF_RGBA8, GX_FALSE);
    GX_CopyTex(g_buf, GX_FALSE);
    GX_DrawDone();
    DCInvalidateRange(g_buf, w * h * 4);
    section(name, g_buf, w, h);
}

static void quad(int x0, int y0, int w, int h, float s0, float s1, float t0, float t1) {
    GX_Begin(GX_QUADS, GX_VTXFMT0, 4);
    GX_Position2f32(x0, y0);
    GX_TexCoord2f32(s0, t0);
    GX_Position2f32(x0 + w, y0);
    GX_TexCoord2f32(s1, t0);
    GX_Position2f32(x0 + w, y0 + h);
    GX_TexCoord2f32(s1, t1);
    GX_Position2f32(x0, y0 + h);
    GX_TexCoord2f32(s0, t1);
    GX_End();
}

static void put_texel(uint8_t *base, int tw, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    uint8_t *tile = base + ((((y >> 2) * (tw >> 2)) + (x >> 2)) << 6);
    int i = ((y & 3) << 2) | (x & 3);
    tile[2 * i] = 0;
    tile[2 * i + 1] = r;
    tile[32 + 2 * i] = g;
    tile[32 + 2 * i + 1] = b;
}

// One TEV stage, color = op(d + bias, lerp(a, b, c)), alpha unused.
static void stage(int st, u8 a, u8 b, u8 c, u8 d, u8 op, u8 bias, u8 scale, u8 clamp) {
    GX_SetTevColorIn(st, a, b, c, d);
    GX_SetTevColorOp(st, op, bias, scale, clamp, GX_TEVPREV);
    GX_SetTevAlphaIn(st, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO);
    GX_SetTevAlphaOp(st, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GX_SetTevSwapMode(st, GX_TEV_SWAP0, GX_TEV_SWAP0);
}

// Real hardware doesn't reset PREV per pixel, so known values come in through C0, which no stage writes.
static void c0_init(s16 r, s16 g, s16 b) {
    GXColorS10 c = {r, g, b, 0};
    GX_SetTevColorS10(GX_TEVREG0, c);
}

// The real thing: raw framebuffer copies of selftest matmuls, and repeated hashes.
static void probe_matmuls(LlmModel *m, const float *x) {
    static const struct { int mode; float headroom; const char *name; } modes[] = {
        {GXM_CLAMP8, 0.5f, "c8"}, {GXM_EXACT11, 1.0f, "e11"}};
    for (int i = 0; i < 2; i++) {
        gxm_set_mode(modes[i].mode, modes[i].headroom);
        struct { void *gpu; const char *name; } mats[2] = {{m->layers[0].w1.gpu, "w1"}, {m->cls.gpu, "cls"}};
        for (int k = 0; k < 2; k++) {
            if (!mats[k].gpu || (modes[i].mode == GXM_EXACT11 && k == 1)) continue; // cls exact11 takes several copies
            char name[16];
            uint32_t h[3];
            for (int r = 0; r < 3; r++) h[r] = gxm_partials_hash(mats[k].gpu, x);
            g_log("hash %s %s: %08lx %08lx %08lx\n", mats[k].name, modes[i].name, (unsigned long)h[0],
                  (unsigned long)h[1], (unsigned long)h[2]);
            int w, hh;
            const uint8_t *buf = gxm_last_copy(&w, &hh);
            snprintf(name, sizeof(name), "%s_%s", mats[k].name, modes[i].name);
            section(name, buf, w, hh);
        }
    }
}

// Texture sampling exactly as the matmul does it: 8 planes side by side, texcoord j shifted by j/8.
// Texel (plane, px, py) = (py * 16 + px, plane * 32 + px, plane * 32 + py).
static void probe_texture(void) {
    const int pw = 16, ph = 16, tw = 8 * pw;
    uint8_t *tex = (uint8_t *)memalign(32, tw * ph * 4);
    for (int j = 0; j < 8; j++)
        for (int py = 0; py < ph; py++)
            for (int px = 0; px < pw; px++)
                put_texel(tex, tw, j * pw + px, py, py * 16 + px, j * 32 + px, j * 32 + py);
    DCFlushRange(tex, tw * ph * 4);
    GXTexObj obj;
    GX_InitTexObj(&obj, tex, tw, ph, GX_TF_RGBA8, GX_CLAMP, GX_CLAMP, GX_FALSE);
    GX_InitTexObjFilterMode(&obj, GX_NEAR, GX_NEAR);
    GX_LoadTexObj(&obj, GX_TEXMAP0);
    GX_InvalidateTexAll();
    GX_SetNumTevStages(1);
    stage(0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE);
    for (int j = 0; j < 8; j++) {
        GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0 + j, GX_TEXMAP0, GX_COLORNULL);
        quad(j * pw, 0, pw, ph, 0.0f, 1.0f / 8, 0.0f, 1.0f);
    }
    copy_out("tex", tw, ph);
    free(tex);
}

// Konst registers written the way the matmul writes them (all four, every pass), then read back
// one channel at a time with the matmul's KCSEL mapping. Round r, register n, channel ch holds
// (r * 37 + n * 16 + ch * 4 + 1) & 255.
static void probe_konst(void) {
    GX_SetNumTevStages(1);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLORNULL);
    stage(0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_KONST, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE);
    for (int r = 0; r < 8; r++) {
        for (int n = 0; n < 4; n++) {
            u8 v[4];
            for (int ch = 0; ch < 4; ch++) v[ch] = (u8)((r * 37 + n * 16 + ch * 4 + 1) & 255);
            GXColor col = {v[0], v[1], v[2], v[3]};
            GX_SetTevKColor(GX_KCOLOR0 + n, col);
        }
        for (int k = 0; k < 16; k++) {
            GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0_R + 4 * (k & 3) + (k >> 2));
            quad(k * 4, r * 4, 4, 4, 0, 0, 0, 0);
        }
    }
    copy_out("konst", 64, 32);
}

// C0 straight to the framebuffer: row 0 without the stage clamp (does the write wrap like a
// u8 cast, or saturate?), row 1 with it.
static const s16 PREV_VALUES[][3] = {{77, 150, 200}, {-50, 300, 1000}, {-1, 256, 511}, {0, 255, -1024},
                                     {1023, -129, 128}, {-200, 600, 384}, {129, 383, -256}, {640, -640, 3}};
#define N_PREV (int)(sizeof(PREV_VALUES) / sizeof(PREV_VALUES[0]))

static void probe_reg(void) {
    GX_SetNumTevStages(1);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLORNULL);
    for (int clamp = 0; clamp < 2; clamp++) {
        stage(0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_C0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, clamp);
        for (int i = 0; i < N_PREV; i++) {
            c0_init(PREV_VALUES[i][0], PREV_VALUES[i][1], PREV_VALUES[i][2]);
            quad(i * 4, clamp * 4, 4, 4, 0, 0, 0, 0);
        }
    }
    copy_out("reg", 4 * N_PREV, 8);
}

// One matmul stage, lerp(HALF, TEXC, KONST) on top of d = HALF (128), for every texel byte and a spread
// of konst values, both signs. Texel (x, y) = (y * 16 + x, 255 - that, that * 7 & 255).
static const u8 KONST_VALUES[10] = {0, 1, 2, 64, 127, 128, 129, 200, 254, 255};

static void probe_arith(void) {
    const int n = 16;
    uint8_t *tex = (uint8_t *)memalign(32, n * n * 4);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            int v = y * 16 + x;
            put_texel(tex, n, x, y, v, 255 - v, (v * 7) & 255);
        }
    DCFlushRange(tex, n * n * 4);
    GXTexObj obj;
    GX_InitTexObj(&obj, tex, n, n, GX_TF_RGBA8, GX_CLAMP, GX_CLAMP, GX_FALSE);
    GX_InitTexObjFilterMode(&obj, GX_NEAR, GX_NEAR);
    GX_LoadTexObj(&obj, GX_TEXMAP0);
    GX_InvalidateTexAll();
    GX_SetNumTevStages(1);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, GX_TEXMAP0, GX_COLORNULL); // TEXMTX0 is a zero shift
    GX_SetTevKColorSel(GX_TEVSTAGE0, GX_TEV_KCSEL_K0_R);
    for (int sub = 0; sub < 2; sub++) {
        stage(0, GX_CC_HALF, GX_CC_TEXC, GX_CC_KONST, GX_CC_HALF, sub ? GX_TEV_SUB : GX_TEV_ADD,
              sub ? GX_TB_ADDHALF : GX_TB_SUBHALF, GX_CS_SCALE_1, GX_TRUE);
        for (int i = 0; i < 10; i++) {
            u8 k = KONST_VALUES[i];
            GXColor col = {k, k, k, k};
            GX_SetTevKColor(GX_KCOLOR0, col);
            quad((sub * 10 + i) * n, 0, n, n, 0.0f, 1.0f, 0.0f, 1.0f);
        }
    }
    copy_out("arith", 20 * n, n);
    free(tex);
}

// exact11's draw 2 tail: three halving stages, the last adding the 1/8 konst and clamping,
// on known values (copied from C0 into PREV by stage 0). Expect floor(v / 8) + 128.
static void probe_div(void) {
    GX_SetNumTevStages(4);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLORNULL);
    stage(0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_C0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_FALSE);
    for (int k = 1; k < 4; k++) {
        GX_SetTevOrder(GX_TEVSTAGE0 + k, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLORNULL);
        GX_SetTevKColorSel(GX_TEVSTAGE0 + k, GX_TEV_KCSEL_1_8);
        stage(k, k == 3 ? GX_CC_KONST : GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_CPREV, GX_TEV_ADD, GX_TB_ADDHALF,
              GX_CS_DIVIDE_2, k == 3);
    }
    for (int i = 0; i < N_PREV; i++) {
        c0_init(PREV_VALUES[i][0], PREV_VALUES[i][1], PREV_VALUES[i][2]);
        quad(i * 4, 0, 4, 4, 0, 0, 0, 0);
    }
    copy_out("div", 4 * N_PREV, 4);
}

int gxp_run(LlmModel *m, const char *bin_path, GxpLogFn log) {
    g_log = log;
    g_out = fopen(bin_path, "wb");
    g_buf = (uint8_t *)memalign(32, 640 * 64 * 4);
    if (!g_out || !g_buf) {
        log("probe: can't open %s or out of memory\n", bin_path);
        if (g_out) fclose(g_out);
        free(g_buf);
        return -1;
    }
    fwrite("IIPROBE1", 1, 8, g_out);

    int n = m->c.dim > m->c.hidden_dim ? m->c.dim : m->c.hidden_dim;
    float *x = (float *)malloc(n * sizeof(float));
    for (int j = 0; j < n; j++) x[j] = (float)(((j * 37) % 101) - 50) / 16.0f; // the selftest's input
    if (m->cls.gpu) probe_matmuls(m, x);
    free(x);

    probe_texture();
    probe_konst();
    probe_reg();
    probe_arith();
    probe_div();
    gxm_forget_layout();

    fclose(g_out);
    free(g_buf);
    return 0;
}
