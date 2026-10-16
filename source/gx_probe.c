// gx_probe.c: hardware probes for the GPU path, for when the Wii disagrees with Dolphin.
//
// Runs when SD:/llama/probe.txt exists. Each probe draws something whose
// framebuffer bytes are known in advance and dumps the raw copy to
// SD:/llama/probe.bin; tools/tev_sim/check_probe.py says what came out
// differently. The probes split the matmul into its parts: texture sampling,
// the konst registers, a color register's value and how the framebuffer write
// wraps, one multiply stage, the exact11 halving stages, and blending. The real
// matmuls from the selftest are dumped too, and hashed more than once to see if
// the GPU is even deterministic. Then timings: raw register writes vs libogc's
// setters, what limits the TEV's rate, and memory (where things live, CPU
// streaming).
#include "gx_probe.h"

#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <gccore.h>
#include <ogc/lwp_watchdog.h>

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
    fflush(g_out);
    fsync(fileno(g_out));
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
static void dump_matmul(void *gpu, const char *name, const float *x) {
    uint32_t h[3];
    for (int r = 0; r < 3; r++) h[r] = gxm_partials_hash(gpu, x);
    g_log("hash %s: %08lx %08lx %08lx\n", name, (unsigned long)h[0], (unsigned long)h[1], (unsigned long)h[2]);
    int w, hh;
    const uint8_t *buf = gxm_last_copy(&w, &hh);
    section(name, buf, w, hh);
}

// A matmul's float output, raw (big-endian floats, w = rows, h = 1): checks the CPU-side readback
// arithmetic bit for bit, not just the GPU's integer sums.
static void dump_output(void *gpu, const char *name, const float *x, int rows) {
    float *y = (float *)malloc(rows * sizeof(float));
    if (!y) return;
    gxm_matmul(gpu, y, x);
    section(name, (const uint8_t *)y, rows, 1);
    free(y);
}

static void probe_matmuls(LlmModel *m, const float *x) {
    if (m->layers[0].w1.gpu) {
        int hid = m->c.hidden_dim;
        gxm_set_mode(GXM_CLAMP8, 0.5f);
        dump_matmul(m->layers[0].w1.gpu, "w1_c8", x);
        dump_output(m->layers[0].w1.gpu, "w1_c8_y", x, hid);
        gxm_set_mode(GXM_EXACT11, 1.0f);
        dump_matmul(m->layers[0].w1.gpu, "w1_e11", x);
        dump_output(m->layers[0].w1.gpu, "w1e11_y", x, hid);
    }
    dump_matmul(m->cls.gpu, "cls_bl", x); // the blended classifier: one copy, one byte per row
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

// Blending the way the classifier uses it, on known values. Each 4x4 quad gets three draws: a plain
// write of D, then S and T blended. Row 0 adds S and subtracts T; row 1 subtracts S and adds T.
// Expect each step to saturate at 0 and 255.
static const s16 BLEND_VALUES[][3][3] = { // [quad][D, S, T][r, g, b]
    {{10, 100, 200}, {20, 100, 100}, {5, 250, 0}},    {{255, 0, 128}, {1, 0, 127}, {0, 1, 255}},
    {{128, 64, 32}, {127, 191, 223}, {200, 100, 50}}, {{0, 0, 0}, {0, 255, 1}, {1, 255, 0}},
    {{200, 50, 250}, {100, 50, 10}, {150, 49, 255}},  {{77, 33, 99}, {88, 44, 11}, {66, 22, 0}},
    {{255, 255, 255}, {255, 255, 255}, {255, 0, 1}},  {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
#define N_BLEND (int)(sizeof(BLEND_VALUES) / sizeof(BLEND_VALUES[0]))

static void probe_blend(void) {
    GX_SetNumTevStages(1);
    GX_SetTevOrder(GX_TEVSTAGE0, GX_TEXCOORDNULL, GX_TEXMAP_NULL, GX_COLORNULL);
    stage(0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_C0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE);
    for (int row = 0; row < 2; row++) {
        for (int i = 0; i < N_BLEND; i++) {
            for (int step = 0; step < 3; step++) {
                const s16 *v = BLEND_VALUES[i][step];
                u8 mode = step == 0 ? GX_BM_NONE : (step == 1) == (row == 0) ? GX_BM_BLEND : GX_BM_SUBTRACT;
                GX_SetBlendMode(mode, GX_BL_ONE, GX_BL_ONE, GX_LO_CLEAR);
                c0_init(v[0], v[1], v[2]);
                quad(i * 4, row * 4, 4, 4, 0, 0, 0, 0);
            }
        }
    }
    GX_SetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR);
    copy_out("blend", 4 * N_BLEND, 8);
}

// Sums `words` words in order: how fast the CPU streams memory it doesn't have cached.
static uint32_t stream(const uint32_t *w, size_t words) {
    uint32_t s = 0;
    for (size_t i = 0; i < words; i += 8)
        s += w[i] + w[i + 1] + w[i + 2] + w[i + 3] + w[i + 4] + w[i + 5] + w[i + 6] + w[i + 7];
    return s;
}

static unsigned long mb_per_s(size_t bytes, uint64_t t0, uint64_t t1) {
    unsigned long us = ticks_to_microsecs(t1 - t0);
    return us ? (unsigned long)(bytes / us) : 0;
}

// Where the big buffers live (cached addresses: MEM1 at 0x80000000, MEM2 at 0x90000000), and how fast the
// CPU streams each. (Locked-cache DMA into 0xE0000000 was tried here too: on a Wii, libogc's LCLoadData
// ended in a Machine Check exception.)
static void probe_memory(LlmModel *m) {
    size_t tex_bytes = 0;
    const void *tex = m->cls.gpu ? gxm_texture_memory(m->cls.gpu, &tex_bytes) : NULL;
    int w, h;
    const void *copy = gxm_last_copy(&w, &h);
    struct { const char *name; const void *p; size_t bytes; } r[] = {
        {"int8 output layer", m->cls.q, m->cls.q ? (size_t)m->c.vocab_size * m->c.dim : 0},
        {"output layer textures", tex, tex_bytes},
        {"int8 layer 0 wq", m->layers[0].wq.q, 0},
        {"framebuffer copies", copy, 0},
        {"KV cache", m->key_cache, 0},
    };
    volatile uint32_t sink = 0;
    for (int i = 0; i < (int)(sizeof(r) / sizeof(r[0])); i++) {
        if (!r[i].p) continue;
        size_t bytes = (r[i].bytes > (8u << 20) ? (8u << 20) : r[i].bytes) & ~(size_t)31;
        if (bytes < (1u << 20)) {
            g_log("memory: %-22s at %08lx\n", r[i].name, (unsigned long)(uintptr_t)r[i].p);
            continue;
        }
        const uint32_t *p = (const uint32_t *)(((uintptr_t)r[i].p + 31) & ~(uintptr_t)31);
        uint64_t t0 = gettime();
        sink += stream(p, bytes / 4 - 8);
        uint64_t t1 = gettime();
        g_log("memory: %-22s at %08lx, %lu KB streamed: %lu MB/s\n", r[i].name, (unsigned long)(uintptr_t)r[i].p,
              (unsigned long)(bytes >> 10), mb_per_s(bytes, t0, t1));
    }
    (void)sink;
}

// What limits the TEV: the blended output layer's passes with fewer stages, without textures, with one
// small texture that stays cached, and with the same texture memory read as smaller texel formats.
static void probe_rate(LlmModel *m) {
    static const struct { const char *name; int stages, fmt, tiny; } v[] = {
        {"RGBA8, 16 stages", 16, GX_TF_RGBA8, 0},    {"RGBA8, 8 stages", 8, GX_TF_RGBA8, 0},
        {"RGBA8, 4 stages", 4, GX_TF_RGBA8, 0},      {"RGBA8, 1 stage", 1, GX_TF_RGBA8, 0},
        {"no texture, 16 stages", 16, -1, 0},        {"one small texture, 16 stages", 16, GX_TF_RGBA8, 1},
        {"read as RGB565, 16 stages", 16, GX_TF_RGB565, 0}, {"read as IA8, 16 stages", 16, GX_TF_IA8, 0},
        {"read as I8, 16 stages", 16, GX_TF_I8, 0},  {"RGBA8, 16 stages again", 16, GX_TF_RGBA8, 0},
    };
    if (!m->cls.gpu) return;
    for (int i = 0; i < (int)(sizeof(v) / sizeof(v[0])); i++)
        g_log("tev: %-30s %7.3f ms\n", v[i].name, gxm_time_draws(m->cls.gpu, v[i].stages, v[i].fmt, v[i].tiny));
}

// Raw register writes vs libogc's setters: the integer sums must match, and the times show what libogc
// cost. Layer 0's w2 (48 passes) ten times each way, in both output modes.
static void probe_issue(LlmModel *m, const float *x) {
    void *w1 = m->layers[0].w1.gpu, *w2 = m->layers[0].w2.gpu;
    float *y = (float *)malloc(m->c.dim * sizeof(float));
    if (!w1 || !w2 || !y) {
        free(y);
        return;
    }
    for (int mode = 0; mode < 2; mode++) {
        gxm_set_mode(mode ? GXM_EXACT11 : GXM_CLAMP8, mode ? 1.0f : 0.5f);
        for (int raw = 1; raw >= 0; raw--) {
            gxm_set_raw_issue(raw);
            uint32_t h1 = gxm_partials_hash(w1, x), h2 = gxm_partials_hash(w2, x);
            GxmStats st;
            gxm_reset_stats();
            for (int i = 0; i < 10; i++) gxm_matmul(w2, y, x);
            gxm_get_stats(&st);
            g_log("issue %s %-6s: w1 %08lx w2 %08lx | w2 x10: issuing %lu us, waiting %lu us, reading back %lu us\n",
                  mode ? "exact11" : "clamp8", raw ? "raw" : "libogc", (unsigned long)h1, (unsigned long)h2,
                  (unsigned long)ticks_to_microsecs(st.draw_ticks), (unsigned long)ticks_to_microsecs(st.wait_ticks),
                  (unsigned long)ticks_to_microsecs(st.reduce_ticks));
        }
    }
    gxm_set_raw_issue(1);
    gxm_reset_stats();
    free(y);
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
    // a line before each step, so a crash or hang shows on screen (and in probe_out.txt) where it was
    log("probe: matmuls\n");
    if (m->cls.gpu) probe_matmuls(m, x);
    if (gxm_tokens_failed()) log("warning: GPU draw-sync tokens timed out; matmuls fell back to GX_DrawDone\n");
    log("probe: raw issue vs libogc\n");
    if (m->cls.gpu) probe_issue(m, x);
    free(x);
    log("probe: TEV arithmetic\n");

    probe_texture();
    probe_konst();
    probe_reg();
    probe_arith();
    probe_div();
    probe_blend();
    gxm_forget_layout();
    log("probe: TEV rate\n");
    probe_rate(m);
    log("probe: memory\n");
    probe_memory(m);

    fclose(g_out);
    free(g_buf);
    return 0;
}
