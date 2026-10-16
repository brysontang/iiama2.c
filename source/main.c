// iiama2.c: TinyStories-class language models running on the Nintendo Wii.
//
// Controls
//   LEFT / RIGHT  pick a prompt
//   UP / DOWN     temperature
//   + / -         engine: CPU, split (GPU classifier), or GPU
//   A             tell a story
//   B             stop the current story
//   2             save a screenshot to SD:/llama/screenshot.bmp
//   HOME          back to the Homebrew Channel
//
// Expects SD:/llama/model.bin and SD:/llama/tokenizer.bin (made by
// tools/export_wii.py). USB storage works too. If SD:/llama/bench.txt exists,
// it runs every engine on the same greedy story, prints a comparison, and
// writes it to SD:/llama/bench_out.txt (see run_bench). If SD:/llama/autoplay.txt
// exists, it tells one story at boot without waiting for the Wii Remote (see
// read_autoplay).

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <fat.h>
#include <gccore.h>
#include <ogc/lwp_watchdog.h>
#include <wiiuse/wpad.h>

#include "dsp_probe.h"
#include "gx_matmul.h"
#include "gx_probe.h"
#include "llm.h"

static void *xfb = NULL;
static GXRModeObj *rmode = NULL;
static int con_cols = 80, con_rows = 25;

static const char *PROMPTS[] = {
    "Once upon a time",
    "The grumpy turtle king",
    "One day, a little girl named Lily",
    "Tom and his dog went to the park",
    "There was a brave little robot",
    "", // let the model pick
};
#define N_PROMPTS ((int)(sizeof(PROMPTS) / sizeof(PROMPTS[0])))

// Where the matmuls run. tools/tev_sim/eval.py measures what each costs in accuracy. The GPU engines
// all run the classifier blended (coarse logits for the whole vocab) and recompute the top 256 exactly
// on the CPU; out_mode and headroom are for the transformer layers.
typedef struct {
    const char *name;
    int mode, out_mode;
    float headroom;
    int rescore_k;
} Engine;

static const Engine ENGINES[] = {
    {"CPU", LLM_MODE_CPU, GXM_CLAMP8, 0.5f, 0},
    {"split (GPU classifier + CPU top-256)", LLM_MODE_SPLIT, GXM_CLAMP8, 0.5f, 256},
    {"GPU, fast (clamp8)", LLM_MODE_GPU, GXM_CLAMP8, 0.5f, 256},
    {"GPU, exact (exact11)", LLM_MODE_GPU, GXM_EXACT11, 1.0f, 256},
};
#define N_ENGINES ((int)(sizeof(ENGINES) / sizeof(ENGINES[0])))

static int g_gpu_cls, g_gpu_layers; // what got uploaded
static char g_gpu_note[160] = "";

static int engine_ok(int i) {
    if (ENGINES[i].mode == LLM_MODE_SPLIT) return g_gpu_cls;
    if (ENGINES[i].mode == LLM_MODE_GPU) return g_gpu_cls && g_gpu_layers;
    return 1;
}

static void apply_engine(LlmModel *m, int i) {
    m->mode = ENGINES[i].mode;
    m->rescore_k = ENGINES[i].rescore_k;
    gxm_set_mode(ENGINES[i].out_mode, ENGINES[i].headroom);
}

// ----------------------------------------------------------------------------
// video + console

static void init_video(void) {
    VIDEO_Init();
    WPAD_Init();
    rmode = VIDEO_GetPreferredMode(NULL);
    xfb = MEM_K0_TO_K1(SYS_AllocateFramebuffer(rmode));
    console_init(xfb, 20, 20, rmode->fbWidth, rmode->xfbHeight, rmode->fbWidth * VI_DISPLAY_PIX_SZ);
    VIDEO_Configure(rmode);
    VIDEO_SetNextFramebuffer(xfb);
    VIDEO_SetBlack(false);
    VIDEO_Flush();
    VIDEO_WaitVSync();
    if (rmode->viTVMode & VI_NON_INTERLACE) VIDEO_WaitVSync();
    CON_GetMetrics(&con_cols, &con_rows);
}

static void clear_screen(void) { printf("\x1b[2J\x1b[1;0H"); }

// Word-wrapping writer, so stories don't break words at the screen edge.
static int wrap_col = 0;
static char wrap_word[128];
static int wrap_len = 0;

static void wrap_flush_word(void) {
    if (wrap_len == 0) return;
    if (wrap_col + wrap_len >= con_cols - 1 && wrap_col > 0) {
        putchar('\n');
        wrap_col = 0;
    }
    fwrite(wrap_word, 1, wrap_len, stdout);
    wrap_col += wrap_len;
    wrap_len = 0;
}

static void wrap_putc(char ch) {
    if (ch == '\n') {
        wrap_flush_word();
        putchar('\n');
        wrap_col = 0;
    } else if (ch == ' ') {
        wrap_flush_word();
        if (wrap_col > 0 && wrap_col < con_cols - 1) { putchar(' '); wrap_col++; }
    } else {
        if (wrap_len >= (int)sizeof(wrap_word)) wrap_flush_word();
        wrap_word[wrap_len++] = ch;
    }
}

static void wrap_reset(void) { wrap_col = 0; wrap_len = 0; }

// Saves what's on the TV to <root>/llama/screenshot.bmp. The console draws
// straight into the framebuffer, which is YUV 4:2:2 (Y1 Cb Y2 Cr per pixel pair).
static int save_screenshot(const char *root) {
    char path[64];
    snprintf(path, sizeof(path), "%s/llama/screenshot.bmp", root);
    int w = rmode->fbWidth, h = rmode->xfbHeight, row_bytes = w * 3;
    unsigned char *row = (unsigned char *)malloc(row_bytes);
    FILE *f = row ? fopen(path, "wb") : NULL;
    if (!f) { free(row); return -1; }
    unsigned file_size = 54 + (unsigned)row_bytes * h;
    unsigned char hdr[54] = {'B', 'M'};
    unsigned fields[][2] = {{2, file_size}, {10, 54}, {14, 40}, {18, (unsigned)w}, {22, (unsigned)h}, {34, file_size - 54}};
    for (unsigned i = 0; i < sizeof(fields) / sizeof(fields[0]); i++)
        for (int b = 0; b < 4; b++) hdr[fields[i][0] + b] = (unsigned char)(fields[i][1] >> (8 * b));
    hdr[26] = 1;  // planes
    hdr[28] = 24; // bits per pixel
    fwrite(hdr, 1, sizeof(hdr), f);
    for (int y = h - 1; y >= 0; y--) { // BMP rows go bottom-up
        const unsigned char *src = (const unsigned char *)xfb + (size_t)y * w * 2;
        for (int x = 0; x < w; x++) {
            int luma = src[x * 2], cb = src[(x & ~1) * 2 + 1] - 128, cr = src[(x & ~1) * 2 + 3] - 128;
            float yy = 1.164f * (luma - 16);
            int rgb[3] = {(int)(yy + 1.596f * cr), (int)(yy - 0.813f * cr - 0.391f * cb), (int)(yy + 2.018f * cb)};
            for (int c = 0; c < 3; c++) row[x * 3 + 2 - c] = rgb[c] < 0 ? 0 : rgb[c] > 255 ? 255 : rgb[c];
        }
        fwrite(row, 1, row_bytes, f);
    }
    fclose(f);
    free(row);
    return 0;
}

// ----------------------------------------------------------------------------
// callbacks for the inference code

static unsigned long clock_ms(void) { return (unsigned long)ticks_to_millisecs(gettime()); }

static void on_progress(size_t done, size_t total, void *user) {
    (void)user;
    static int last = -1;
    int pct = total ? (int)((done * 100) / total) : 0;
    if (pct == last) return;
    last = pct;
    int bar = pct / 4;
    printf("\r  [");
    for (int i = 0; i < 25; i++) putchar(i < bar ? '#' : '.');
    printf("] %3d%%  (%lu / %lu KB)", pct, (unsigned long)(done / 1024), (unsigned long)(total / 1024));
    fflush(stdout);
}

static int on_token(const char *piece, void *user) {
    (void)user;
    for (const char *c = piece; *c; c++) wrap_putc(*c);
    fflush(stdout);
    // let the player bail out mid-story
    WPAD_ScanPads();
    u32 down = WPAD_ButtonsDown(0);
    if (down & WPAD_BUTTON_HOME) exit(0);
    return (down & WPAD_BUTTON_B) ? 1 : 0;
}

static void print_gpu_stats(void (*out)(const char *, ...)) {
    GxmStats st;
    gxm_get_stats(&st);
    if (st.matmuls == 0) return;
    out("  GPU: %lu matmuls, %lu draws | issuing %lu ms, waiting %lu ms, reading back %lu ms\n",
        (unsigned long)st.matmuls, (unsigned long)st.draws, (unsigned long)ticks_to_millisecs(st.draw_ticks),
        (unsigned long)ticks_to_millisecs(st.wait_ticks), (unsigned long)ticks_to_millisecs(st.reduce_ticks));
}

static void out_screen(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

// Status messages go to the TV and to Dolphin's log (SYS_Report is a no-op on a real Wii).
static void say(const char *fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fputs(buf, stdout);
    SYS_Report("%s", buf);
}

// ----------------------------------------------------------------------------

static void wait_for_home(void) {
    printf("\n\n  Press HOME to return to the Homebrew Channel.\n");
    while (SYS_MainLoop()) {
        WPAD_ScanPads();
        if (WPAD_ButtonsDown(0) & WPAD_BUTTON_HOME) exit(0);
        VIDEO_WaitVSync();
    }
    exit(0);
}

static int try_load(LlmModel *m, LlmTokenizer *t, const char *root) {
    char model_path[64], tok_path[64];
    snprintf(model_path, sizeof(model_path), "%s/llama/model.bin", root);
    snprintf(tok_path, sizeof(tok_path), "%s/llama/tokenizer.bin", root);
    FILE *probe = fopen(model_path, "rb");
    if (!probe) return 1; // not on this device
    fclose(probe);

    say("  Loading %s\n", model_path);
    gxm_reserve(); // first, so the GPU's framebuffer copies land in MEM1
    if (llm_load_model(m, model_path, on_progress, NULL)) {
        say("\n\n  Couldn't load the model: %s\n", llm_error());
        wait_for_home();
    }
    say("\n  Loading tokenizer...\n");
    if (llm_load_tokenizer(t, tok_path, m->c.vocab_size)) {
        say("\n  Couldn't load the tokenizer: %s\n", llm_error());
        wait_for_home();
    }
    return 0;
}

// Upload what fits: the classifier (for split), then the layers (for GPU only).
static void setup_gpu(LlmModel *m) {
    say("  Uploading weights to the GPU...\n");
    size_t bytes = 0;
    if (gxm_init()) {
        snprintf(g_gpu_note, sizeof(g_gpu_note), "GPU off: %s", llm_error());
        return;
    }
    if (gxm_upload_model(m, 0, &bytes) == 0) g_gpu_cls = 1;
    if (g_gpu_cls && gxm_upload_model(m, 1, &bytes) == 0) g_gpu_layers = 1;
    if (g_gpu_layers) snprintf(g_gpu_note, sizeof(g_gpu_note), "GPU textures: %lu KB", (unsigned long)(bytes / 1024));
    else snprintf(g_gpu_note, sizeof(g_gpu_note), "GPU %s: %s", g_gpu_cls ? "classifier only" : "off", llm_error());
}

static long param_count(const LlmModel *m) {
    return (long)m->c.vocab_size * m->c.dim +
           (long)m->c.n_layers * (2L * m->c.dim * m->c.dim + 2L * m->c.dim * (m->c.dim * m->c.n_kv_heads / m->c.n_heads) +
                                  3L * m->c.dim * m->c.hidden_dim);
}

static void draw_menu(const LlmModel *m, int prompt_idx, float temp, int engine) {
    clear_screen();
    printf("  iiama2.c  |  %.1fM params, %s, %d layers  |  %lu KB in use\n", param_count(m) / 1e6,
           m->dtype == LLM_DTYPE_Q8 ? "int8" : "fp32", m->c.n_layers, (unsigned long)(m->bytes / 1024));
    printf("  free: MEM1 %lu KB, MEM2 %lu KB  |  %s\n\n", (unsigned long)(SYS_GetArena1Size() / 1024),
           (unsigned long)(SYS_GetArena2Size() / 1024), g_gpu_note);
    printf("  Prompt  (LEFT/RIGHT):  < %s >\n", PROMPTS[prompt_idx][0] ? PROMPTS[prompt_idx] : "(surprise me)");
    printf("  Temperature (UP/DOWN): %.1f%s\n", temp, temp == 0.0f ? "  (greedy, same story every time)" : "");
    printf("  Engine (+/-):          < %s >\n\n", ENGINES[engine].name);
    printf("  A: tell a story    B: stop    2: screenshot    HOME: exit\n");
}

// ----------------------------------------------------------------------------
// bench mode: every engine, same greedy story, results on screen + SD + Dolphin's log

static FILE *g_bench_file;

static void report(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fputs(buf, stdout);
    SYS_Report("%s", buf);
    if (g_bench_file) { // synced line by line, so a crash or hang still leaves the log on the card
        fputs(buf, g_bench_file);
        fflush(g_bench_file);
        fsync(fileno(g_bench_file));
    }
}

static char g_story[4096];
static int g_story_len;

static int bench_token(const char *piece, void *user) {
    (void)user;
    for (const char *c = piece; *c && g_story_len < (int)sizeof(g_story) - 1; c++) g_story[g_story_len++] = *c;
    g_story[g_story_len] = '\0';
    return 0;
}

// Where each token's time went (LlmModel.prof), averaged like tok/s over the tokens after the first.
// Matmuls are whole calls, on the GPU or the CPU; the rest is the CPU work between them. The last
// line splits attention by position, since the KV cache it reads grows with every token.
static LlmProfile g_prof;

static uint32_t prof_ticks(void) { return gettick(); } // Broadway's time base, TB_TIMER_CLOCK ticks per ms

static void print_profile(unsigned long gen_ms) {
    const LlmProfile *pf = &g_prof;
    if (pf->tokens == 0) return;
    double ms[LLM_PROF_N], sum = 0.0;
    for (int i = 0; i < LLM_PROF_N; i++) {
        ms[i] = (double)pf->ticks[i] / TB_TIMER_CLOCK / pf->tokens;
        sum += ms[i];
    }
#define P(part) ms[LLM_PROF_##part]
    report("  per token (ms): matmuls %.2f (q/k/v %.2f, wo %.2f, w1/w3 %.2f, w2 %.2f, classifier %.2f), "
           "rescore %.2f (top-k %.2f, recompute %.2f), sampling %.2f\n",
           P(QKV) + P(WO) + P(W1W3) + P(W2) + P(CLS), P(QKV), P(WO), P(W1W3), P(W2), P(CLS),
           P(TOPK) + P(RECOMPUTE), P(TOPK), P(RECOMPUTE), P(SAMPLE));
    report("  per token (ms): attention %.2f (kv quant %.2f, q.k %.2f, softmax %.2f, att*v %.2f), rmsnorm %.2f, "
           "rope %.2f (table %.2f), silu %.2f, residual %.2f, embedding %.2f | profiled %.2f of %.2f\n",
           P(KV_QUANT) + P(SCORES) + P(SOFTMAX) + P(MIX), P(KV_QUANT), P(SCORES), P(SOFTMAX), P(MIX), P(RMSNORM),
           P(ROPE) + P(ROPE_TABLE), P(ROPE_TABLE), P(SILU), P(RESIDUAL), P(EMBED), sum, (double)gen_ms / pf->tokens);
#undef P
    char line[256];
    int n = snprintf(line, sizeof(line), "  per token (ms): attention by position");
    for (int q = 0; q < 4; q++) { // quarters of the run, positions lo..hi
        int lo = 1 + q * pf->tokens / 4, hi = (q + 1) * pf->tokens / 4;
        if (hi >= LLM_PROF_MAX_POS) hi = LLM_PROF_MAX_POS - 1;
        if (hi < lo) continue;
        uint64_t t = 0;
        for (int p = lo; p <= hi; p++) t += pf->attention[p];
        n += snprintf(line + n, sizeof(line) - n, "%s %d-%d %.2f", q ? "," : "", lo, hi,
                      (double)t / TB_TIMER_CLOCK / (hi - lo + 1));
    }
    report("%s\n", line);
}

static void run_bench(LlmModel *m, LlmTokenizer *t, const char *root) {
    char path[64];
    int steps = 64, poweroff = 0;
    float temp = 0.0f; // "temp X": after the greedy runs, every engine again sampling at X (top-p 0.9, seed 1)
    snprintf(path, sizeof(path), "%s/llama/bench.txt", root);
    FILE *f = fopen(path, "r");
    if (f) {
        char line[64];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "poweroff", 8) == 0) poweroff = 1;
            if (strncmp(line, "steps ", 6) == 0) steps = atoi(line + 6);
            if (sscanf(line, "temp %f", &temp) == 1 && (temp < 0.0f || temp > 1.5f)) temp = 0.0f;
        }
        fclose(f);
    }
    snprintf(path, sizeof(path), "%s/llama/bench_out.txt", root);
    g_bench_file = fopen(path, "w");

    clear_screen();
    report("iiama2.c bench\n");
    report("model: %.1fM params, dim %d, hidden %d, %d layers, vocab %d, %s group %d\n", param_count(m) / 1e6,
           m->c.dim, m->c.hidden_dim, m->c.n_layers, m->c.vocab_size, m->dtype == LLM_DTYPE_Q8 ? "int8" : "fp32", m->gs);
    report("%s\n", g_gpu_note);

    if (g_gpu_cls) {
        // the same input tools/tev_sim checks against, exactly representable in float
        int n = m->c.dim > m->c.hidden_dim ? m->c.dim : m->c.hidden_dim;
        float *x = (float *)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) x[j] = (float)(((j * 37) % 101) - 50) / 16.0f;
        static const struct { int mode; float headroom; const char *name; } checks[] = {
            {GXM_CLAMP8, 0.5f, "clamp8"}, {GXM_EXACT11, 1.0f, "exact11"}};
        LlmLayer *L = &m->layers[0];
        void *w1 = L->gpu_w13 ? L->gpu_w13 : L->w1.gpu; // stacked, w1's rows come first
        report("selftest blend: cls %08lx\n", (unsigned long)gxm_partials_hash(m->cls.gpu, x, m->c.vocab_size));
        for (int i = 0; i < 2 && g_gpu_layers; i++) {
            gxm_set_mode(checks[i].mode, checks[i].headroom);
            report("selftest %s headroom %.2f: w1 %08lx w2 %08lx\n", checks[i].name, checks[i].headroom,
                   (unsigned long)gxm_partials_hash(w1, x, m->c.hidden_dim),
                   (unsigned long)gxm_partials_hash(L->w2.gpu, x, m->c.dim));
        }
        if (gxm_tokens_failed()) report("warning: GPU draw-sync tokens timed out; matmuls fell back to GX_DrawDone\n");
        free(x);
    }
    report("selftest exp: fn %08lx\n", (unsigned long)llm_check_exp());
    int ps_total, ps_bad = llm_check_paired_singles(m, &ps_total);
    if (ps_total) report("paired singles: %d of %d int8 matmul and attention outputs differ from the scalar loops\n", ps_bad, ps_total);

    LlmSampler s;
    for (int run = 0; run < N_ENGINES * (temp > 0.0f ? 2 : 1); run++) {
        int e = run % N_ENGINES;
        float tp = run < N_ENGINES ? 0.0f : temp;
        if (!engine_ok(e)) continue;
        apply_engine(m, e);
        gxm_reset_stats();
        memset(&g_prof, 0, sizeof(g_prof));
        g_prof.now = prof_ticks;
        m->prof = &g_prof;
        g_story_len = 0;
        g_story[0] = '\0';
        if (llm_init_sampler(&s, m->c.vocab_size, tp, 0.9f, 1)) break;
        unsigned long ms = 0;
        unsigned long t0 = clock_ms();
        int pos = llm_generate(m, t, &s, PROMPTS[0], steps, bench_token, NULL, clock_ms, &ms);
        unsigned long total = clock_ms() - t0;
        llm_free_sampler(&s);
        char name[64];
        snprintf(name, sizeof(name), tp > 0.0f ? "%s, temperature %.1f" : "%s", ENGINES[e].name, tp);
        report("\n[%s] %d tokens, %lu ms total, %.2f tok/s after the first token\n", name, pos, total,
               pos > 1 && ms > 0 ? (pos - 1) * 1000.0 / ms : 0.0);
        print_gpu_stats(report);
        print_profile(ms);
        report("  %s\n", g_story);
    }
    m->prof = NULL;
    report("\ndone\n");
    if (g_bench_file) fclose(g_bench_file);
    g_bench_file = NULL;
    if (poweroff) SYS_ResetSystem(SYS_POWEROFF, 0, 0); // ends a headless Dolphin run
    wait_for_home();
}

// SD:/llama/probe.txt: GPU hardware probes (gx_probe.c), to SD:/llama/probe.bin + probe_out.txt.
static int file_exists(const char *root, const char *name) {
    char path[64];
    snprintf(path, sizeof(path), "%s/llama/%s", root, name);
    FILE *f = fopen(path, "r");
    if (f) fclose(f);
    return f != NULL;
}

static int bench_requested(const char *root) { return file_exists(root, "bench.txt"); }

static void run_probe(LlmModel *m, const char *root) {
    char path[64];
    int poweroff = 0;
    snprintf(path, sizeof(path), "%s/llama/probe.txt", root);
    FILE *f = fopen(path, "r");
    if (f) {
        char line[64];
        while (fgets(line, sizeof(line), f))
            if (strncmp(line, "poweroff", 8) == 0) poweroff = 1;
        fclose(f);
    }
    snprintf(path, sizeof(path), "%s/llama/probe_out.txt", root);
    g_bench_file = fopen(path, "w");
    clear_screen();
    report("iiama2.c GPU probe\n%s\nfree: MEM1 %lu KB, MEM2 %lu KB\n", g_gpu_note,
           (unsigned long)(SYS_GetArena1Size() / 1024), (unsigned long)(SYS_GetArena2Size() / 1024));
    if (g_gpu_cls) {
        snprintf(path, sizeof(path), "%s/llama/probe.bin", root);
        gxp_run(m, path, report);
    }
    dsp_probe_run(m, report); // last, so a hang there can't cost the GPU results
    report("\nprobe done\n");
    if (g_bench_file) fclose(g_bench_file);
    g_bench_file = NULL;
    if (bench_requested(root)) return; // the bench runs next
    if (poweroff) SYS_ResetSystem(SYS_POWEROFF, 0, 0);
    wait_for_home();
}


// autoplay.txt: optional "engine N", "prompt N" and "temp X" lines (indexes into ENGINES and PROMPTS), and
// "screenshot" to save one as soon as the story ends.
static int read_autoplay(const char *root, int *engine, int *prompt_idx, float *temp, int *screenshot) {
    char path[64], line[64];
    snprintf(path, sizeof(path), "%s/llama/autoplay.txt", root);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    while (fgets(line, sizeof(line), f)) {
        int n;
        if (sscanf(line, "engine %d", &n) == 1 && n >= 0 && n < N_ENGINES && engine_ok(n)) *engine = n;
        if (sscanf(line, "prompt %d", &n) == 1 && n >= 0 && n < N_PROMPTS) *prompt_idx = n;
        if (sscanf(line, "temp %f", temp) == 1 && (*temp < 0.0f || *temp > 1.5f)) *temp = 0.8f;
        if (strncmp(line, "screenshot", 10) == 0) *screenshot = 1;
    }
    fclose(f);
    return 1;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    init_video();
    clear_screen();
    printf("\n  iiama2.c: a tiny language model on a 2006 game console\n\n");

    if (!fatInitDefault()) {
        say("  Couldn't mount the SD card (or USB drive).\n");
        wait_for_home();
    }

    LlmModel model;
    LlmTokenizer tok;
    LlmSampler sampler;
    const char *root = "sd:";
    if (try_load(&model, &tok, root)) {
        root = "usb:";
        if (try_load(&model, &tok, root)) {
            say("  No model found. Put model.bin and tokenizer.bin in /llama/ on your SD card.\n");
            wait_for_home();
        }
    }
    setup_gpu(&model);
    if (file_exists(root, "probe.txt")) run_probe(&model, root);
    if (bench_requested(root)) run_bench(&model, &tok, root);

    int prompt_idx = 0, engine = 0;
    float temp = 0.8f;
    int auto_shot = 0;
    int again = read_autoplay(root, &engine, &prompt_idx, &temp, &auto_shot); // also "A: another story" later
    draw_menu(&model, prompt_idx, temp, engine);

    while (SYS_MainLoop()) {
        WPAD_ScanPads();
        u32 down = WPAD_ButtonsDown(0);
        if (down & WPAD_BUTTON_HOME) break;
        if (down & WPAD_BUTTON_2) save_screenshot(root);
        if (down & (WPAD_BUTTON_LEFT | WPAD_BUTTON_RIGHT | WPAD_BUTTON_UP | WPAD_BUTTON_DOWN | WPAD_BUTTON_PLUS |
                    WPAD_BUTTON_MINUS)) {
            if (down & WPAD_BUTTON_LEFT) prompt_idx = (prompt_idx + N_PROMPTS - 1) % N_PROMPTS;
            if (down & WPAD_BUTTON_RIGHT) prompt_idx = (prompt_idx + 1) % N_PROMPTS;
            if (down & WPAD_BUTTON_UP && temp < 1.45f) temp += 0.1f;
            if (down & WPAD_BUTTON_DOWN && temp > 0.05f) temp -= 0.1f;
            if (temp < 0.05f) temp = 0.0f;
            int step = (down & WPAD_BUTTON_PLUS) ? 1 : (down & WPAD_BUTTON_MINUS) ? N_ENGINES - 1 : 0;
            if (step)
                do engine = (engine + step) % N_ENGINES;
                while (!engine_ok(engine));
            draw_menu(&model, prompt_idx, temp, engine);
        }
        if ((down & WPAD_BUTTON_A) || again) {
            again = 0;
            clear_screen();
            printf("\n");
            wrap_reset();
            apply_engine(&model, engine);
            gxm_reset_stats();
            if (llm_init_sampler(&sampler, model.c.vocab_size, temp, 0.9f, (unsigned long long)gettime())) {
                printf("  %s\n", llm_error());
                wait_for_home();
            }
            unsigned long ms = 0;
            int pos = llm_generate(&model, &tok, &sampler, PROMPTS[prompt_idx], model.c.seq_len,
                                   on_token, NULL, clock_ms, &ms);
            wrap_flush_word();
            llm_free_sampler(&sampler);
            if (auto_shot) {
                auto_shot = 0;
                if (save_screenshot(root) == 0) say("screenshot saved after %d tokens\n", pos);
            }
            if (pos < 0) printf("\n\n  Error: %s\n", llm_error());
            else if (pos > 1 && ms > 0)
                printf("\n\n  -- %d tokens, %.2f tok/s, %s --\n", pos, (pos - 1) * 1000.0 / ms, ENGINES[engine].name);
            print_gpu_stats(out_screen);
            printf("  A: another story    1: back to menu    2: screenshot    HOME: exit\n");
            while (SYS_MainLoop()) {
                WPAD_ScanPads();
                u32 d = WPAD_ButtonsDown(0);
                if (d & WPAD_BUTTON_HOME) exit(0);
                if (d & WPAD_BUTTON_1) { draw_menu(&model, prompt_idx, temp, engine); break; }
                if (d & WPAD_BUTTON_2) save_screenshot(root);
                if (d & WPAD_BUTTON_A) { again = 1; break; }
                VIDEO_WaitVSync();
            }
            continue;
        }
        VIDEO_WaitVSync();
    }
    return 0;
}
