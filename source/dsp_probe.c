// dsp_probe.c: hardware probes for the Wii's audio DSP, which the app otherwise leaves idle.
//
// Runs at the end of the probe mode (SD:/llama/probe.txt), after gx_probe.c, so a crash or hang here
// can't cost the GPU results. Boots dsp_probe.ds (built by the Makefile with gcdsptool) through libogc's
// DSP task API, then measures what we'd need to know before handing the DSP part of a token's work:
// mailbox round trips, its clock, DMA bandwidth to and from MEM1 and MEM2 (and what DMA traffic does to
// the CPU's and the GPU's), dot-product kernels and the GPU readback's sums, each checked bit for bit
// against the CPU. Everything that only needs MEM1 runs before anything touches MEM2. Every wait on the
// DSP has a timeout; the first one that expires ends the DSP probes. Results go to probe_out.txt as
// text, nothing to probe.bin.
#include "dsp_probe.h"

#include <malloc.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <gccore.h>
#include <ogc/dsp.h>
#include <ogc/lwp_watchdog.h>

#include "dsp_probe_bin.h" // dsp_probe_bin[], dsp_probe_bin_size: the assembled dsp_probe.ds
#include "gx_matmul.h"

// Mails. The CPU reads the DSP's with the top bit (mail pending) set; mail_wait strips it.
#define MAIL_PING 0x50500000u
#define MAIL_PONG 0x50510000u
#define MAIL_CMD 0x0C0D0000u
#define MAIL_HALT 0x48540000u
#define MAIL_UP 0x5ADB0001u
#define MAIL_DONE 0x5D000000u

enum { OP_NOP, OP_SPIN, OP_DMA, OP_DOT16, OP_DOT8, OP_ACC8, OP_RB };

// DSP data memory, in words, as laid out in dsp_probe.ds
#define D_OUTBUF 0x0100
#define D_OUT_WORDS 768
#define D_BUF0 0x0800
#define D_BUF_WORDS 1024

#define T_SHORT 200000 // us: the boot ROM, the microcode's hello, single mails
#define T_CMD 2000000  // us: one command
#define TICKS_PER_US (TB_TIMER_CLOCK / 1000.0)
#define PHYS(p) MEM_VIRTUAL_TO_PHYSICAL(p)

#define TEST_N 288     // columns in the synthetic tests (stories15M's dim)
#define TEST_ROWS 448
#define M1_BYTES (256 * 1024)
#define M2_BYTES (512 * 1024)
#define OUT_BYTES (64 * 1024)

static DspLogFn g_log;
static int g_dead;      // a wait timed out or the DSP said something unexpected: skip the rest
static double g_mhz;    // the DSP's measured clock (0 until known)
static double g_cmd_us; // a do-nothing command's round trip

static u16 g_params[16] ATTRIBUTE_ALIGN(32);
static dsptask_t g_task;
static DSPCallback g_prev_cb;
static u8 *g_ucode;
static u8 *g_m1, *g_m2;         // scratch carved from the tops of MEM1's and MEM2's arenas
static void *g_m1_hi, *g_m2_hi; // the arena tops to put back
static int8_t *g_w8;            // synthetic int8 weights, TEST_ROWS x TEST_N
static int16_t g_xs[1024];      // activations
static u16 *g_xbuf;             // x as the DSP loads it
static u16 *g_out;              // the DSP's results
static u32 g_seed = 12345;

static u32 rnd(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

static u32 us_since(u64 t0) { return (u32)ticks_to_microsecs(gettime() - t0); }

static int die(const char *fmt, ...) {
    char buf[200];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_log("%s", buf);
    g_dead = 1;
    return -1;
}

static int mail_wait(u32 *mail, u32 timeout_us) {
    u64 t0 = gettime();
    while (!DSP_CheckMailFrom())
        if (us_since(t0) > timeout_us) return -1;
    *mail = DSP_ReadMailFrom() & 0x7fffffffu;
    return 0;
}

static int mail_send(u32 mail, u32 timeout_us) {
    u64 t0 = gettime();
    while (DSP_CheckMailTo()) // the DSP hasn't read the previous one
        if (us_since(t0) > timeout_us) return -1;
    DSP_SendMailTo(mail);
    return 0;
}

// One command: op and its parameter block (addresses a, b, c are physical), timed from the first mail
// to the answer. cmd_start and cmd_end split it, so the CPU can work while the DSP does.
static int g_op;
static u64 g_t0;

static int cmd_start(int op, u32 a, u32 b, u32 c, const u16 *n) {
    if (g_dead) return -1;
    g_params[0] = (u16)op;
    g_params[1] = (u16)(a >> 16), g_params[2] = (u16)a;
    g_params[3] = (u16)(b >> 16), g_params[4] = (u16)b;
    g_params[5] = (u16)(c >> 16), g_params[6] = (u16)c;
    for (int i = 0; i < 9; i++) g_params[7 + i] = n ? n[i] : 0;
    DCFlushRange(g_params, sizeof(g_params));
    g_op = op;
    g_t0 = gettime();
    if (mail_send(MAIL_CMD, T_SHORT) || mail_send(PHYS(g_params), T_SHORT))
        return die("dsp: the DSP didn't take command %d\n", op);
    return 0;
}

static int cmd_end(u64 *ticks) {
    u32 mail;
    if (mail_wait(&mail, T_CMD)) return die("dsp: command %d timed out after %d ms\n", g_op, T_CMD / 1000);
    u64 t1 = gettime();
    if (mail != (MAIL_DONE | g_op)) return die("dsp: command %d answered %08lx\n", g_op, (unsigned long)mail);
    if (ticks) *ticks = t1 - g_t0;
    return 0;
}

static int run(int op, u32 a, u32 b, u32 c, const u16 *n, u64 *ticks) {
    return cmd_start(op, a, b, c, n) || cmd_end(ticks);
}

static void no_dsp_irq(void) {} // the microcode never interrupts; libogc's default handler would wait for mail

// libogc's task loader: the DSP's boot ROM DMAs the microcode into instruction RAM and jumps to 0x10.
static int boot(void) {
    u32 len = (dsp_probe_bin_size + 31) & ~31u, mail;
    g_ucode = (u8 *)memalign(32, len);
    if (!g_ucode) return die("dsp: out of memory\n");
    memset(g_ucode, 0, len);
    memcpy(g_ucode, dsp_probe_bin, dsp_probe_bin_size);
    DCFlushRange(g_ucode, len);

    DSP_Init(); // resets the DSP into its boot ROM, which mails 0x8071feed and waits for a task
    g_prev_cb = DSP_RegisterCallback(no_dsp_irq);
    u64 t0 = gettime();
    // DSP_AddTask waits for the ROM's mail without a timeout, so wait here first (without reading it)
    while (!DSP_CheckMailFrom())
        if (us_since(t0) > T_SHORT)
            return die("dsp: no mail from the DSP's boot ROM within %d ms (Dolphin needs DSP LLE for this)\n",
                       T_SHORT / 1000);
    u32 rom_us = us_since(t0);
    u16 dmbh = ((volatile u16 *)0xCC005000)[2]; // the mail's high half; reading the low half would take it

    memset(&g_task, 0, sizeof(g_task));
    g_task.prio = 255;
    g_task.iram_maddr = (void *)PHYS(g_ucode);
    g_task.iram_len = len;
    g_task.iram_addr = 0;
    g_task.init_vec = 0x10;
    t0 = gettime();
    DSP_AddTask(&g_task);
    u32 add_us = us_since(t0);
    if (mail_wait(&mail, T_SHORT)) return die("dsp: the microcode sent nothing within %d ms\n", T_SHORT / 1000);
    u32 up_us = us_since(t0);
    if (mail != MAIL_UP) return die("dsp: the microcode said %08lx instead of hello\n", (unsigned long)mail);
    g_log("dsp: boot ROM mailed %04x.... after %lu us; DSP_AddTask took %lu us, the %lu-byte microcode said "
          "hello after %lu us\n",
          dmbh, (unsigned long)rom_us, (unsigned long)add_us, (unsigned long)len, (unsigned long)up_us);
    return 0;
}

static void probe_mailbox(void) {
    const int n = 1000;
    u64 sum = 0, lo = ~0ull, hi = 0;
    for (int i = 0; i < n; i++) {
        u32 mail;
        u64 t0 = gettime();
        if (mail_send(MAIL_PING | i, T_SHORT) || mail_wait(&mail, T_SHORT)) {
            die("dsp: ping %d timed out\n", i);
            return;
        }
        u64 dt = gettime() - t0;
        if (mail != (MAIL_PONG | i)) {
            die("dsp: ping %d answered %08lx\n", i, (unsigned long)mail);
            return;
        }
        sum += dt;
        if (dt < lo) lo = dt;
        if (dt > hi) hi = dt;
    }
    g_log("dsp: mailbox round trip CPU -> DSP -> CPU, %d pings: %.2f us average, %.2f min, %.2f max\n", n,
          sum / TICKS_PER_US / n, lo / TICKS_PER_US, hi / TICKS_PER_US);
    sum = 0;
    for (int i = 0; i < 100; i++) {
        u64 t;
        if (run(OP_NOP, 0, 0, 0, NULL, &t)) return;
        sum += t;
    }
    g_cmd_us = sum / TICKS_PER_US / 100;
    g_log("dsp: empty command (2 mails, 32-byte parameter DMA, answer): %.2f us average\n", g_cmd_us);
}

// The clock, from how long known numbers of nops take: two run lengths each, so the slope cancels the
// fixed costs. Assumes a nop takes one cycle.
static void probe_clock(void) {
    static const u16 runs[4][9] = {{1000, 10000, 0}, {1000, 20000, 0}, {30000, 0, 1}, {60000, 0, 1}};
    double us[4];
    for (int i = 0; i < 4; i++) {
        u64 t;
        if (run(OP_SPIN, 0, 0, 0, runs[i], &t)) return;
        us[i] = t / TICKS_PER_US;
    }
    double loop_mhz = 1000.0 * 10000 / (us[1] - us[0]), block_mhz = 30000.0 * 32 / (us[3] - us[2]);
    g_log("dsp: clock %.2f MHz from LOOP over 10M and 20M nops (%.2f, %.2f ms; %.1f cycles of overhead per pass)\n",
          loop_mhz, us[0] / 1000, us[1] / 1000, us[0] * loop_mhz / 1000 - 10000);
    g_log("dsp: clock %.2f MHz from BLOOP over 32 nops x 30000 and 60000 (%.2f, %.2f ms; %.2f cycles of overhead "
          "per pass)\n",
          block_mhz, us[2] / 1000, us[3] / 1000, us[2] * block_mhz / 30000 - 32);
    g_mhz = loop_mhz;
}

static int dma(u32 main, int block, int blocks, int dir, int passes, u64 *ticks) {
    u16 n[9] = {D_BUF0, (u16)block, (u16)blocks, (u16)dir, (u16)passes};
    return run(OP_DMA, main, 0, 0, n, ticks);
}

static const char *bank(const void *p) { return (u32)p >= 0x90000000u ? "MEM2" : "MEM1"; }

// 4 KB from src through the DSP's data memory to dst, compared.
static void dma_check(u8 *src, u8 *dst, int salt) {
    if (g_dead) return;
    for (int i = 0; i < 4096; i++) src[i] = (u8)(i * 7 + (i >> 8) + salt);
    memset(dst, 0xa5, 4096);
    DCFlushRange(src, 4096);
    DCFlushRange(dst, 4096);
    if (dma(PHYS(src), 4096, 1, 0, 1, NULL) || dma(PHYS(dst), 4096, 1, 1, 1, NULL)) return;
    DCInvalidateRange(dst, 4096);
    int bad = 0, first = -1;
    for (int i = 0; i < 4096; i++)
        if (dst[i] != src[i] && !bad++) first = i;
    if (bad)
        g_log("dsp: DMA %s -> DSP -> %s: %d of 4096 bytes WRONG, first at %d: %02x, want %02x\n", bank(src), bank(dst),
              bad, first, dst[first], src[first]);
    else
        g_log("dsp: DMA %s -> DSP -> %s: 4096 bytes ok\n", bank(src), bank(dst));
}

// 1 MB each way through 256 KB of buf, one block in flight at a time.
static void dma_rate(u8 *buf) {
    static const int sizes[] = {32, 256, 1024, 4096};
    for (int dir = 0; dir < 2; dir++) {
        for (int s = 0; s < 4; s++) {
            int blocks = 256 * 1024 / sizes[s], passes = 4;
            u64 t;
            DCFlushRange(buf, 256 * 1024);
            if (dma(PHYS(buf), sizes[s], blocks, dir, passes, &t)) return;
            DCInvalidateRange(buf, 256 * 1024);
            double us = t / TICKS_PER_US - g_cmd_us;
            g_log("dsp: DMA %s -> %s, %4d-byte blocks: %6.1f MB/s, %.2f us per block\n", dir ? "DSP" : bank(buf),
                  dir ? bank(buf) : "DSP", sizes[s], 1048576.0 / us, us / (blocks * passes));
        }
    }
}

// Sums 64 KB in order, as gx_probe.c's memory probe does: how fast the CPU streams memory it hasn't cached.
static u32 stream64k(const u32 *w) {
    u32 s = 0;
    for (int i = 0; i < 16384; i += 8)
        s += w[i] + w[i + 1] + w[i + 2] + w[i + 3] + w[i + 4] + w[i + 5] + w[i + 6] + w[i + 7];
    return s;
}

// The CPU streaming cpu (at least 1 MB, not cached) while the DSP DMAs 2 MB from buf in 4 KB blocks:
// each alone, then both at once, the CPU going until the DSP answers.
static void contention(const u8 *cpu, size_t cpu_bytes, const char *what, u8 *buf) {
    if (g_dead || !cpu || cpu_bytes < (1u << 20) + 32) return;
    const u32 *w = (const u32 *)(((u32)cpu + 31) & ~31u);
    size_t chunks = (cpu_bytes - 32) / 65536, k = 0;
    if (chunks > 128) chunks = 128;
    volatile u32 sink = 0;
    u64 t0 = gettime();
    for (int i = 0; i < 64; i++, k = (k + 1) % chunks) sink += stream64k(w + k * 16384);
    double cpu_alone = 64 * 65536.0 / ((gettime() - t0) / TICKS_PER_US);
    u64 t;
    if (dma(PHYS(buf), 4096, 64, 0, 8, &t)) return;
    double dsp_alone = 2097152.0 / (t / TICKS_PER_US - g_cmd_us);
    u16 n[9] = {D_BUF0, 4096, 64, 0, 8};
    if (cmd_start(OP_DMA, PHYS(buf), 0, 0, n)) return;
    u64 c0 = gettime();
    int done = 0;
    while (!DSP_CheckMailFrom() && us_since(c0) < T_CMD) {
        sink += stream64k(w + k * 16384);
        k = (k + 1) % chunks;
        done++;
    }
    u64 c1 = gettime();
    if (cmd_end(&t)) return;
    (void)sink;
    g_log("dsp: CPU streaming %s (%s) alone %.0f MB/s, DSP DMA %s -> DSP alone %.0f MB/s; at the same time: CPU %.0f "
          "MB/s, DSP %.0f MB/s\n",
          bank(cpu), what, cpu_alone, bank(buf), dsp_alone, done * 65536.0 / ((c1 - c0) / TICKS_PER_US),
          2097152.0 / (t / TICKS_PER_US - g_cmd_us));
}

// The GPU's blended output layer (gxm_matmul; its texture fetch is the GPU's limit) alone, then over and
// over while the DSP DMAs 16 MB from buf in 4 KB blocks: does the DSP's traffic slow the GPU down?
static void gpu_contention(LlmModel *m, u8 *buf) {
    if (g_dead || !m->cls.gpu) return;
    float *x = (float *)malloc(m->c.dim * sizeof(float)), *y = (float *)malloc(m->c.vocab_size * sizeof(float));
    if (x && y) {
        for (int j = 0; j < m->c.dim; j++) x[j] = (float)(((j * 37) % 101) - 50) / 16.0f; // the selftest's input
        gxm_matmul(m->cls.gpu, y, x);
        u64 t0 = gettime(), t;
        for (int i = 0; i < 5; i++) gxm_matmul(m->cls.gpu, y, x);
        double alone = (gettime() - t0) / TICKS_PER_US / 5000;
        u16 n[9] = {D_BUF0, 4096, 64, 0, 64};
        int done = 0;
        if (!cmd_start(OP_DMA, PHYS(buf), 0, 0, n)) {
            u64 c0 = gettime();
            while (!DSP_CheckMailFrom() && us_since(c0) < T_CMD) {
                gxm_matmul(m->cls.gpu, y, x);
                done++;
            }
            u64 c1 = gettime();
            if (!cmd_end(&t)) {
                double dsp = 16777216.0 / (t / TICKS_PER_US - g_cmd_us);
                if (done)
                    g_log("dsp: GPU output layer (gxm_matmul) alone %.2f ms; while the DSP DMAs %s -> DSP: %d of them, "
                          "%.2f ms each, and the DSP %.0f MB/s\n",
                          alone, bank(buf), done, (c1 - c0) / TICKS_PER_US / 1000 / done, dsp);
                else
                    g_log("dsp: GPU output layer (gxm_matmul) alone %.2f ms; the DSP DMA'd 16 MB from %s at %.0f MB/s "
                          "before one finished\n",
                          alone, bank(buf), dsp);
            }
        }
    }
    free(x);
    free(y);
}

// rc rows per weight chunk and rpf rows per results flush: chunks are multiples of 32 bytes and fit a
// 1K-word buffer with 32 bytes to spare (for unaligned weights), and each flush is a multiple of 32 bytes
// (4 results) that fits OUTBUF.
static int plan(int wpr, int ng, int *rc, int *rpf) {
    for (int r = (D_BUF_WORDS - 16) / wpr; r >= 1; r--) {
        if ((r * wpr * 2) % 32) continue;
        int f = r;
        while ((f * ng) % 4) f += r;
        if (f * ng * 4 <= D_OUT_WORDS) {
            *rc = r;
            *rpf = f;
            return 0;
        }
    }
    return -1;
}

static long long dot(const int8_t *w, const int16_t *x, int n) {
    long long s = 0;
    for (int j = 0; j < n; j++) s += (long long)w[j] * x[j];
    return s;
}

// a result as the DSP stores it: $acc.h, .m, .l of a 40-bit sum, and a gap word
static long long result40(const u16 *e) {
    unsigned long long v = ((unsigned long long)(e[0] & 0xff) << 32) | ((unsigned long long)e[1] << 16) | e[2];
    return (long long)(v << 24) >> 24;
}

// One dot-product kernel: rows x n weights (reference copy w; the DSP reads src), groups of g columns
// (one 40-bit sum per group). op is OP_DOT16 (src is int16), OP_DOT8 (src is int8 as stored) or OP_ACC8
// (int8, read by the accelerator). resident: DOT16/DOT8 DMA one chunk and compute it over and over.
static void kernel(const char *name, int op, const int8_t *w, const void *src, int rows, int n, int g, int resident) {
    if (g_dead) return;
    int ng = n / g, wpr = op == OP_DOT16 ? n : n / 2, rc = 1, rpf = 1;
    if (op == OP_ACC8) { // no chunks; a flush is a multiple of 4 results, and so are all of them
        for (rpf = D_OUT_WORDS / (4 * ng); rpf > 1 && (rpf * ng) % 4; rpf--) {}
        rows &= ~3;
    } else if (plan(wpr, ng, &rc, &rpf)) {
        g_log("dsp: %s: no chunking fits %d columns\n", name, n);
        return;
    } else {
        rows = rows / rpf * rpf;
    }
    if (((u32)src & 1) && op != OP_ACC8) { // DMA from 32-byte boundaries, compute from a word boundary
        g_log("dsp: %s: skipped, the weights start at an odd address\n", name);
        return;
    }
    if (rows < 4 || n % g || g % 2 || n > 1024 || rows * ng * 8 > OUT_BYTES) {
        g_log("dsp: %s: can't run %d rows x %d in groups of %d\n", name, rows, n, g);
        return;
    }
    // x: straight, except that the packed kernel wants each group's even columns, then its odd ones
    for (int j = 0; j < n; j++) {
        int q = j / g, k = j % g;
        g_xbuf[op == OP_DOT8 ? q * g + (k & 1) * (g / 2) + k / 2 : j] = (u16)g_xs[j];
    }
    int xbytes = (n * 2 + 31) & ~31, out_words = rows * ng * 4;
    DCFlushRange(g_xbuf, xbytes);
    DCFlushRange((void *)src, op == OP_DOT16 ? rows * n * 2 : rows * n);
    memset(g_out, 0, out_words * 2);
    DCFlushRange(g_out, out_words * 2);
    u32 a = PHYS(src), skip = a & 31; // DMA from a 32-byte boundary, compute from skip bytes in
    u16 p[9];
    if (op == OP_ACC8) {
        u32 end = a + rows * n + 64;
        u16 q[9] = {(u16)rows, 0, (u16)(end >> 16), (u16)ng, (u16)end, (u16)(D_OUTBUF + rpf * ng * 4), (u16)xbytes,
                    (u16)(g - 1), 0};
        memcpy(p, q, sizeof(p));
    } else {
        u16 q[9] = {(u16)(rows / rc), (u16)(skip / 2), (u16)rc, (u16)ng, (u16)resident,
                    (u16)(D_OUTBUF + rpf * ng * 4), (u16)xbytes, (u16)(op == OP_DOT16 ? g - 1 : g / 2 - 1),
                    (u16)(rc * wpr * 2)};
        memcpy(p, q, sizeof(p));
        a -= skip;
    }
    u64 best = ~0ull;
    for (int i = 0; i < 3; i++) {
        u64 t;
        if (run(op, a, PHYS(g_xbuf), PHYS(g_out), p, &t)) return;
        if (t < best) best = t;
    }
    DCInvalidateRange(g_out, out_words * 2);

    int bad = 0, br = 0, bq = 0;
    double got0 = 0, want0 = 0; // printed as doubles: exact below 2^53, and newlib may lack %lld
    for (int r = 0; r < rows; r++) {
        for (int q = 0; q < ng; q++) {
            int wr = resident ? r % rc : r;
            long long want = dot(w + (size_t)wr * n + q * g, g_xs + q * g, g) * (op == OP_DOT16 ? 1 : 256);
            long long got = result40(g_out + (r * ng + q) * 4);
            if (got != want && !bad++) br = r, bq = q, got0 = (double)got, want0 = (double)want;
        }
    }
    double us = best / TICKS_PER_US, macs = (double)rows * n;
    char check[96], stream[48] = "";
    if (bad)
        snprintf(check, sizeof(check), "%d of %d sums WRONG (row %d group %d: %.0f, want %.0f)", bad, rows * ng, br,
                 bq, got0, want0);
    else
        snprintf(check, sizeof(check), "all %d sums match the CPU", rows * ng);
    if (!resident) snprintf(stream, sizeof(stream), ", %.1f MB/s of weights", rows * wpr * 2.0 / us);
    g_log("dsp: %s, %d x %d, groups of %d: %.0f us, %.3f rows/us, %.2f cycles/MAC%s; %s\n", name, rows, n, g, us,
          rows / us, g_mhz > 0 ? us * g_mhz / macs : 0.0, stream, check);
}

// The GPU readback's sums: p passes of t framebuffer-copy tiles (RGBA8, 4x4 pixels: 32 AR bytes, then 32
// GB bytes), and for each pixel's R, G and B the sum over passes of (byte - 128) * a 16-bit scale.
static void readback(int p, int t, int reps, int dma_each) {
    if (g_dead) return;
    int bytes = p * t * 64, rows = t * 16 * 3;
    for (int i = 0; i < bytes; i++) g_m1[i] = (u8)rnd();
    for (int i = 0; i < p; i++) g_xbuf[i] = (u16)rnd();
    int sbytes = (p * 2 + 31) & ~31;
    DCFlushRange(g_m1, bytes);
    DCFlushRange(g_xbuf, sbytes);
    memset(g_out, 0, rows * 4);
    DCFlushRange(g_out, rows * 4);
    u16 n[9] = {(u16)p, (u16)t, (u16)reps, (u16)dma_each, (u16)(p - 1), (u16)(t * 32), (u16)sbytes, (u16)(rows * 4),
                (u16)bytes};
    u64 best = ~0ull;
    for (int i = 0; i < 3; i++) {
        u64 tk;
        if (run(OP_RB, PHYS(g_m1), PHYS(g_xbuf), PHYS(g_out), n, &tk)) return;
        if (tk < best) best = tk;
    }
    DCInvalidateRange(g_out, rows * 4);
    int bad = 0;
    for (int r = 0; r < rows; r++) {
        int pix = r / 3, c = r % 3, tile = pix / 16, i = pix % 16;
        int off = tile * 64 + (c == 0 ? 2 * i + 1 : c == 1 ? 32 + 2 * i : 32 + 2 * i + 1);
        long long want = 0;
        for (int q = 0; q < p; q++) want += (long long)(g_m1[q * t * 64 + off] - 128) * (int16_t)g_xbuf[q];
        long long got = (int32_t)((u32)g_out[2 * r] << 16 | g_out[2 * r + 1]);
        if (got != want) bad++;
    }
    double us = best / TICKS_PER_US - g_cmd_us, per = us / reps, rp = rows * p / per;
    g_log("dsp: readback sums, %d passes x %d rows%s: %.1f us each, %.1f rows/us, %.1f row-passes/us, %.2f cycles per "
          "row-pass; %s\n",
          p, rows, dma_each ? ", tiles DMA'd from MEM1 each time" : ", tiles resident", per, rows / per, rp,
          g_mhz > 0 ? g_mhz / rp : 0.0, bad ? "WRONG" : "all match the CPU");
    if (bad) g_log("dsp: readback: %d of %d sums wrong\n", bad, rows);
}

static u8 *carve(int mem2, u32 bytes, void **old_hi) {
    u8 *hi = (u8 *)(mem2 ? SYS_GetArena2Hi() : SYS_GetArena1Hi());
    u8 *lo = (u8 *)(mem2 ? SYS_GetArena2Lo() : SYS_GetArena1Lo());
    u8 *p = (u8 *)(((u32)hi - bytes) & ~31u);
    if (p < lo) return NULL;
    *old_hi = hi;
    if (mem2) SYS_SetArena2Hi(p);
    else SYS_SetArena1Hi(p);
    return p;
}

// Everything that only touches MEM1 runs before anything touches MEM2, in case the DSP can't reach it.
static void sections(LlmModel *m) {
    // synthetic weights and x, with the extremes: row 0 all 127, row 1 all -128, x starting with -32768s
    for (int i = 0; i < TEST_ROWS * TEST_N; i++) g_w8[i] = (int8_t)rnd();
    memset(g_w8, 127, TEST_N);
    memset(g_w8 + TEST_N, -128, TEST_N);
    for (int j = 0; j < 1024; j++) g_xs[j] = j < TEST_N / 4 ? -32768 : j == TEST_N / 4 ? 32767 : (int16_t)rnd();
    int rows16 = M1_BYTES / (TEST_N * 2) < TEST_ROWS ? M1_BYTES / (TEST_N * 2) : TEST_ROWS;
    const int8_t *cls = m->dtype == LLM_DTYPE_Q8 ? m->cls.q : NULL;
    int dim = m->c.dim, real = cls && dim % 32 == 0 && dim <= 1024 && m->c.vocab_size >= TEST_ROWS;
    const u8 *cpu = m->dtype == LLM_DTYPE_Q8 ? (const u8 *)m->cls.q : (const u8 *)m->cls.f;
    size_t cpu_bytes = (size_t)m->c.vocab_size * m->c.dim * (m->dtype == LLM_DTYPE_Q8 ? 1 : 4);
    char what[64];

    g_log("probe: DSP DMA, MEM1\n");
    dma_check(g_m1, g_m1 + 4096, 0);
    if (!g_dead) dma_rate(g_m1);
    contention(cpu, cpu_bytes, "the output layer", g_m1);
    gpu_contention(m, g_m1);

    g_log("probe: DSP dot products, int16 weights (pre-widened): MADDX with a dual load per column\n");
    int16_t *w16 = (int16_t *)g_m1;
    for (int i = 0; i < rows16 * TEST_N; i++) w16[i] = g_w8[i];
    kernel("int16, resident", OP_DOT16, g_w8, w16, rows16, TEST_N, TEST_N, 1);
    kernel("int16, resident", OP_DOT16, g_w8, w16, rows16, TEST_N, 32, 1);
    kernel("int16 from MEM1", OP_DOT16, g_w8, w16, rows16, TEST_N, TEST_N, 0);

    g_log("probe: DSP dot products, int8 weights as stored (two per word, unpacked on the DSP)\n");
    memcpy(g_m1, g_w8, TEST_ROWS * TEST_N);
    kernel("int8, resident", OP_DOT8, g_w8, g_m1, TEST_ROWS, TEST_N, TEST_N, 1);
    kernel("int8, resident", OP_DOT8, g_w8, g_m1, TEST_ROWS, TEST_N, 32, 1);
    kernel("int8 from MEM1", OP_DOT8, g_w8, g_m1, TEST_ROWS, TEST_N, TEST_N, 0);
    if (real) {
        snprintf(what, sizeof(what), "int8 output layer at %08lx", (unsigned long)(u32)cls);
        kernel(what, OP_DOT8, cls, cls, TEST_ROWS, dim, dim, 0);
    }

    g_log("probe: DSP readback sums\n");
    readback(6, 6, 20, 0);
    readback(6, 6, 20, 1);
    readback(16, 4, 20, 0);

    g_log("probe: DSP and MEM2\n");
    dma_check(g_m1, g_m2, 101);
    dma_check(g_m2, g_m1, 202);
    if (!g_dead) dma_rate(g_m2);
    contention(cpu, cpu_bytes, "the output layer", g_m2);
    gpu_contention(m, g_m2);
    int16_t *w16b = (int16_t *)g_m2;
    for (int i = 0; i < rows16 * TEST_N; i++) w16b[i] = g_w8[i];
    kernel("int16 from MEM2", OP_DOT16, g_w8, w16b, rows16, TEST_N, TEST_N, 0);
    memcpy(g_m2, g_w8, TEST_ROWS * TEST_N);
    kernel("int8 from MEM2", OP_DOT8, g_w8, g_m2, TEST_ROWS, TEST_N, TEST_N, 0);

    // last: reading the accelerator from an extended opcode is the least certain thing here
    g_log("probe: DSP dot products, int8 weights through the accelerator (no DMA)\n");
    memcpy(g_m1, g_w8, TEST_ROWS * TEST_N); // the readback sums used it
    kernel("accelerator, MEM1", OP_ACC8, g_w8, g_m1, TEST_ROWS, TEST_N, TEST_N, 0);
    kernel("accelerator, MEM1", OP_ACC8, g_w8, g_m1, TEST_ROWS, TEST_N, 32, 0);
    if (real) {
        snprintf(what, sizeof(what), "accelerator, output layer at %08lx", (unsigned long)(u32)cls);
        kernel(what, OP_ACC8, cls, cls, TEST_ROWS, dim, dim, 0);
    }
    kernel("accelerator, MEM2", OP_ACC8, g_w8, g_m2, TEST_ROWS, TEST_N, TEST_N, 0);
}

void dsp_probe_run(LlmModel *m, DspLogFn log) {
    g_log = log;
    g_dead = 0;
    log("probe: DSP boot\n");
    g_m1 = carve(0, M1_BYTES, &g_m1_hi);
    g_m2 = carve(1, M2_BYTES, &g_m2_hi);
    g_w8 = (int8_t *)malloc(TEST_ROWS * TEST_N);
    g_xbuf = (u16 *)memalign(32, 2048);
    g_out = (u16 *)memalign(32, OUT_BYTES);
    if (!g_m1 || !g_m2 || !g_w8 || !g_xbuf || !g_out) {
        log("dsp: out of memory\n");
    } else if (!boot()) {
        log("dsp: scratch at %08lx (MEM1) and %08lx (MEM2)\n", (unsigned long)(u32)g_m1, (unsigned long)(u32)g_m2);
        log("probe: DSP mailbox\n");
        probe_mailbox();
        log("probe: DSP clock\n");
        if (!g_dead) probe_clock();
        if (!g_dead) sections(m);
        u32 mail;
        if (!g_dead && (mail_send(MAIL_HALT, T_SHORT) || mail_wait(&mail, T_SHORT) || mail != (MAIL_DONE | 0xff)))
            die("dsp: no answer to halt\n");
    }
    if (g_ucode) {
        DSP_Halt();
        DSP_RegisterCallback(g_prev_cb);
    }
    log(g_dead ? "dsp: stopped early; the DSP is halted\n" : "dsp: finished; the DSP is halted\n");
    if (g_m2) SYS_SetArena2Hi(g_m2_hi);
    if (g_m1) SYS_SetArena1Hi(g_m1_hi);
    free(g_w8);
    free(g_xbuf);
    free(g_out);
    free(g_ucode);
    g_m1 = g_m2 = NULL;
    g_w8 = NULL;
    g_xbuf = g_out = NULL;
    g_ucode = NULL;
}
