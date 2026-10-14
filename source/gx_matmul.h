// gx_matmul.h: weight matmuls on the Wii's GPU (Hollywood's TEV).
#ifndef GX_MATMUL_H
#define GX_MATMUL_H

#include <stdint.h>

#include "llm.h"

// How each pass's 11-bit sum gets out through 8-bit pixels (tools/tev_sim/gxmatmul.py has the details).
#define GXM_CLAMP8 0  // 16 columns per draw; sums outside [-128, 127] saturate. Fast, ~4% error per matmul.
#define GXM_EXACT11 1 // 8 columns, 2 draws; exact 11-bit sums. ~2.4x the GPU work, ~1% error.

typedef struct {
    uint64_t draw_ticks;   // CPU time issuing GX commands
    uint64_t wait_ticks;   // CPU blocked in GX_DrawDone (GPU still busy)
    uint64_t reduce_ticks; // CPU reading the framebuffer copy back and summing passes
    uint32_t matmuls, draws;
} GxmStats;

int gxm_init(void);
void gxm_set_mode(int out_mode, float headroom);

// Uploads one matrix (must be int8 with a group size that's a multiple of 16). Returns a handle for
// LlmMatrix.gpu, or NULL if it's unsupported or memory ran out (llm_error() says which).
void *gxm_upload(const LlmModel *m, const LlmMatrix *w, int rows, int cols);
// Uploads the classifier, then (if `layers`) every layer matrix. Sets m->gpu_matmul.
// Returns 0 if everything asked for was uploaded.
int gxm_upload_model(LlmModel *m, int layers, size_t *bytes);

void gxm_matmul(void *gpu, float *xout, const float *x); // an LlmGpuMatmulFn

// Bit-exactness check against tools/tev_sim: FNV-1a over every pass's integer sum
// (row-major, int16 little-endian) for input x.
uint32_t gxm_partials_hash(void *gpu, const float *x);

void gxm_get_stats(GxmStats *out);
void gxm_reset_stats(void);

#endif
