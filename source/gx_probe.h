// gx_probe.h: hardware probes for the GPU path (see gx_probe.c).
#ifndef GX_PROBE_H
#define GX_PROBE_H

#include "llm.h"

typedef void (*GxpLogFn)(const char *fmt, ...);

// Runs every probe and writes the raw framebuffer copies to bin_path. Needs gxm_init() and the
// model's matrices uploaded. Returns 0 on success.
int gxp_run(LlmModel *m, const char *bin_path, GxpLogFn log);

#endif
