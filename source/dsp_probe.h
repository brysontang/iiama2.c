// dsp_probe.h: hardware probes for the Wii's audio DSP (see dsp_probe.c).
#ifndef DSP_PROBE_H
#define DSP_PROBE_H

#include "llm.h"

typedef void (*DspLogFn)(const char *fmt, ...);

// Boots the probe microcode on the DSP and logs what it measures, one line at a time. Doesn't need the
// GPU; uses the model's int8 output layer as real weights when there is one. Leaves the DSP halted.
void dsp_probe_run(LlmModel *m, DspLogFn log);

#endif
