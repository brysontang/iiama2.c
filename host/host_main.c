// Desktop build of the same inference code, for testing exported files on your
// PC (or under qemu-ppc) before copying them to the SD card.
//
//   make -C host
//   ./host/iiama2-host sd/llama/model.bin sd/llama/tokenizer.bin -i "Once upon a time"
//
// Flags mirror llama2.c's run: -t temperature, -p top-p, -s seed, -n steps, -i prompt.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../source/llm.h"

static unsigned long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long)ts.tv_sec * 1000ul + (unsigned long)(ts.tv_nsec / 1000000);
}

static int print_piece(const char *piece, void *user) {
    (void)user;
    fputs(piece, stdout);
    fflush(stdout);
    return 0;
}

static void usage(void) {
    fprintf(stderr, "usage: iiama2-host model.bin tokenizer.bin [-t temp] [-p topp] [-s seed] [-n steps] [-i prompt]\n");
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 3) usage();
    const char *model_path = argv[1], *tok_path = argv[2];
    float temperature = 1.0f, topp = 0.9f;
    unsigned long long seed = 0;
    int steps = 256;
    const char *prompt = "";
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) usage();
        switch (argv[i][1]) {
            case 't': temperature = (float)atof(argv[i + 1]); break;
            case 'p': topp = (float)atof(argv[i + 1]); break;
            case 's': seed = strtoull(argv[i + 1], NULL, 10); break;
            case 'n': steps = atoi(argv[i + 1]); break;
            case 'i': prompt = argv[i + 1]; break;
            default: usage();
        }
    }
    if (seed == 0) seed = (unsigned long long)time(NULL);

    LlmModel m;
    LlmTokenizer t;
    LlmSampler s;
    if (llm_load_model(&m, model_path, NULL, NULL)) { fprintf(stderr, "error: %s\n", llm_error()); return 1; }
    if (llm_load_tokenizer(&t, tok_path, m.c.vocab_size)) { fprintf(stderr, "error: %s\n", llm_error()); return 1; }
    if (llm_init_sampler(&s, m.c.vocab_size, temperature, topp, seed)) { fprintf(stderr, "error: %s\n", llm_error()); return 1; }

    unsigned long ms = 0;
    int pos = llm_generate(&m, &t, &s, prompt, steps, print_piece, NULL, now_ms, &ms);
    printf("\n");
    if (pos < 0) { fprintf(stderr, "error: %s\n", llm_error()); return 1; }
    if (pos > 1 && ms > 0) fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)ms * 1000);

    llm_free_sampler(&s);
    llm_free_tokenizer(&t);
    llm_free_model(&m);
    return 0;
}
