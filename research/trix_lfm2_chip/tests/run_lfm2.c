/*
 * run_lfm2 — Standalone token generator using the TriX LFM2 frozen chip
 *
 * Loads a Q4_0 GGUF, runs greedy token generation, prints timing.
 *
 * Usage: ./trix_lfm2_run <model.gguf> [n_tokens] [prompt_token_id]
 *
 * Default: generate 32 tokens starting from token 1 (BOS-like)
 *
 * Created by: Tripp + Claude
 * Date: February 1, 2026
 */

#include "../include/lfm2_trix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Simple argmax over logits */
static uint32_t argmax(const float *logits, int n) {
    float max_val = logits[0];
    uint32_t max_idx = 0;
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = (uint32_t)i;
        }
    }
    return max_idx;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [n_tokens] [start_token_id]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    int n_gen = argc >= 3 ? atoi(argv[2]) : 32;
    uint32_t start_token = argc >= 4 ? (uint32_t)atoi(argv[3]) : 1;

    printf("═══════════════════════════════════════════════════\n");
    printf("  TriX LFM2 Frozen Chip — Token Generator\n");
    printf("═══════════════════════════════════════════════════\n\n");
    printf("  model:  %s\n", model_path);
    printf("  tokens: %d\n", n_gen);
    printf("  start:  %u\n\n", start_token);

    /* Load model */
    printf("Loading GGUF...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    trix_gguf_model_t *model = trix_load_gguf(model_path);
    if (!model) {
        fprintf(stderr, "FATAL: failed to load model\n");
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double load_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("Model loaded in %.1f ms\n\n", load_ms);

    const trix_lfm2_weights_t *w = trix_get_weights(model);

    /* Initialize state */
    trix_lfm2_state_t *state = trix_lfm2_init(2048);
    if (!state) {
        fprintf(stderr, "FATAL: failed to allocate state\n");
        trix_unload_gguf(model);
        return 1;
    }

    /* Allocate buffers */
    float *x = (float *)malloc(LFM2_D_MODEL * sizeof(float));
    float *logits = (float *)malloc(LFM2_VOCAB_SIZE * sizeof(float));
    if (!x || !logits) {
        fprintf(stderr, "FATAL: malloc failed\n");
        return 1;
    }

    /* Generate tokens */
    printf("Generating tokens:\n");
    printf("  [0] token=%u (start)\n", start_token);

    uint32_t token = start_token;
    double total_gen_ms = 0.0;

    for (int t = 0; t < n_gen; t++) {
        /* Embed current token */
        trix_embed_token(w, token, x);

        /* Forward pass */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        trix_lfm2_forward(state, w, x, logits);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double gen_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        total_gen_ms += gen_ms;

        /* Greedy sample */
        token = argmax(logits, LFM2_VOCAB_SIZE);

        printf("  [%d] token=%u  (%.1f ms)\n", t + 1, token, gen_ms);
    }

    printf("\n═══════════════════════════════════════════════════\n");
    printf("  Generation complete\n");
    printf("  Total: %.1f ms for %d tokens\n", total_gen_ms, n_gen);
    printf("  Average: %.1f ms/token (%.1f tok/s)\n",
           total_gen_ms / n_gen, n_gen * 1000.0 / total_gen_ms);
    printf("═══════════════════════════════════════════════════\n");

    /* Cleanup */
    free(logits);
    free(x);
    trix_lfm2_free(state);
    trix_unload_gguf(model);

    return 0;
}
