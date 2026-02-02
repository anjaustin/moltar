/*
 * debug_gguf — Dump tensor info from a GGUF file
 * Usage: ./debug_gguf <model.gguf>
 */

#include "../include/lfm2_trix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    printf("Loading %s...\n", argv[1]);
    trix_gguf_model_t *model = trix_load_gguf(argv[1]);
    if (!model) {
        fprintf(stderr, "FATAL: failed to load model\n");
        return 1;
    }

    const trix_lfm2_weights_t *w = trix_get_weights(model);

    /* Test embedding */
    float emb[LFM2_D_MODEL];
    uint32_t test_token = 1;  /* BOS-like */
    trix_embed_token(w, test_token, emb);

    printf("\n--- Embedding for token %u ---\n", test_token);
    printf("  first 8: ");
    for (int i = 0; i < 8; i++) printf("%.6f ", emb[i]);
    printf("\n");

    float emb_norm = 0;
    for (int i = 0; i < LFM2_D_MODEL; i++) emb_norm += emb[i] * emb[i];
    emb_norm = sqrtf(emb_norm);
    printf("  L2 norm: %.6f\n", emb_norm);

    /* Test layer 0 shortconv: dump conv kernel values */
    printf("\n--- Layer 0 ShortConv conv kernel ---\n");
    const float *ck = w->shortconv[0].conv_kernel;
    /* If kernel is [L_CACHE, D_MODEL] = [4, 1024], we'd have 4096 floats
     * If kernel is [D_CONV, D_MODEL] = [3, 1024], we'd have 3072 floats */
    printf("  First 16 values:\n  ");
    for (int i = 0; i < 16; i++) printf("%.6f ", ck[i]);
    printf("\n");
    printf("  Values at offset 3072:\n  ");
    for (int i = 3072; i < 3088 && i < 4096; i++) printf("%.6f ", ck[i]);
    printf("\n");

    /* Test forward pass: single token */
    printf("\n--- Forward pass: single token %u ---\n", test_token);
    trix_lfm2_state_t *state = trix_lfm2_init(2048);
    float *x = (float *)malloc(LFM2_D_MODEL * sizeof(float));
    float *logits = (float *)malloc(LFM2_VOCAB_SIZE * sizeof(float));

    trix_embed_token(w, test_token, x);

    printf("  embedding first 4: %.6f %.6f %.6f %.6f\n", x[0], x[1], x[2], x[3]);

    /* Run forward */
    trix_lfm2_forward(state, w, x, logits);

    /* Find top-5 tokens */
    printf("  logits first 4: %.4f %.4f %.4f %.4f\n", logits[0], logits[1], logits[2], logits[3]);

    int top5_idx[5] = {0};
    float top5_val[5] = {-1e30, -1e30, -1e30, -1e30, -1e30};
    for (int i = 0; i < LFM2_VOCAB_SIZE; i++) {
        for (int k = 0; k < 5; k++) {
            if (logits[i] > top5_val[k]) {
                for (int j = 4; j > k; j--) {
                    top5_idx[j] = top5_idx[j-1];
                    top5_val[j] = top5_val[j-1];
                }
                top5_idx[k] = i;
                top5_val[k] = logits[i];
                break;
            }
        }
    }
    printf("  Top 5 logits:\n");
    for (int k = 0; k < 5; k++) {
        printf("    [%d] token=%d  logit=%.4f\n", k, top5_idx[k], top5_val[k]);
    }

    /* Run a second token to see if it changes */
    uint32_t next_token = top5_idx[0];
    printf("\n--- Forward pass: second token %u ---\n", next_token);
    trix_embed_token(w, next_token, x);
    trix_lfm2_forward(state, w, x, logits);

    for (int i = 0; i < LFM2_VOCAB_SIZE; i++) {
        for (int k = 0; k < 5; k++) {
            if (logits[i] > top5_val[k]) {
                for (int j = 4; j > k; j--) {
                    top5_idx[j] = top5_idx[j-1];
                    top5_val[j] = top5_val[j-1];
                }
                top5_idx[k] = i;
                top5_val[k] = logits[i];
                break;
            }
        }
    }
    printf("  Top 5 logits:\n");
    for (int k = 0; k < 5; k++) {
        printf("    [%d] token=%d  logit=%.4f\n", k, top5_idx[k], top5_val[k]);
    }

    free(logits);
    free(x);
    trix_lfm2_free(state);
    trix_unload_gguf(model);
    return 0;
}
