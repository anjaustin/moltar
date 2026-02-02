/*
 * test_trix_lfm2 — Unit tests for the TriX LFM2 frozen chip
 *
 * Tests each sub-chip independently with known inputs, then tests
 * the full forward pass against expected outputs.
 *
 * Two modes:
 *   1. Standalone: test math primitives without any model file
 *   2. With GGUF:  load model, run forward pass, print logits
 *      Usage: ./test_trix_lfm2 [path-to-gguf]
 *
 * Created by: Tripp + Claude
 * Date: February 1, 2026
 */

#include "../include/lfm2_trix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Test infrastructure
 * ═══════════════════════════════════════════════════════════════════════════ */

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_CLOSE(name, got, expected, tol) do {                     \
    float _g = (got), _e = (expected), _t = (tol);                     \
    if (fabsf(_g - _e) > _t) {                                         \
        fprintf(stderr, "FAIL: %s: got %f, expected %f (tol %f)\n",    \
                name, _g, _e, _t);                                     \
        tests_failed++;                                                 \
    } else {                                                            \
        tests_passed++;                                                 \
    }                                                                   \
} while(0)

#define ASSERT_TRUE(name, cond) do {                                    \
    if (!(cond)) {                                                      \
        fprintf(stderr, "FAIL: %s\n", name);                            \
        tests_failed++;                                                 \
    } else {                                                            \
        tests_passed++;                                                 \
    }                                                                   \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: RMSNorm
 *
 * Known values:
 *   x = [1, 2, 3, 4], gamma = [1, 1, 1, 1]
 *   mean(x^2) = (1+4+9+16)/4 = 7.5
 *   rms = sqrt(7.5 + 1e-5) = 2.73861...
 *   out = x / rms = [0.36515, 0.73030, 1.09545, 1.46060]
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_rmsnorm(void) {
    printf("test_rmsnorm...\n");

    float x[]     = {1.0f, 2.0f, 3.0f, 4.0f};
    float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];

    trix_rmsnorm(x, gamma, out, 4);

    float rms = sqrtf(7.5f + 1e-5f);
    ASSERT_CLOSE("rmsnorm[0]", out[0], 1.0f / rms, 1e-4f);
    ASSERT_CLOSE("rmsnorm[1]", out[1], 2.0f / rms, 1e-4f);
    ASSERT_CLOSE("rmsnorm[2]", out[2], 3.0f / rms, 1e-4f);
    ASSERT_CLOSE("rmsnorm[3]", out[3], 4.0f / rms, 1e-4f);

    /* With non-unit gamma */
    float gamma2[] = {2.0f, 0.5f, 1.0f, 3.0f};
    trix_rmsnorm(x, gamma2, out, 4);
    ASSERT_CLOSE("rmsnorm_gamma[0]", out[0], 2.0f * 1.0f / rms, 1e-4f);
    ASSERT_CLOSE("rmsnorm_gamma[3]", out[3], 3.0f * 4.0f / rms, 1e-4f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: RoPE
 *
 * At pos=0, angle=0 for all pairs, so cos=1, sin=0 → identity.
 * At pos=1, first pair gets angle = 1.0 / (1000000 ^ (0/64)) = 1.0
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_rope(void) {
    printf("test_rope...\n");

    /* pos=0 should be identity */
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float orig[4];
    memcpy(orig, x, sizeof(x));

    trix_rope(x, 4, 0, LFM2_ROPE_BASE);
    ASSERT_CLOSE("rope_pos0[0]", x[0], orig[0], 1e-6f);
    ASSERT_CLOSE("rope_pos0[1]", x[1], orig[1], 1e-6f);
    ASSERT_CLOSE("rope_pos0[2]", x[2], orig[2], 1e-6f);
    ASSERT_CLOSE("rope_pos0[3]", x[3], orig[3], 1e-6f);

    /* pos=1: verify rotation for first pair */
    float y[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    trix_rope(y, 4, 1, LFM2_ROPE_BASE);

    float freq0 = 1.0f / powf(LFM2_ROPE_BASE, 0.0f / 4.0f); /* = 1.0 */
    float angle0 = 1.0f * freq0;
    ASSERT_CLOSE("rope_pos1[0]", y[0], cosf(angle0), 1e-5f);
    ASSERT_CLOSE("rope_pos1[1]", y[1], sinf(angle0), 1e-5f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Q4_0 Matvec
 *
 * Create a known Q4_0 block and verify dequant + dot product.
 * A block with scale=1.0 (FP16) and all nibbles = 8 should give
 * all-zero dequantized values (since val - 8 = 0).
 * A block with scale=1.0 and nibbles = 9 gives val = +1.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_matvec_q4_0(void) {
    printf("test_matvec_q4_0...\n");

    /* Construct a 1x32 matrix (1 row, 1 Q4_0 block) */
    q4_0_block_t block;

    /* FP16 for 1.0: sign=0, expo=15 (biased), mant=0
     * FP16: 0 01111 0000000000 = 0x3C00 */
    block.scale_f16 = 0x3C00;

    /* All nibbles = 9: dequant = (9 - 8) * 1.0 = 1.0 for all 32 values */
    memset(block.qs, 0x99, 16);

    float x[32];
    for (int i = 0; i < 32; i++) x[i] = 1.0f;

    float y[1] = {0.0f};
    trix_matvec_q4_0(&block, x, y, 1, 32);

    /* Each of 32 values is 1.0, dot with all-ones x = 32.0 */
    ASSERT_CLOSE("matvec_ones", y[0], 32.0f, 0.01f);

    /* All nibbles = 8: dequant = 0 → dot product = 0 */
    memset(block.qs, 0x88, 16);
    y[0] = 999.0f;
    trix_matvec_q4_0(&block, x, y, 1, 32);
    ASSERT_CLOSE("matvec_zeros", y[0], 0.0f, 0.01f);

    /* Mixed: nibbles = 0x0A → low=10-8=2, high=0-8=-8 */
    memset(block.qs, 0x0A, 16);
    y[0] = 0.0f;
    trix_matvec_q4_0(&block, x, y, 1, 32);
    /* 16 pairs of (2, -8) dotted with 1.0 each = 16*(2 + (-8)) = 16*(-6) = -96 */
    ASSERT_CLOSE("matvec_mixed", y[0], -96.0f, 0.01f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: State init/free
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_state_lifecycle(void) {
    printf("test_state_lifecycle...\n");

    trix_lfm2_state_t *state = trix_lfm2_init(2048);
    ASSERT_TRUE("state_alloc", state != NULL);
    ASSERT_TRUE("state_pos_zero", state->pos == 0);
    ASSERT_TRUE("state_max_ctx", state->max_ctx == 2048);
    ASSERT_TRUE("state_k_cache", state->k_cache != NULL);
    ASSERT_TRUE("state_v_cache", state->v_cache != NULL);

    /* conv_state should be zero-initialized */
    float sum = 0.0f;
    for (int l = 0; l < 10; l++)
        for (int t = 0; t < LFM2_D_CONV; t++)
            for (int d = 0; d < LFM2_D_MODEL; d++)
                sum += state->conv_state[l][t][d];
    ASSERT_CLOSE("conv_state_zeroed", sum, 0.0f, 0.0f);

    trix_lfm2_free(state);
    tests_passed++;
    printf("  (free did not crash)\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test: Full forward pass with GGUF model (if provided)
 *
 * Loads the model, embeds token 1 ("a" or similar), runs one forward pass,
 * and prints the top-5 logits. This verifies the entire pipeline works
 * end-to-end without crashing.
 *
 * Correctness validation requires comparing against llama.cpp output
 * for the same token at the same position, which is done separately.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_forward_with_gguf(const char *gguf_path) {
    printf("test_forward_with_gguf('%s')...\n", gguf_path);

    trix_gguf_model_t *model = trix_load_gguf(gguf_path);
    if (!model) {
        fprintf(stderr, "FAIL: could not load GGUF\n");
        tests_failed++;
        return;
    }
    tests_passed++;
    printf("  GGUF loaded\n");

    const trix_lfm2_weights_t *w = trix_get_weights(model);

    /* Initialize state */
    trix_lfm2_state_t *state = trix_lfm2_init(2048);
    ASSERT_TRUE("state_for_forward", state != NULL);

    /* Embed token 1 */
    float x[LFM2_D_MODEL];
    trix_embed_token(w, 1, x);

    /* Verify embedding is non-zero */
    float emb_sum = 0.0f;
    for (int i = 0; i < LFM2_D_MODEL; i++) emb_sum += fabsf(x[i]);
    ASSERT_TRUE("embedding_nonzero", emb_sum > 0.1f);
    printf("  embedding L1 norm = %.4f\n", emb_sum);

    /* Allocate logits */
    float *logits = (float *)malloc(LFM2_VOCAB_SIZE * sizeof(float));
    ASSERT_TRUE("logits_alloc", logits != NULL);

    /* Run forward pass */
    printf("  running forward pass (16 layers)...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    trix_lfm2_forward(state, w, x, logits);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("  forward pass: %.1f ms\n", elapsed_ms);

    /* State should have advanced */
    ASSERT_TRUE("pos_advanced", state->pos == 1);

    /* Find top-5 logits */
    printf("  top-5 logits:\n");
    for (int rank = 0; rank < 5; rank++) {
        float max_val = -1e30f;
        int max_idx = -1;
        for (int i = 0; i < LFM2_VOCAB_SIZE; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        printf("    [%d] token %d = %.4f\n", rank, max_idx, max_val);
        logits[max_idx] = -1e30f; /* mask out for next rank */
    }

    /* Verify logits are not all zero or NaN */
    float logit_sum = 0.0f;
    int nan_count = 0;
    for (int i = 0; i < LFM2_VOCAB_SIZE; i++) {
        if (isnan(logits[i])) nan_count++;
        logit_sum += fabsf(logits[i]);
    }
    ASSERT_TRUE("logits_not_nan", nan_count == 0);
    ASSERT_TRUE("logits_not_zero", logit_sum > 0.0f);

    free(logits);
    trix_lfm2_free(state);
    trix_unload_gguf(model);

    printf("  forward pass test complete\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    printf("═══════════════════════════════════════════════════\n");
    printf("  TriX LFM2 Frozen Chip — Test Suite\n");
    printf("═══════════════════════════════════════════════════\n\n");

    /* Standalone math tests (no model needed) */
    test_rmsnorm();
    test_rope();
    test_matvec_q4_0();
    test_state_lifecycle();

    /* Full forward pass test (requires GGUF) */
    if (argc >= 2) {
        printf("\n─── GGUF Model Tests ───\n");
        test_forward_with_gguf(argv[1]);
    } else {
        printf("\n(skipping GGUF tests — pass path as argument)\n");
    }

    /* Summary */
    printf("\n═══════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
