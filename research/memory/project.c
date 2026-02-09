/* ==========================================================================
 * Moltar Memory — Random Projection Implementation
 * ==========================================================================
 * Deterministic random projection from LFM2 hidden states (2048D float)
 * to LCVDB vectors (48D int8).
 *
 * Uses xorshift32 PRNG to generate Rademacher {-1, +1} int8 weights.
 * The projection preserves relative distances (Johnson-Lindenstrauss).
 * ========================================================================== */

#include <math.h>
#include "project.h"

/* xorshift32 PRNG */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t s = *state;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *state = s;
    return s;
}

void moltar_proj_init(moltar_proj_t *proj, int32_t n_embd, uint32_t seed) {
    proj->n_embd = n_embd;

    /* Scale factor: 1/sqrt(n_embd) for unit-variance preservation.
     * Combined with int8 quantization, this keeps projected values
     * in a reasonable range. */
    proj->scale = 1.0f / sqrtf((float)n_embd);

    uint32_t state = seed;

    /* Generate int8 projection weights using Rademacher distribution {-1, +1}.
     * Each bit of the PRNG output determines the sign of one entry.
     * Rademacher is optimal for Johnson-Lindenstrauss: sub-Gaussian with σ²=1.
     *
     * Why not uniform [-64, 63]? Variance is ~1365/entry, causing all output
     * dimensions to converge to the same magnitude via CLT. The "signal"
     * (inter-vector differences) is lost after max-abs quantization.
     * Rademacher with {-1, +1} keeps variance = 1 per entry, so each
     * output dimension is sum(±h[e]) with variance = sum(h[e]²). */
    int total = MOLTAR_PROJ_DIM * n_embd;
    for (int i = 0; i < total; i += 32) {
        uint32_t r = xorshift32(&state);
        int remaining = total - i;
        int batch = remaining < 32 ? remaining : 32;
        for (int b = 0; b < batch; b++) {
            proj->matrix[i + b] = (r & (1u << b)) ? 1 : -1;
        }
    }

    /* Zero-pad remaining columns if n_embd < MOLTAR_MAX_EMBD */
    for (int i = total; i < MOLTAR_PROJ_DIM * MOLTAR_MAX_EMBD; i++) {
        proj->matrix[i] = 0;
    }
}

void moltar_proj_apply(const moltar_proj_t *proj, const float *hidden, int8_t *out) {
    int n_embd = proj->n_embd;
    float scale = proj->scale;
    float projected[MOLTAR_PROJ_DIM];

    /* Matrix multiply: projected[d] = sum_e(matrix[d][e] * hidden[e]) * scale */
    for (int d = 0; d < MOLTAR_PROJ_DIM; d++) {
        const int8_t *row = &proj->matrix[d * n_embd];
        float sum = 0.0f;
        for (int e = 0; e < n_embd; e++) {
            sum += (float)row[e] * hidden[e];
        }
        projected[d] = sum * scale;
    }

    /* Quantize to int8 via max-abs scaling */
    float max_abs = 0.0f;
    for (int d = 0; d < MOLTAR_PROJ_DIM; d++) {
        float a = projected[d] < 0 ? -projected[d] : projected[d];
        if (a > max_abs) max_abs = a;
    }

    if (max_abs < 1e-10f) {
        /* Zero vector — all zeros */
        for (int d = 0; d < MOLTAR_PROJ_DIM; d++)
            out[d] = 0;
        return;
    }

    float inv_scale = 127.0f / max_abs;
    for (int d = 0; d < MOLTAR_PROJ_DIM; d++) {
        float val = projected[d] * inv_scale;
        int32_t ival = (int32_t)(val + (val >= 0 ? 0.5f : -0.5f));
        if (ival > 127) ival = 127;
        if (ival < -128) ival = -128;
        out[d] = (int8_t)ival;
    }
}
