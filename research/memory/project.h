/* ==========================================================================
 * Moltar Memory — Random Projection from LFM2 Hidden States to LCVDB
 * ==========================================================================
 * Projects n_embd-dimensional float hidden states to 48D int8 vectors
 * suitable for LCVDB insertion.
 *
 * The projection matrix is generated deterministically from a fixed seed
 * (Gaussian i.i.d. entries via xorshift + Box-Muller, then quantized to
 * int8). At query time, the float hidden state is multiplied by the int8
 * matrix, producing an int32 result that is then quantized to int8 via
 * max-abs scaling.
 *
 * Memory: 2048 * 48 = 98,304 bytes (~96 KB) for LFM2-1.2B
 *
 * The projection preserves relative distances (Johnson-Lindenstrauss)
 * with high probability for the coarse semantic matching LCVDB needs.
 * ========================================================================== */

#ifndef MOLTAR_PROJECT_H
#define MOLTAR_PROJECT_H

#include <stdint.h>

/* Maximum input dimension (LFM2-1.2B = 2048, LFM2-350M = 1024) */
#define MOLTAR_MAX_EMBD     2048

/* Output dimension (must match LCVDB_VEC_DIM = 48) */
#define MOLTAR_PROJ_DIM     48

/* Projection matrix: int8[MOLTAR_PROJ_DIM][n_embd]
 * Row-major: proj[d][e] = matrix[d * n_embd + e]
 * Each row is one output dimension.
 */
typedef struct {
    int8_t  matrix[MOLTAR_PROJ_DIM * MOLTAR_MAX_EMBD];  /* projection weights */
    int32_t n_embd;                                       /* actual input dim   */
    float   scale;                                        /* normalization      */
} moltar_proj_t;

/* Initialize projection matrix with deterministic random weights.
 *   n_embd: input dimension (1024 for 350M, 2048 for 1.2B)
 *   seed:   PRNG seed (use 0x4D4F4C54 = "MOLT" for default)
 */
void moltar_proj_init(moltar_proj_t *proj, int32_t n_embd, uint32_t seed);

/* Project a float hidden state to 48D int8.
 *   hidden: float[n_embd] — hidden state from LFM2
 *   out:    int8[48] — output vector for LCVDB
 *
 * Steps:
 *   1. Matrix multiply: int32[48] = proj_matrix[48][n_embd] * hidden[n_embd]
 *      (int8 * float, accumulated in float, then converted)
 *   2. Quantize to int8 via max-abs scaling
 */
void moltar_proj_apply(const moltar_proj_t *proj, const float *hidden, int8_t *out);

#endif /* MOLTAR_PROJECT_H */
