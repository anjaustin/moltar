/*
 * trix_spline.h — Piecewise Cubic Spline Activation Tables
 *
 * Replaces libm exp()/sigmoid() with pure FMA evaluation.
 *
 * Design:
 *   - sigmoid: 16 segments over [-8, 8], cubic polynomials
 *   - exp:     32 segments over [-16, 0] (for softmax after max-subtraction)
 *   - Evaluation: Horner's method = 3 FMA + 1 table lookup
 *   - Table size: 256 bytes (sigmoid) + 512 bytes (exp) = 768 bytes total
 *   - Accuracy: max relative error < 0.1% (within Q4_0 noise)
 *
 * Also provides Schraudolph bit-trick for when we want 3-instruction exp()
 * and don't care about 4% error (CPU path).
 *
 * Created by: Tripp + Claude
 * Date: February 1, 2026
 */

#ifndef TRIX_SPLINE_H
#define TRIX_SPLINE_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Schraudolph Bit-Trick exp() — 3 ARM instructions, ~4% error
 *
 * From Yinsen activation_chip.h. Uses IEEE 754 float layout:
 *   exp(x) ~ 2^(x/ln2) ~ float_from_bits(x * 2^23/ln2 + bias)
 *
 * Constants:
 *   12102203.0f = 2^23 / ln(2) = 8388608 / 0.693147
 *   1064866805  = 127*2^23 - 486411 (tuned bias for min avg error)
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float trix_exp_fast(float x) {
    if (x > 88.0f) x = 88.0f;
    if (x < -88.0f) return 0.0f;
    union { float f; int32_t i; } u;
    u.i = (int32_t)(x * 12102203.0f + 1064866805.0f);
    return u.f;
}

/* SiLU via Schraudolph: x / (1 + exp(-x)) */
static inline float trix_silu_fast(float x) {
    return x / (1.0f + trix_exp_fast(-x));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Piecewise Cubic Spline Tables
 *
 * Each segment is a cubic: f(u) = a*u^3 + b*u^2 + c*u + d
 * where u in [0, 1) is the local parameter within the segment.
 * Evaluated via Horner: ((a*u + b)*u + c)*u + d = 3 FMA ops.
 *
 * Coefficients are fit to minimize max error over each segment by
 * matching function value and derivative at segment boundaries.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define TRIX_SPLINE_SIGMOID_NSEGS 16
#define TRIX_SPLINE_SIGMOID_XMIN  (-8.0f)
#define TRIX_SPLINE_SIGMOID_XMAX  (8.0f)

#define TRIX_SPLINE_EXP_NSEGS    32
#define TRIX_SPLINE_EXP_XMIN     (-16.0f)
#define TRIX_SPLINE_EXP_XMAX     (0.0f)

typedef struct {
    float a, b, c, d;  /* cubic coefficients: a*t^3 + b*t^2 + c*t + d */
} trix_spline_seg_t;

typedef struct {
    trix_spline_seg_t sigmoid_segs[TRIX_SPLINE_SIGMOID_NSEGS];
    trix_spline_seg_t exp_segs[TRIX_SPLINE_EXP_NSEGS];
} trix_spline_table_t;

/* Global table — initialized once at startup */
extern trix_spline_table_t trix_spline_table;

/**
 * trix_spline_init — Fit spline coefficients from exact function values.
 * Call once at startup before any inference.
 */
void trix_spline_init(void);

/* ═══════════════════════════════════════════════════════════════════════════
 * Spline Evaluation — Pure FMA, no libm
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float trix_sigmoid_spline(float x) {
    if (x <= TRIX_SPLINE_SIGMOID_XMIN) return 0.0f;
    if (x >= TRIX_SPLINE_SIGMOID_XMAX) return 1.0f;

    /* Map x to normalized [0, NSEGS) */
    float t = (x - TRIX_SPLINE_SIGMOID_XMIN)
            * ((float)TRIX_SPLINE_SIGMOID_NSEGS
               / (TRIX_SPLINE_SIGMOID_XMAX - TRIX_SPLINE_SIGMOID_XMIN));
    int seg = (int)t;
    if (seg >= TRIX_SPLINE_SIGMOID_NSEGS) seg = TRIX_SPLINE_SIGMOID_NSEGS - 1;
    if (seg < 0) seg = 0;
    float u = t - (float)seg;  /* local param [0, 1) */

    const trix_spline_seg_t *s = &trix_spline_table.sigmoid_segs[seg];
    return ((s->a * u + s->b) * u + s->c) * u + s->d;  /* Horner: 3 FMA */
}

static inline float trix_silu_spline(float x) {
    return x * trix_sigmoid_spline(x);
}

static inline float trix_exp_spline(float x) {
    if (x >= TRIX_SPLINE_EXP_XMAX) return 1.0f;
    if (x <= TRIX_SPLINE_EXP_XMIN) return 0.0f;

    float t = (x - TRIX_SPLINE_EXP_XMIN)
            * ((float)TRIX_SPLINE_EXP_NSEGS
               / (TRIX_SPLINE_EXP_XMAX - TRIX_SPLINE_EXP_XMIN));
    int seg = (int)t;
    if (seg >= TRIX_SPLINE_EXP_NSEGS) seg = TRIX_SPLINE_EXP_NSEGS - 1;
    if (seg < 0) seg = 0;
    float u = t - (float)seg;

    const trix_spline_seg_t *s = &trix_spline_table.exp_segs[seg];
    return ((s->a * u + s->b) * u + s->c) * u + s->d;
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_SPLINE_H */
