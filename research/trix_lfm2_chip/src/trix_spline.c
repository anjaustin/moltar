/*
 * trix_spline.c — Spline coefficient fitting for frozen activation tables
 *
 * Fits cubic Hermite spline segments to sigmoid and exp.
 * Each segment matches function value AND derivative at both endpoints,
 * giving C1 continuity (no slope discontinuities).
 *
 * Hermite basis on [0,1]:
 *   h00(t) = 2t^3 - 3t^2 + 1      (value at t=0)
 *   h10(t) = t^3 - 2t^2 + t        (slope at t=0)
 *   h01(t) = -2t^3 + 3t^2          (value at t=1)
 *   h11(t) = t^3 - t^2             (slope at t=1)
 *
 *   f(t) = h00*p0 + h10*m0 + h01*p1 + h11*m1
 *
 * Converting to standard form a*t^3 + b*t^2 + c*t + d:
 *   a = 2*p0 - 2*p1 + m0 + m1
 *   b = -3*p0 + 3*p1 - 2*m0 - m1
 *   c = m0
 *   d = p0
 *
 * Created by: Tripp + Claude
 * Date: February 1, 2026
 */

#include "../include/trix_spline.h"
#include <math.h>

/* The global table */
trix_spline_table_t trix_spline_table;

/* ═══════════════════════════════════════════════════════════════════════════
 * Exact reference functions (libm, used only during init)
 * ═══════════════════════════════════════════════════════════════════════════ */

static float sigmoid_exact(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float sigmoid_deriv(float x) {
    float s = sigmoid_exact(x);
    return s * (1.0f - s);
}

static float exp_exact(float x) {
    return expf(x);
}

static float exp_deriv(float x) {
    return expf(x);  /* d/dx exp(x) = exp(x) */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Hermite spline fitting
 *
 * Given function values and derivatives at segment endpoints, compute
 * cubic coefficients in standard form.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void fit_hermite_segment(
    float p0, float m0,  /* value and derivative at t=0 */
    float p1, float m1,  /* value and derivative at t=1 */
    float dx,            /* segment width in x-space (for derivative scaling) */
    trix_spline_seg_t *seg
) {
    /* Scale derivatives: m0, m1 are df/dx, but we need df/dt where t = (x-x0)/dx
     * df/dt = df/dx * dx */
    float sm0 = m0 * dx;
    float sm1 = m1 * dx;

    seg->a = 2.0f * p0 - 2.0f * p1 + sm0 + sm1;
    seg->b = -3.0f * p0 + 3.0f * p1 - 2.0f * sm0 - sm1;
    seg->c = sm0;
    seg->d = p0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Initialize spline tables
 * ═══════════════════════════════════════════════════════════════════════════ */

void trix_spline_init(void) {
    /* ─── Sigmoid: 16 segments over [-8, 8] ─── */
    {
        const int N = TRIX_SPLINE_SIGMOID_NSEGS;
        const float xmin = TRIX_SPLINE_SIGMOID_XMIN;
        const float xmax = TRIX_SPLINE_SIGMOID_XMAX;
        const float dx = (xmax - xmin) / (float)N;

        for (int i = 0; i < N; i++) {
            float x0 = xmin + (float)i * dx;
            float x1 = x0 + dx;

            float p0 = sigmoid_exact(x0);
            float p1 = sigmoid_exact(x1);
            float m0 = sigmoid_deriv(x0);
            float m1 = sigmoid_deriv(x1);

            fit_hermite_segment(p0, m0, p1, m1, dx,
                                &trix_spline_table.sigmoid_segs[i]);
        }
    }

    /* ─── Exp: 32 segments over [-16, 0] ─── */
    {
        const int N = TRIX_SPLINE_EXP_NSEGS;
        const float xmin = TRIX_SPLINE_EXP_XMIN;
        const float xmax = TRIX_SPLINE_EXP_XMAX;
        const float dx = (xmax - xmin) / (float)N;

        for (int i = 0; i < N; i++) {
            float x0 = xmin + (float)i * dx;
            float x1 = x0 + dx;

            float p0 = exp_exact(x0);
            float p1 = exp_exact(x1);
            float m0 = exp_deriv(x0);
            float m1 = exp_deriv(x1);

            fit_hermite_segment(p0, m0, p1, m1, dx,
                                &trix_spline_table.exp_segs[i]);
        }
    }
}
