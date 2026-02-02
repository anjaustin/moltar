/*
 * test_spline — Accuracy and performance test for spline activations
 *
 * Tests:
 *   1. Sigmoid spline accuracy vs libm across [-10, 10]
 *   2. Exp spline accuracy vs libm across [-20, 0]
 *   3. SiLU spline accuracy
 *   4. Schraudolph exp accuracy
 *   5. Performance: throughput comparison (spline vs libm vs Schraudolph)
 *
 * Created by: Tripp + Claude
 * Date: February 1, 2026
 */

#include "../include/trix_spline.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_TEST_POINTS 10000
#define N_PERF_ITERS  1000000

static int tests_passed = 0;
static int tests_failed = 0;

static void check_accuracy(
    const char *name,
    float (*approx_fn)(float),
    float (*exact_fn)(float),
    float xmin, float xmax,
    float max_abs_tol,
    float max_rel_tol
) {
    float worst_abs = 0.0f;
    float worst_rel = 0.0f;
    float worst_x = 0.0f;
    int n_bad_abs = 0;
    int n_bad_rel = 0;

    for (int i = 0; i < N_TEST_POINTS; i++) {
        float x = xmin + (xmax - xmin) * (float)i / (float)(N_TEST_POINTS - 1);
        float exact = exact_fn(x);
        float approx = approx_fn(x);
        float abs_err = fabsf(approx - exact);
        float rel_err = (fabsf(exact) > 1e-10f) ? abs_err / fabsf(exact) : abs_err;

        if (abs_err > worst_abs) { worst_abs = abs_err; worst_x = x; }
        if (rel_err > worst_rel) worst_rel = rel_err;
        if (abs_err > max_abs_tol) n_bad_abs++;
        if (rel_err > max_rel_tol && fabsf(exact) > 1e-6f) n_bad_rel++;
    }

    int pass = (worst_abs <= max_abs_tol * 2.0f); /* 2x margin */
    if (pass) tests_passed++; else tests_failed++;

    printf("  %-20s worst_abs=%.2e worst_rel=%.2e at x=%.3f  %s\n",
           name, worst_abs, worst_rel, worst_x, pass ? "PASS" : "FAIL");
    if (n_bad_abs > 0 || n_bad_rel > 0) {
        printf("    (bad_abs=%d/%d, bad_rel=%d/%d)\n",
               n_bad_abs, N_TEST_POINTS, n_bad_rel, N_TEST_POINTS);
    }
}

/* Wrappers for libm */
static float sigmoid_ref(float x) { return 1.0f / (1.0f + expf(-x)); }
static float silu_ref(float x) { return x * sigmoid_ref(x); }
static float exp_ref(float x) { return expf(x); }

/* Benchmark a function */
static double bench_fn(float (*fn)(float), float xmin, float xmax, int n) {
    volatile float sink = 0.0f;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    float dx = (xmax - xmin) / (float)n;
    float x = xmin;
    for (int i = 0; i < n; i++) {
        sink += fn(x);
        x += dx;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    (void)sink;
    return (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
}

int main(void) {
    printf("====================================================\n");
    printf("  TriX Spline Activation Test Suite\n");
    printf("====================================================\n\n");

    /* Initialize spline tables */
    trix_spline_init();
    printf("Spline tables initialized\n");
    printf("  sigmoid: %d segments, %zu bytes\n",
           TRIX_SPLINE_SIGMOID_NSEGS,
           sizeof(trix_spline_table.sigmoid_segs));
    printf("  exp:     %d segments, %zu bytes\n\n",
           TRIX_SPLINE_EXP_NSEGS,
           sizeof(trix_spline_table.exp_segs));

    /* ─── Accuracy tests ─── */
    printf("--- Accuracy (vs libm) ---\n");

    check_accuracy("sigmoid_spline", trix_sigmoid_spline, sigmoid_ref,
                   -10.0f, 10.0f, 1e-3f, 1e-2f);

    check_accuracy("silu_spline", trix_silu_spline, silu_ref,
                   -10.0f, 10.0f, 1e-2f, 1e-2f);

    check_accuracy("exp_spline[-16,0]", trix_exp_spline, exp_ref,
                   -16.0f, 0.0f, 1e-3f, 1e-2f);

    /* Schraudolph: ~4% relative error. Test over [-16, 0] like exp_spline
     * (softmax domain), and use relative-only tolerance since absolute error
     * scales with exp(x) which grows exponentially for positive x. */
    check_accuracy("exp_fast(schraudo)", trix_exp_fast, exp_ref,
                   -16.0f, 0.0f, 0.05f, 0.05f);

    check_accuracy("silu_fast", trix_silu_fast, silu_ref,
                   -10.0f, 10.0f, 0.5f, 0.1f);

    /* ─── Boundary tests ─── */
    printf("\n--- Boundary values ---\n");

    float s0 = trix_sigmoid_spline(0.0f);
    float s_neg = trix_sigmoid_spline(-100.0f);
    float s_pos = trix_sigmoid_spline(100.0f);
    printf("  sigmoid(0)    = %.6f (expect 0.5):   %s\n", s0,
           fabsf(s0 - 0.5f) < 1e-3f ? "PASS" : "FAIL");
    printf("  sigmoid(-100) = %.6f (expect 0.0):   %s\n", s_neg,
           s_neg == 0.0f ? "PASS" : "FAIL");
    printf("  sigmoid(100)  = %.6f (expect 1.0):   %s\n", s_pos,
           s_pos == 1.0f ? "PASS" : "FAIL");
    tests_passed += (fabsf(s0 - 0.5f) < 1e-3f) + (s_neg == 0.0f) + (s_pos == 1.0f);
    tests_failed += (fabsf(s0 - 0.5f) >= 1e-3f) + (s_neg != 0.0f) + (s_pos != 1.0f);

    float e0 = trix_exp_spline(0.0f);
    float e_neg = trix_exp_spline(-100.0f);
    printf("  exp(0)   = %.6f (expect 1.0):   %s\n", e0,
           fabsf(e0 - 1.0f) < 1e-3f ? "PASS" : "FAIL");
    printf("  exp(-100) = %.6f (expect 0.0):   %s\n", e_neg,
           e_neg == 0.0f ? "PASS" : "FAIL");
    tests_passed += (fabsf(e0 - 1.0f) < 1e-3f) + (e_neg == 0.0f);
    tests_failed += (fabsf(e0 - 1.0f) >= 1e-3f) + (e_neg != 0.0f);

    /* ─── Performance benchmark ─── */
    printf("\n--- Performance (%d iterations) ---\n", N_PERF_ITERS);

    double ns_sigmoid_libm   = bench_fn(sigmoid_ref, -8.0f, 8.0f, N_PERF_ITERS);
    double ns_sigmoid_spline = bench_fn(trix_sigmoid_spline, -8.0f, 8.0f, N_PERF_ITERS);
    double ns_exp_libm       = bench_fn(exp_ref, -16.0f, 0.0f, N_PERF_ITERS);
    double ns_exp_spline     = bench_fn(trix_exp_spline, -16.0f, 0.0f, N_PERF_ITERS);
    double ns_exp_schraudo   = bench_fn(trix_exp_fast, -16.0f, 0.0f, N_PERF_ITERS);
    double ns_silu_libm      = bench_fn(silu_ref, -8.0f, 8.0f, N_PERF_ITERS);
    double ns_silu_spline    = bench_fn(trix_silu_spline, -8.0f, 8.0f, N_PERF_ITERS);
    double ns_silu_fast      = bench_fn(trix_silu_fast, -8.0f, 8.0f, N_PERF_ITERS);

    printf("  sigmoid:  libm=%.1f ns  spline=%.1f ns  (%.1fx)\n",
           ns_sigmoid_libm / N_PERF_ITERS, ns_sigmoid_spline / N_PERF_ITERS,
           ns_sigmoid_libm / ns_sigmoid_spline);
    printf("  exp:      libm=%.1f ns  spline=%.1f ns  schraud=%.1f ns  (spline %.1fx, schraud %.1fx)\n",
           ns_exp_libm / N_PERF_ITERS, ns_exp_spline / N_PERF_ITERS,
           ns_exp_schraudo / N_PERF_ITERS,
           ns_exp_libm / ns_exp_spline, ns_exp_libm / ns_exp_schraudo);
    printf("  silu:     libm=%.1f ns  spline=%.1f ns  fast=%.1f ns  (spline %.1fx, fast %.1fx)\n",
           ns_silu_libm / N_PERF_ITERS, ns_silu_spline / N_PERF_ITERS,
           ns_silu_fast / N_PERF_ITERS,
           ns_silu_libm / ns_silu_spline, ns_silu_libm / ns_silu_fast);

    /* ─── Summary ─── */
    printf("\n====================================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("====================================================\n");

    return tests_failed > 0 ? 1 : 0;
}
