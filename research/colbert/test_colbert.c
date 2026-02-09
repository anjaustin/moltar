/*
 * test_colbert.c — ColBERT engine tests
 *
 * Tests: dot product, MaxSim, quantization, add/search, top-k ranking.
 * Run on device: taskset c0 /data/local/tmp/test_colbert
 */

#include "colbert.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* ── Helpers ── */

static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void fill_random_i8(int8_t *buf, int n, uint32_t *state)
{
    for (int i = 0; i < n; i++) {
        buf[i] = (int8_t)(xorshift32(state) % 256 - 128);
    }
}

static void fill_random_f32(float *buf, int n, uint32_t *state)
{
    for (int i = 0; i < n; i++) {
        buf[i] = ((float)(xorshift32(state) % 20000) - 10000.0f) / 10000.0f;
    }
}

/* Reference dot product (C, no NEON) */
static int32_t ref_dot_i8(const int8_t *a, const int8_t *b, int dim)
{
    int32_t sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

/* Reference MaxSim (C, no NEON) */
static int32_t ref_maxsim(const int8_t *query, const int8_t *doc,
                           int n_q, int n_d)
{
    int32_t total = 0;
    for (int i = 0; i < n_q; i++) {
        int32_t max_dot = -2147483647 - 1;
        for (int j = 0; j < n_d; j++) {
            int32_t d = ref_dot_i8(&query[i * COLBERT_EMB_DIM],
                                    &doc[j * COLBERT_EMB_DIM],
                                    COLBERT_EMB_DIM);
            if (d > max_dot) max_dot = d;
        }
        total += max_dot;
    }
    return total;
}

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-40s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

/* ── Tests ── */

static void test_dot_product(void)
{
    TEST("dot product correctness");

    uint32_t state = 0xDEADBEEF;
    int8_t a[COLBERT_EMB_DIM], b[COLBERT_EMB_DIM];
    int errors = 0;

    for (int trial = 0; trial < 1000; trial++) {
        fill_random_i8(a, COLBERT_EMB_DIM, &state);
        fill_random_i8(b, COLBERT_EMB_DIM, &state);

        int32_t neon_result = colbert_dot_i8(a, b);
        int32_t ref_result  = ref_dot_i8(a, b, COLBERT_EMB_DIM);

        if (neon_result != ref_result) {
            if (errors == 0) {
                printf("FAIL: trial %d: neon=%d ref=%d\n", trial, neon_result, ref_result);
            }
            errors++;
        }
    }

    if (errors == 0) PASS();
    else { char msg[64]; snprintf(msg, sizeof(msg), "%d/1000 mismatches", errors); FAIL(msg); }
}

static void test_maxsim(void)
{
    TEST("MaxSim correctness");

    uint32_t state = 0xCAFEBABE;
    int errors = 0;

    /* Test various n_q, n_d combinations */
    int configs[][2] = {{1,1}, {1,10}, {4,8}, {8,16}, {32,64}, {32,256}};
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int c = 0; c < n_configs; c++) {
        int n_q = configs[c][0];
        int n_d = configs[c][1];

        int8_t *query = (int8_t *)malloc(n_q * COLBERT_EMB_DIM);
        int8_t *doc   = (int8_t *)malloc(n_d * COLBERT_EMB_DIM);
        fill_random_i8(query, n_q * COLBERT_EMB_DIM, &state);
        fill_random_i8(doc,   n_d * COLBERT_EMB_DIM, &state);

        int32_t neon_result = colbert_maxsim_i8(query, doc, n_q, n_d);
        int32_t ref_result  = ref_maxsim(query, doc, n_q, n_d);

        if (neon_result != ref_result) {
            if (errors == 0) {
                printf("FAIL: n_q=%d n_d=%d: neon=%d ref=%d\n",
                       n_q, n_d, neon_result, ref_result);
            }
            errors++;
        }

        free(query);
        free(doc);
    }

    if (errors == 0) PASS();
    else { char msg[64]; snprintf(msg, sizeof(msg), "%d/%d mismatches", errors, n_configs); FAIL(msg); }
}

static void test_quantize(void)
{
    TEST("quantize f32 -> i8");

    float f[COLBERT_EMB_DIM];
    int8_t q[COLBERT_EMB_DIM];

    /* Test: all zeros */
    memset(f, 0, sizeof(f));
    colbert_quantize_f32_to_i8(f, q, COLBERT_EMB_DIM);
    int all_zero = 1;
    for (int i = 0; i < COLBERT_EMB_DIM; i++) {
        if (q[i] != 0) { all_zero = 0; break; }
    }
    if (!all_zero) { FAIL("zero vector not preserved"); return; }

    /* Test: max element should map to ±127 */
    memset(f, 0, sizeof(f));
    f[0] = 1.0f;
    f[1] = -0.5f;
    colbert_quantize_f32_to_i8(f, q, COLBERT_EMB_DIM);
    if (q[0] != 127) { FAIL("max not 127"); return; }
    if (q[1] != -64 && q[1] != -63) { FAIL("half not ~-64"); return; }

    PASS();
}

static void test_add_and_search(void)
{
    TEST("add docs + search");

    /* Allocate buffers */
    int8_t  *tok_buf  = (int8_t *)calloc(1024 * COLBERT_EMB_DIM, 1);
    colbert_doc_t *doc_buf = (colbert_doc_t *)calloc(64, sizeof(colbert_doc_t));
    char    *text_buf = (char *)calloc(16 * 1024, 1);

    colbert_t db;
    colbert_init(&db, tok_buf, 1024, doc_buf, 64, text_buf, 16 * 1024);

    /* Create 3 documents with distinct embeddings */
    uint32_t state = 0x12345678;
    const char *texts[] = {
        "Machine learning algorithms",
        "Cooking recipes for pasta",
        "Neural network architectures"
    };
    int n_toks[] = {8, 6, 10};

    for (int d = 0; d < 3; d++) {
        int8_t embs[512 * COLBERT_EMB_DIM];
        fill_random_i8(embs, n_toks[d] * COLBERT_EMB_DIM, &state);
        /* Bias doc 0 and doc 2 to be "similar" by adding offset */
        if (d == 0 || d == 2) {
            for (int i = 0; i < n_toks[d] * COLBERT_EMB_DIM; i++) {
                int v = embs[i] + 30;
                if (v > 127) v = 127;
                embs[i] = (int8_t)v;
            }
        }
        int id = colbert_add_doc_i8(&db, embs, n_toks[d],
                                     texts[d], strlen(texts[d]));
        if (id != d) { FAIL("wrong doc id"); goto cleanup; }
    }

    if (db.n_docs != 3) { FAIL("wrong n_docs"); goto cleanup; }

    /* Search with query similar to doc 0 */
    int8_t query[COLBERT_MAX_QTOKENS * COLBERT_EMB_DIM];
    memset(query, 0, sizeof(query));
    /* Use first few tokens of doc 0 as the query */
    memcpy(query, tok_buf, 4 * COLBERT_EMB_DIM);

    colbert_result_t results[3];
    int n = colbert_search_i8(&db, query, 4, results, 3);
    if (n != 3) { FAIL("expected 3 results"); goto cleanup; }

    /* Doc 0 should be the top result (query is its own tokens) */
    if (results[0].doc_id != 0) {
        char msg[64];
        snprintf(msg, sizeof(msg), "top result is doc %d, expected 0", results[0].doc_id);
        FAIL(msg);
        goto cleanup;
    }

    /* Verify text retrieval */
    uint16_t len;
    const char *txt = colbert_get_text(&db, 0, &len);
    if (txt == 0 || len != strlen(texts[0]) ||
        memcmp(txt, texts[0], len) != 0) {
        FAIL("text retrieval mismatch");
        goto cleanup;
    }

    PASS();

cleanup:
    free(tok_buf);
    free(doc_buf);
    free(text_buf);
}

static void test_bench_maxsim(void)
{
    /* Benchmark MaxSim for realistic sizes */
    printf("\n  MaxSim Benchmarks:\n");
    printf("  %-12s %-8s %-8s %-12s %-12s\n",
           "n_q x n_d", "docs", "time_us", "docs/sec", "score");

    uint32_t state = 0xFEEDFACE;

    /* Simulate: 8 query tokens vs N documents of ~100 tokens each */
    int n_q = 8;
    int n_d_per_doc = 100;
    int doc_counts[] = {10, 50, 100, 500, 1000};
    int n_configs = sizeof(doc_counts) / sizeof(doc_counts[0]);

    int8_t query[32 * COLBERT_EMB_DIM];
    fill_random_i8(query, n_q * COLBERT_EMB_DIM, &state);

    for (int c = 0; c < n_configs; c++) {
        int n_docs = doc_counts[c];
        int total_dtokens = n_docs * n_d_per_doc;

        int8_t *doc_embs = (int8_t *)malloc(total_dtokens * COLBERT_EMB_DIM);
        fill_random_i8(doc_embs, total_dtokens * COLBERT_EMB_DIM, &state);

        /* Warm up */
        for (int d = 0; d < n_docs; d++) {
            colbert_maxsim_i8(query, &doc_embs[d * n_d_per_doc * COLBERT_EMB_DIM],
                              n_q, n_d_per_doc);
        }

        /* Benchmark: score all documents */
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        int32_t best_score = -2147483647 - 1;
        int reps = 5;
        for (int r = 0; r < reps; r++) {
            for (int d = 0; d < n_docs; d++) {
                int32_t s = colbert_maxsim_i8(query,
                    &doc_embs[d * n_d_per_doc * COLBERT_EMB_DIM],
                    n_q, n_d_per_doc);
                if (s > best_score) best_score = s;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        int64_t ns = (t1.tv_sec - t0.tv_sec) * 1000000000LL +
                     (t1.tv_nsec - t0.tv_nsec);
        int64_t us_per_search = ns / (reps * 1000);
        int64_t docs_per_sec = (int64_t)n_docs * reps * 1000000000LL / ns;

        printf("  %-12s %-8d %-8lld %-12lld %d\n",
               "", n_docs, (long long)us_per_search,
               (long long)docs_per_sec, best_score);

        free(doc_embs);
    }
}

int main(void)
{
    printf("ColBERT Engine Tests\n");
    printf("====================\n\n");

    printf("Correctness:\n");
    test_dot_product();
    test_maxsim();
    test_quantize();
    test_add_and_search();

    test_bench_maxsim();

    printf("\n====================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
