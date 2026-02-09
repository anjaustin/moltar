/* L-Cache VDB — Performance benchmark (split storage) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "lcvdb.h"

static uint32_t rng_state;
static void rng_seed(uint32_t seed) { rng_state = seed; }
static int8_t rand_i8(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (int8_t)(rng_state & 0xFF);
}
static void rand_vector(int8_t *v) {
    for (int i = 0; i < LCVDB_VEC_DIM; i++)
        v[i] = rand_i8();
}

volatile uint16_t sink_u16;
volatile int32_t sink_i32;

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(void) {
    printf("L-Cache VDB Performance Benchmark (Split Storage)\n");
    printf("==================================================\n");
    printf("ef_search=%d, M=%d, dim=%d\n\n", LCVDB_EF_SEARCH, LCVDB_M, LCVDB_VEC_DIM);

    int test_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_sizes = 6;
    int k_values[] = {1, 5, 10};
    int num_k = 3;

    /* Verify timer works */
    {
        int64_t a = now_ns();
        volatile int x = 0;
        for (int i = 0; i < 1000000; i++) x += i;
        int64_t b = now_ns();
        printf("Timer check: diff=%lld ns (expect ~ms)\n\n", (long long)(b - a));
    }

    int warmup = 2000;
    int bench_iters = 100000;

    for (int si = 0; si < num_sizes; si++) {
        int N = test_sizes[si];

        rng_seed(12345 + N);

        /* Allocate split storage */
        lcvdb_t db __attribute__((aligned(64)));
        void *topo_buf = NULL, *vec_buf = NULL;
        if (posix_memalign(&topo_buf, 64, N * sizeof(lcvdb_topo_t)) ||
            posix_memalign(&vec_buf, 64, N * sizeof(lcvdb_vec_t))) {
            fprintf(stderr, "alloc failed for N=%d\n", N);
            return 1;
        }
        lcvdb_init(&db, topo_buf, vec_buf, N);

        for (int i = 0; i < N; i++) {
            int8_t v[LCVDB_VEC_DIM];
            rand_vector(v);
            lcvdb_insert(&db, v, (uint32_t)i);
        }

        printf("N=%4d  topo=%.1fKB vec=%.1fKB total=%.1fKB\n", N,
               (double)N * 32 / 1024.0,
               (double)N * 64 / 1024.0,
               (double)N * 96 / 1024.0);

        /* Reduce iterations for large N */
        int iters = bench_iters;
        if (N >= 1024) iters = 10000;
        else if (N >= 512) iters = 50000;

        for (int ki = 0; ki < num_k; ki++) {
            int k = k_values[ki];
            if (k > N) continue;

            int nq = 1000;
            int8_t *queries = malloc(nq * LCVDB_VEC_DIM);
            rng_seed(99999 + N * 100 + k);
            for (int i = 0; i < nq; i++)
                rand_vector(queries + i * LCVDB_VEC_DIM);

            uint16_t result_ids[16];
            int32_t result_scores[16];

            for (int i = 0; i < warmup; i++) {
                lcvdb_search(&db, queries + (i % nq) * LCVDB_VEC_DIM,
                            k, result_ids, result_scores);
                sink_u16 = result_ids[0];
            }

            int64_t t0 = now_ns();
            for (int i = 0; i < iters; i++) {
                lcvdb_search(&db, queries + (i % nq) * LCVDB_VEC_DIM,
                            k, result_ids, result_scores);
                sink_u16 = result_ids[0];
                sink_i32 = result_scores[0];
            }
            int64_t t1 = now_ns();

            int64_t elapsed = t1 - t0;
            int64_t ns_per = elapsed / iters;
            int64_t cycles = ns_per * 22 / 10;  /* 2.2 GHz A78 */
            int64_t qps = (int64_t)iters * 1000000000LL / elapsed;

            printf("  k=%2d: %5lld ns  %5lld cyc  %7lld qps\n",
                   k, (long long)ns_per, (long long)cycles, (long long)qps);

            free(queries);
        }

        /* Bench insert */
        {
            int insert_iters = N >= 1024 ? 200 : 2000;

            int64_t t0 = now_ns();
            for (int iter = 0; iter < insert_iters; iter++) {
                lcvdb_init(&db, topo_buf, vec_buf, N);
                rng_seed(12345 + N);
                for (int i = 0; i < N; i++) {
                    int8_t v[LCVDB_VEC_DIM];
                    rand_vector(v);
                    lcvdb_insert(&db, v, (uint32_t)i);
                }
                sink_u16 = (uint16_t)db.node_count;
            }
            int64_t t1 = now_ns();

            int64_t elapsed = t1 - t0;
            int64_t us_build = elapsed / insert_iters / 1000;
            int64_t ns_insert = elapsed / ((int64_t)insert_iters * N);

            printf("  build: %lld us  (%lld ns/insert)\n",
                   (long long)us_build, (long long)ns_insert);
        }

        printf("\n");
        free(topo_buf);
        free(vec_buf);
    }

    return 0;
}
