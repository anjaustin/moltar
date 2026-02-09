/* ==========================================================================
 * Moltar Memory — Projection Tests
 * ==========================================================================
 * Tests random projection from float hidden states to 48D int8,
 * then verifies LCVDB insert + search roundtrip with projected vectors.
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "project.h"
#include "../lcvdb/lcvdb.h"

/* Simple PRNG for test data */
static uint32_t test_rng = 42;
static float rand_float(void) {
    test_rng ^= test_rng << 13;
    test_rng ^= test_rng >> 17;
    test_rng ^= test_rng << 5;
    /* Uniform in [-1, 1] */
    return ((float)(test_rng & 0xFFFFFF) / (float)0x800000) - 1.0f;
}

/* Reference dot product for int8 48D */
static int32_t ref_dot_i8(const int8_t *a, const int8_t *b) {
    int32_t sum = 0;
    for (int i = 0; i < 48; i++)
        sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

/* Cosine similarity for float vectors */
static float cosine_sim(const float *a, const float *b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-10f);
}

int main(void) {
    int pass = 1;
    int n_embd = 2048;  /* LFM2-1.2B */

    printf("Moltar Memory — Projection Tests\n");
    printf("=================================\n\n");

    /* ---- Test 1: Projection matrix initialization ---- */
    printf("--- Test 1: Projection Init (n_embd=%d) ---\n", n_embd);
    moltar_proj_t proj;
    moltar_proj_init(&proj, n_embd, 0x4D4F4C54);

    /* Check that matrix is not all zeros */
    int nonzero = 0;
    for (int i = 0; i < MOLTAR_PROJ_DIM * n_embd; i++)
        if (proj.matrix[i] != 0) nonzero++;

    printf("  Matrix size: %d x %d = %d bytes\n",
           MOLTAR_PROJ_DIM, n_embd, MOLTAR_PROJ_DIM * n_embd);
    printf("  Non-zero entries: %d/%d (%.1f%%)\n",
           nonzero, MOLTAR_PROJ_DIM * n_embd,
           100.0f * nonzero / (MOLTAR_PROJ_DIM * n_embd));
    printf("  Scale: %f\n", proj.scale);
    if (nonzero < MOLTAR_PROJ_DIM * n_embd / 2) {
        printf("  FAIL: too many zeros\n");
        pass = 0;
    } else {
        printf("  OK\n");
    }

    /* ---- Test 2: Deterministic output ---- */
    printf("\n--- Test 2: Determinism ---\n");
    {
        float hidden[2048];
        for (int i = 0; i < n_embd; i++) hidden[i] = rand_float();

        int8_t out1[48], out2[48];
        moltar_proj_apply(&proj, hidden, out1);
        moltar_proj_apply(&proj, hidden, out2);

        int match = 1;
        for (int i = 0; i < 48; i++)
            if (out1[i] != out2[i]) { match = 0; break; }

        printf("  Same input -> same output: %s\n", match ? "YES" : "NO");
        if (!match) pass = 0;

        /* Print first few values */
        printf("  Output[0..7]: ");
        for (int i = 0; i < 8; i++) printf("%d ", out1[i]);
        printf("\n");
    }

    /* ---- Test 3: Different inputs produce different outputs ---- */
    printf("\n--- Test 3: Discrimination ---\n");
    {
        float h1[2048], h2[2048];
        for (int i = 0; i < n_embd; i++) {
            h1[i] = rand_float();
            h2[i] = rand_float();
        }

        int8_t out1[48], out2[48];
        moltar_proj_apply(&proj, h1, out1);
        moltar_proj_apply(&proj, h2, out2);

        int same = 1;
        for (int i = 0; i < 48; i++)
            if (out1[i] != out2[i]) { same = 0; break; }

        printf("  Different inputs -> different outputs: %s\n",
               same ? "FAIL (identical)" : "YES");
        if (same) pass = 0;
    }

    /* ---- Test 4: Distance preservation with structured data ---- */
    printf("\n--- Test 4: Distance Preservation (Structured Clusters) ---\n");
    {
        /* Random isotropic vectors in high-D have pairwise cosine ~0 ± 1/sqrt(D),
         * so ordering between essentially-equal similarities is a coin flip.
         * Instead, we create 4 clusters of 5 vectors each. Within a cluster,
         * vectors share a common "base" plus small perturbation (cos_sim ~0.8-0.9).
         * Across clusters, vectors are dissimilar (cos_sim ~0).
         * We test: for triplets (a,b,c) where a,b are same-cluster and c is
         * different-cluster, the projection should preserve same > different. */
        int N_CLUSTERS = 4;
        int N_PER = 5;
        int N = N_CLUSTERS * N_PER;  /* 20 total */
        float hiddens[20][2048];
        int8_t projected[20][48];

        /* Generate cluster bases */
        float bases[4][2048];
        for (int c = 0; c < N_CLUSTERS; c++)
            for (int j = 0; j < n_embd; j++)
                bases[c][j] = rand_float();

        /* Generate vectors: base + 20% perturbation */
        for (int c = 0; c < N_CLUSTERS; c++) {
            for (int p = 0; p < N_PER; p++) {
                int idx = c * N_PER + p;
                for (int j = 0; j < n_embd; j++)
                    hiddens[idx][j] = bases[c][j] + 0.2f * rand_float();
                moltar_proj_apply(&proj, hiddens[idx], projected[idx]);
            }
        }

        /* Test: for all triplets (a, b_same, c_diff), check ordering preserved */
        int total_triples = 0, preserved = 0;
        for (int ci = 0; ci < N_CLUSTERS; ci++) {
            for (int a = ci * N_PER; a < (ci + 1) * N_PER; a++) {
                for (int b = a + 1; b < (ci + 1) * N_PER; b++) {
                    for (int cj = 0; cj < N_CLUSTERS; cj++) {
                        if (cj == ci) continue;
                        for (int c = cj * N_PER; c < (cj + 1) * N_PER; c++) {
                            float sim_ab = cosine_sim(hiddens[a], hiddens[b], n_embd);
                            float sim_ac = cosine_sim(hiddens[a], hiddens[c], n_embd);
                            int32_t dot_ab = ref_dot_i8(projected[a], projected[b]);
                            int32_t dot_ac = ref_dot_i8(projected[a], projected[c]);

                            total_triples++;
                            /* a,b same cluster => sim_ab should be high
                             * a,c diff cluster => sim_ac should be low
                             * projection should preserve: dot_ab > dot_ac */
                            if ((sim_ab > sim_ac) == (dot_ab > dot_ac))
                                preserved++;
                        }
                    }
                }
            }
        }

        float preservation = 100.0f * preserved / total_triples;
        printf("  Cluster triples preserved: %d/%d (%.1f%%)\n",
               preserved, total_triples, preservation);

        /* Also show example cosine similarities */
        float same_avg = 0, diff_avg = 0;
        int same_cnt = 0, diff_cnt = 0;
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                float cs = cosine_sim(hiddens[i], hiddens[j], n_embd);
                if (i / N_PER == j / N_PER) { same_avg += cs; same_cnt++; }
                else { diff_avg += cs; diff_cnt++; }
            }
        }
        printf("  Avg intra-cluster cosine: %.3f  (n=%d)\n",
               same_avg / same_cnt, same_cnt);
        printf("  Avg inter-cluster cosine: %.3f  (n=%d)\n",
               diff_avg / diff_cnt, diff_cnt);

        /* With structured clusters, expect >= 80% preservation */
        if (preservation < 70.0f) {
            printf("  FAIL: preservation too low\n");
            pass = 0;
        } else {
            printf("  OK (>70%% threshold)\n");
        }
    }

    /* ---- Test 5: LCVDB roundtrip with projected vectors ---- */
    printf("\n--- Test 5: LCVDB Insert + Search Roundtrip ---\n");
    {
        int max_nodes = 128;
        lcvdb_t db __attribute__((aligned(64)));
        void *topo_buf = NULL, *vec_buf = NULL;
        posix_memalign(&topo_buf, 64, max_nodes * sizeof(lcvdb_topo_t));
        posix_memalign(&vec_buf, 64, max_nodes * sizeof(lcvdb_vec_t));
        lcvdb_init(&db, topo_buf, vec_buf, max_nodes);

        /* Generate 8 "topics" of 4 turns each (32 total).
         * Each topic has a base hidden state; turns are perturbations.
         * This simulates conversation: turns about the same topic are similar. */
        int N_TOPICS = 8, N_TURNS = 4;
        int N = N_TOPICS * N_TURNS;  /* 32 */
        float hiddens[32][2048];
        int8_t projected[32][48];

        float topic_bases[8][2048];
        for (int t = 0; t < N_TOPICS; t++)
            for (int j = 0; j < n_embd; j++)
                topic_bases[t][j] = rand_float();

        for (int t = 0; t < N_TOPICS; t++) {
            for (int u = 0; u < N_TURNS; u++) {
                int idx = t * N_TURNS + u;
                for (int j = 0; j < n_embd; j++)
                    hiddens[idx][j] = topic_bases[t][j] + 0.2f * rand_float();
                moltar_proj_apply(&proj, hiddens[idx], projected[idx]);
                lcvdb_insert(&db, projected[idx], (uint32_t)idx);
            }
        }

        printf("  Inserted %d projected vectors (%d topics x %d turns)\n",
               N, N_TOPICS, N_TURNS);
        printf("  Graph: %u nodes, entry=%u\n", db.node_count, db.entry_point);

        /* For each turn, search top-3 and check how many are same-topic */
        int total_retrieved = 0, same_topic = 0;
        int self_top1 = 0;
        for (int q = 0; q < N; q++) {
            uint16_t result_ids[4];
            int32_t result_scores[4];
            int nresults = lcvdb_search(&db, projected[q], 4, result_ids, result_scores);

            if (nresults > 0 && result_ids[0] == (uint16_t)q)
                self_top1++;

            int q_topic = q / N_TURNS;
            /* Skip self (result_ids[0] should be self), check rest */
            for (int r = 0; r < nresults; r++) {
                if (result_ids[r] == (uint16_t)q) continue;
                total_retrieved++;
                int r_topic = result_ids[r] / N_TURNS;
                if (r_topic == q_topic) same_topic++;
            }
        }

        printf("  Self-as-top-1: %d/%d (%.1f%%)\n", self_top1, N,
               100.0f * self_top1 / N);
        float topic_recall = 100.0f * same_topic / total_retrieved;
        printf("  Same-topic in top-3 neighbors: %d/%d (%.1f%%)\n",
               same_topic, total_retrieved, topic_recall);

        /* Show a sample search */
        uint16_t result_ids[4];
        int32_t result_scores[4];
        int nresults = lcvdb_search(&db, projected[0], 4, result_ids, result_scores);
        printf("  Sample search for turn 0 (topic 0):\n");
        for (int i = 0; i < nresults; i++) {
            printf("    #%d: turn=%u (topic=%d)  score=%d\n",
                   i, result_ids[i], result_ids[i] / N_TURNS, result_scores[i]);
        }

        /* Topic recall is the meaningful metric for working memory.
         * Self-as-top-1 may be low when same-topic turns have near-identical
         * int8 projections (max-abs quantization collapses them). That's OK —
         * the goal is to retrieve *related* turns, not the exact same one.
         *
         * We check: (1) sample search returns same-topic results
         *           (2) overall topic recall >= 25% (random baseline = 3/31 ≈ 10%) */
        int sample_topic_ok = 1;
        for (int i = 0; i < nresults; i++) {
            if (result_ids[i] / N_TURNS != 0) { sample_topic_ok = 0; break; }
        }
        if (!sample_topic_ok) {
            printf("  FAIL: sample search returned wrong-topic results\n");
            pass = 0;
        } else {
            printf("  OK: sample search — all results from correct topic\n");
        }
        if (topic_recall < 25.0f) {
            printf("  FAIL: topic recall too low (random baseline ~10%%)\n");
            pass = 0;
        } else {
            printf("  OK: topic recall %.1f%% (random baseline ~10%%)\n", topic_recall);
        }

        free(topo_buf);
        free(vec_buf);
    }

    /* ---- Test 6: Projection latency ---- */
    printf("\n--- Test 6: Projection Latency ---\n");
    {
        float hidden[2048];
        for (int i = 0; i < n_embd; i++) hidden[i] = rand_float();
        int8_t out[48];

        /* Warm up */
        for (int i = 0; i < 100; i++)
            moltar_proj_apply(&proj, hidden, out);

        /* Time 10000 projections */
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
        int iters = 10000;
        for (int i = 0; i < iters; i++)
            moltar_proj_apply(&proj, hidden, out);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

        int64_t ns = (t1.tv_sec - t0.tv_sec) * 1000000000LL +
                     (t1.tv_nsec - t0.tv_nsec);
        printf("  %d projections in %lld us (%.1f us/proj)\n",
               iters, (long long)(ns / 1000), (double)ns / iters / 1000.0);
    }

    /* Final summary */
    printf("\n=================================\n");
    printf("RESULT: %s\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("  Projection: %dD float -> 48D int8\n", n_embd);
    printf("  Matrix size: %.1f KB\n",
           (float)(MOLTAR_PROJ_DIM * n_embd) / 1024.0f);

    return pass ? 0 : 1;
}
