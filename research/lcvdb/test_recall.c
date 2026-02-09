/* L-Cache VDB — Recall benchmark
 * Tests search recall@k at various N with statistical rigor.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
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

static int32_t ref_dot(const int8_t *a, const int8_t *b) {
    int32_t sum = 0;
    for (int i = 0; i < LCVDB_VEC_DIM; i++)
        sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

/* Brute-force top-k. Returns IDs in bf_ids sorted by score descending. */
static void brute_force_topk(const int8_t *query, const int8_t vecs[][LCVDB_VEC_DIM],
                             int n, int k, uint8_t *bf_ids) {
    int32_t scores[256];
    uint8_t ids[256];
    for (int i = 0; i < n; i++) {
        scores[i] = ref_dot(query, vecs[i]);
        ids[i] = i;
    }
    /* Partial selection sort for top-k */
    for (int i = 0; i < k && i < n; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) {
            if (scores[j] > scores[best])
                best = j;
        }
        if (best != i) {
            int32_t ts = scores[i]; scores[i] = scores[best]; scores[best] = ts;
            uint8_t ti = ids[i]; ids[i] = ids[best]; ids[best] = ti;
        }
        bf_ids[i] = ids[i];
    }
}

int main(void) {
    printf("L-Cache VDB Recall Benchmark\n");
    printf("============================\n");
    printf("ef_search=%d, M=%d\n\n", LCVDB_EF_SEARCH, LCVDB_M);

    int test_sizes[] = {32, 64, 128, 256};
    int num_sizes = 4;
    int num_queries = 50;
    int k_values[] = {1, 5, 10};
    int num_k = 3;

    for (int si = 0; si < num_sizes; si++) {
        int N = test_sizes[si];
        if (N > LCVDB_MAX_NODES) continue;

        rng_seed(12345 + N);  /* Reproducible per N */

        /* Allocate */
        lcvdb_t db __attribute__((aligned(64)));
        void *node_buf = NULL;
        if (posix_memalign(&node_buf, 64, LCVDB_MAX_NODES * LCVDB_NODE_SIZE)) {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }
        memset(node_buf, 0, LCVDB_MAX_NODES * LCVDB_NODE_SIZE);
        lcvdb_init(&db, node_buf);

        /* Insert N vectors */
        int8_t vecs[256][LCVDB_VEC_DIM];
        for (int i = 0; i < N; i++) {
            rand_vector(vecs[i]);
            lcvdb_insert(&db, vecs[i]);
        }

        /* Check graph connectivity */
        lcvdb_node_t *nodes = (lcvdb_node_t *)node_buf;
        uint8_t bfs_visited[256] = {0};
        uint8_t bfs_queue[256];
        int bfs_head = 0, bfs_tail = 0;
        bfs_queue[bfs_tail++] = db.entry_point;
        bfs_visited[db.entry_point] = 1;
        while (bfs_head < bfs_tail) {
            uint8_t n = bfs_queue[bfs_head++];
            for (int i = 0; i < nodes[n].neighbor_count; i++) {
                uint8_t nb = nodes[n].neighbors[i];
                if (!bfs_visited[nb]) {
                    bfs_visited[nb] = 1;
                    bfs_queue[bfs_tail++] = nb;
                }
            }
        }
        int reachable = 0;
        for (int i = 0; i < N; i++)
            if (bfs_visited[i]) reachable++;

        /* Count bidirectional edges and neighbor stats */
        int bidir = 0, total_edges = 0;
        int nbr_hist[9] = {0};  /* histogram of neighbor counts 0..8 */
        for (int i = 0; i < N; i++) {
            int nc = nodes[i].neighbor_count;
            if (nc <= 8) nbr_hist[nc]++;
            for (int j = 0; j < nc; j++) {
                total_edges++;
                uint8_t nb = nodes[i].neighbors[j];
                for (int k2 = 0; k2 < nodes[nb].neighbor_count; k2++) {
                    if (nodes[nb].neighbors[k2] == i) { bidir++; break; }
                }
            }
        }

        /* Average neighbor count */
        double avg_nbr = (double)total_edges / N;

        printf("N=%3d | reachable=%d/%d | edges=%d (%.0f%% bidir) | avg_nbr=%.1f\n",
               N, reachable, N, total_edges, 100.0 * bidir / total_edges, avg_nbr);
        printf("       nbr_hist: ");
        for (int i = 0; i <= 8; i++)
            if (nbr_hist[i]) printf("%d:%d ", i, nbr_hist[i]);
        printf("\n");

        /* Run queries and measure recall */
        for (int ki = 0; ki < num_k; ki++) {
            int k = k_values[ki];
            if (k > N) continue;

            int total_hits = 0;
            int total_possible = 0;

            for (int q = 0; q < num_queries; q++) {
                int8_t query[LCVDB_VEC_DIM];
                rand_vector(query);

                /* HNSW search */
                uint8_t hnsw_ids[16];
                int32_t hnsw_scores[16];
                memset(hnsw_ids, 0xFF, sizeof(hnsw_ids));
                lcvdb_search(&db, query, k, hnsw_ids, hnsw_scores);

                /* Brute force */
                uint8_t bf_ids[16];
                brute_force_topk(query, vecs, N, k, bf_ids);

                /* Count hits: how many of brute-force top-k are in HNSW results */
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        if (hnsw_ids[j] == bf_ids[i]) {
                            total_hits++;
                            break;
                        }
                    }
                }
                total_possible += k;
            }

            double recall = (double)total_hits / total_possible;
            printf("       recall@%d = %.1f%% (%d/%d)\n",
                   k, 100.0 * recall, total_hits, total_possible);
        }

        printf("\n");
        free(node_buf);
    }

    return 0;
}
