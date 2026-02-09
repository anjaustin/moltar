/* L-Cache VDB — Recall benchmark (split storage)
 * Tests search recall@k at various N with statistical rigor.
 * Now supports N up to 4096+ with split storage layout.
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
static void brute_force_topk(const int8_t *query, const int8_t *vecs,
                             int n, int k, uint16_t *bf_ids) {
    /* Use heap-like approach for large N */
    int32_t *scores = malloc(n * sizeof(int32_t));
    uint16_t *ids = malloc(n * sizeof(uint16_t));
    for (int i = 0; i < n; i++) {
        scores[i] = ref_dot(query, vecs + i * LCVDB_VEC_DIM);
        ids[i] = (uint16_t)i;
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
            uint16_t ti = ids[i]; ids[i] = ids[best]; ids[best] = ti;
        }
        bf_ids[i] = ids[i];
    }
    free(scores);
    free(ids);
}

int main(void) {
    printf("L-Cache VDB Recall Benchmark (Split Storage)\n");
    printf("=============================================\n");
    printf("ef_search=%d, M=%d\n\n", LCVDB_EF_SEARCH, LCVDB_M);

    int test_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_sizes = 6;
    int num_queries = 50;
    int k_values[] = {1, 5, 10};
    int num_k = 3;

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

        /* Store vectors separately for brute-force comparison */
        int8_t *vecs_flat = malloc(N * LCVDB_VEC_DIM);

        /* Insert N vectors */
        for (int i = 0; i < N; i++) {
            int8_t v[LCVDB_VEC_DIM];
            rand_vector(v);
            memcpy(vecs_flat + i * LCVDB_VEC_DIM, v, LCVDB_VEC_DIM);
            lcvdb_insert(&db, v, (uint32_t)i);
        }

        /* Check graph connectivity */
        uint8_t *bfs_visited = calloc((N + 7) / 8, 1);
        uint16_t *bfs_queue = malloc(N * sizeof(uint16_t));
        int bfs_head = 0, bfs_tail = 0;

        uint16_t ep = db.entry_point;
        bfs_visited[ep >> 3] |= (1 << (ep & 7));
        bfs_queue[bfs_tail++] = ep;

        while (bfs_head < bfs_tail) {
            uint16_t n = bfs_queue[bfs_head++];
            for (int i = 0; i < db.topo_array[n].neighbor_count; i++) {
                uint16_t nb = db.topo_array[n].neighbors[i];
                if (!(bfs_visited[nb >> 3] & (1 << (nb & 7)))) {
                    bfs_visited[nb >> 3] |= (1 << (nb & 7));
                    bfs_queue[bfs_tail++] = nb;
                }
            }
        }

        int reachable = 0;
        for (int i = 0; i < N; i++)
            if (bfs_visited[i >> 3] & (1 << (i & 7))) reachable++;

        /* Count bidirectional edges and neighbor stats */
        int bidir = 0, total_edges = 0;
        int nbr_hist[LCVDB_M + 1] = {0};
        for (int i = 0; i < N; i++) {
            int nc = db.topo_array[i].neighbor_count;
            if (nc <= LCVDB_M) nbr_hist[nc]++;
            for (int j = 0; j < nc; j++) {
                total_edges++;
                uint16_t nb = db.topo_array[i].neighbors[j];
                for (int k2 = 0; k2 < db.topo_array[nb].neighbor_count; k2++) {
                    if (db.topo_array[nb].neighbors[k2] == (uint16_t)i) { bidir++; break; }
                }
            }
        }

        int bidir_pct = total_edges > 0 ? (int)(100LL * bidir / total_edges) : 0;
        int avg_nbr_x10 = (int)(10LL * total_edges / N);

        printf("N=%4d | reach=%d/%d | edges=%d (%d%% bidir) | avg_nbr=%d.%d\n",
               N, reachable, N, total_edges,
               bidir_pct, avg_nbr_x10 / 10, avg_nbr_x10 % 10);
        int topo_kb = N * (int)sizeof(lcvdb_topo_t) / 1024;
        int vec_kb = N * (int)sizeof(lcvdb_vec_t) / 1024;
        printf("        topo=%dKB vec=%dKB total=%dKB\n", topo_kb, vec_kb, topo_kb + vec_kb);
        printf("        nbr_hist: ");
        for (int i = 0; i <= LCVDB_M; i++)
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
                uint16_t hnsw_ids[16];
                int32_t hnsw_scores[16];
                memset(hnsw_ids, 0xFF, sizeof(hnsw_ids));
                int nresults = lcvdb_search(&db, query, k, hnsw_ids, hnsw_scores);

                /* Brute force */
                uint16_t bf_ids[16];
                brute_force_topk(query, vecs_flat, N, k, bf_ids);

                /* Count hits */
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < nresults; j++) {
                        if (hnsw_ids[j] == bf_ids[i]) {
                            total_hits++;
                            break;
                        }
                    }
                }
                total_possible += k;
            }

            int recall_pct_x10 = (int)(1000LL * total_hits / total_possible);
            printf("        recall@%d = %d.%d%% (%d/%d)\n",
                   k, recall_pct_x10 / 10, recall_pct_x10 % 10, total_hits, total_possible);
        }

        printf("\n");
        free(bfs_visited);
        free(bfs_queue);
        free(vecs_flat);
        free(topo_buf);
        free(vec_buf);
    }

    return 0;
}
