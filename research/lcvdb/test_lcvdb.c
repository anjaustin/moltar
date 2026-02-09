/* ==========================================================================
 * L-Cache VDB — Test Harness (split storage)
 * ==========================================================================
 * Exercises init, insert, delete, and search with new split storage layout.
 * Tests correctness of NEON dot products, HNSW graph build, and search.
 *
 * Compile:
 *   aarch64-linux-gnu-gcc -O2 -Wall -march=armv8-a -o test_lcvdb \
 *       test_lcvdb.c init_ref.c build_ref.c search_ref.c distance.S -static
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "lcvdb.h"

/* Simple PRNG for test data */
static uint32_t test_rng = 42;
static int8_t rand_i8(void) {
    test_rng ^= test_rng << 13;
    test_rng ^= test_rng >> 17;
    test_rng ^= test_rng << 5;
    return (int8_t)(test_rng & 0xFF);
}

static void rand_vector(int8_t *v) {
    for (int i = 0; i < LCVDB_VEC_DIM; i++)
        v[i] = rand_i8();
}

/* Reference dot product (for verification) */
static int32_t ref_dot(const int8_t *a, const int8_t *b) {
    int32_t sum = 0;
    for (int i = 0; i < LCVDB_VEC_DIM; i++)
        sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

int main(void) {
    int pass = 1;
    printf("L-Cache VDB Test Harness (Split Storage)\n");
    printf("=========================================\n\n");

    /* Allocate aligned memory */
    int max_nodes = 512;
    lcvdb_t db __attribute__((aligned(64)));
    void *topo_buf = NULL, *vec_buf = NULL;
    if (posix_memalign(&topo_buf, 64, max_nodes * sizeof(lcvdb_topo_t))) {
        fprintf(stderr, "Failed to allocate topo buffer\n");
        return 1;
    }
    if (posix_memalign(&vec_buf, 64, max_nodes * sizeof(lcvdb_vec_t))) {
        fprintf(stderr, "Failed to allocate vec buffer\n");
        return 1;
    }

    /* Initialize */
    lcvdb_init(&db, topo_buf, vec_buf, max_nodes);
    printf("Initialized: node_count=%u, M=%u, max_nodes=%u, entry=%u\n",
           db.node_count, db.M, db.max_nodes, db.entry_point);

    /* Size report */
    printf("\nMemory layout (split storage):\n");
    printf("  sizeof(lcvdb_t):      %zu bytes (1 cache line)\n", sizeof(lcvdb_t));
    printf("  sizeof(lcvdb_topo_t): %zu bytes (1/2 cache line)\n", sizeof(lcvdb_topo_t));
    printf("  sizeof(lcvdb_vec_t):  %zu bytes (1 cache line)\n", sizeof(lcvdb_vec_t));
    printf("  Topo buffer:          %zu bytes (%d nodes x %zu)\n",
           (size_t)max_nodes * sizeof(lcvdb_topo_t), max_nodes, sizeof(lcvdb_topo_t));
    printf("  Vec  buffer:          %zu bytes (%d nodes x %zu)\n",
           (size_t)max_nodes * sizeof(lcvdb_vec_t), max_nodes, sizeof(lcvdb_vec_t));
    printf("  Total data:           %zu bytes (%.1f KB)\n",
           sizeof(lcvdb_t) + max_nodes * (sizeof(lcvdb_topo_t) + sizeof(lcvdb_vec_t)),
           (sizeof(lcvdb_t) + max_nodes * (sizeof(lcvdb_topo_t) + sizeof(lcvdb_vec_t))) / 1024.0);

    /* Test 1: Distance function */
    printf("\n--- Test 1: Distance Function ---\n");
    {
        int8_t va[LCVDB_VEC_DIM], vb[LCVDB_VEC_DIM];
        rand_vector(va);
        rand_vector(vb);

        int32_t neon_dot = lcvdb_dot_i8(va, vb);
        int32_t ref = ref_dot(va, vb);
        printf("  NEON dot:  %d\n", neon_dot);
        printf("  Ref dot:   %d\n", ref);
        printf("  Match:     %s\n", neon_dot == ref ? "YES" : "NO");
        if (neon_dot != ref) pass = 0;
    }

    /* Test 2: Insert and graph structure */
    printf("\n--- Test 2: Insert (%d nodes) ---\n", max_nodes > 256 ? 256 : max_nodes);
    int num_insert = max_nodes > 256 ? 256 : max_nodes;
    int8_t (*vectors)[LCVDB_VEC_DIM] = malloc(num_insert * LCVDB_VEC_DIM);

    for (int i = 0; i < num_insert; i++) {
        rand_vector(vectors[i]);
        uint16_t id = lcvdb_insert(&db, vectors[i], (uint32_t)i * 100);
        if (i < 5 || i == num_insert - 1) {
            lcvdb_topo_t *t = &db.topo_array[id];
            printf("  Inserted node %u (layer=%u, nbrs=%u, payload=%u)\n",
                   id, t->max_layer, t->neighbor_count, t->payload_id);
        } else if (i == 5) {
            printf("  ...\n");
        }
    }
    printf("  Total nodes: %u, entry_point: %u, max_level: %u\n",
           db.node_count, db.entry_point, db.max_level);

    /* Verify vectors were stored correctly */
    printf("\n--- Test 3: Vector Storage Integrity ---\n");
    {
        int vec_ok = 1;
        for (int i = 0; i < num_insert; i++) {
            if (memcmp(db.vec_array[i].vector, vectors[i], LCVDB_VEC_DIM) != 0) {
                printf("  FAIL: vector %d mismatch\n", i);
                vec_ok = 0;
                pass = 0;
                break;
            }
        }
        if (vec_ok) printf("  All %d vectors stored correctly\n", num_insert);
    }

    /* Verify payload IDs */
    printf("\n--- Test 4: Payload IDs ---\n");
    {
        int pay_ok = 1;
        for (int i = 0; i < num_insert; i++) {
            if (db.topo_array[i].payload_id != (uint32_t)i * 100) {
                printf("  FAIL: node %d payload %u expected %u\n",
                       i, db.topo_array[i].payload_id, (uint32_t)i * 100);
                pay_ok = 0;
                pass = 0;
                break;
            }
        }
        if (pay_ok) printf("  All %d payload IDs correct\n", num_insert);
    }

    /* Test 5: Graph connectivity */
    printf("\n--- Test 5: Graph Connectivity ---\n");
    {
        uint8_t *bfs_visited = calloc((num_insert + 7) / 8, 1);
        uint16_t *bfs_queue = malloc(num_insert * sizeof(uint16_t));
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
        for (int i = 0; i < num_insert; i++)
            if (bfs_visited[i >> 3] & (1 << (i & 7))) reachable++;

        printf("  Reachable: %d/%d\n", reachable, num_insert);
        if (reachable < num_insert) {
            printf("  WARN: not fully connected\n");
        }

        free(bfs_visited);
        free(bfs_queue);
    }

    /* Test 6: Search correctness */
    printf("\n--- Test 6: Search ---\n");
    {
        int8_t query[LCVDB_VEC_DIM];
        rand_vector(query);

        uint16_t result_ids[8];
        int32_t result_scores[8];
        int k = 5;
        int nresults = lcvdb_search(&db, query, k, result_ids, result_scores);

        printf("  Query results (k=%d, returned=%d):\n", k, nresults);
        for (int i = 0; i < nresults; i++) {
            int32_t verify = ref_dot(query, vectors[result_ids[i]]);
            printf("    #%d: node=%u  score=%d  (verify=%d)  payload=%u  %s\n",
                   i, result_ids[i], result_scores[i], verify,
                   db.topo_array[result_ids[i]].payload_id,
                   result_scores[i] == verify ? "OK" : "MISMATCH");
            if (result_scores[i] != verify) pass = 0;
        }

        /* Brute force verification */
        printf("\n  Brute force top-%d:\n", k);
        int32_t *all_scores = malloc(num_insert * sizeof(int32_t));
        for (int i = 0; i < num_insert; i++)
            all_scores[i] = ref_dot(query, vectors[i]);

        for (int i = 0; i < k; i++) {
            int best_idx = 0;
            int32_t best_score = -999999;
            for (int j = 0; j < num_insert; j++) {
                if (all_scores[j] > best_score) {
                    best_score = all_scores[j];
                    best_idx = j;
                }
            }
            printf("    #%d: node=%d  score=%d\n", i, best_idx, best_score);
            all_scores[best_idx] = -999999;
        }
        free(all_scores);
    }

    /* Test 7: Delete */
    printf("\n--- Test 7: Delete ---\n");
    {
        /* Delete node 0 */
        lcvdb_delete(&db, 0);
        printf("  Deleted node 0, flags=0x%04x\n", db.topo_array[0].flags);
        if (!(db.topo_array[0].flags & LCVDB_FLAG_DELETED)) {
            printf("  FAIL: delete flag not set\n");
            pass = 0;
        }

        /* Search should not return node 0 */
        int8_t query[LCVDB_VEC_DIM];
        /* Use vector[0] as query — it should be its own best match,
         * but since deleted, it shouldn't appear */
        memcpy(query, vectors[0], LCVDB_VEC_DIM);

        uint16_t result_ids[8];
        int32_t result_scores[8];
        int nresults = lcvdb_search(&db, query, 5, result_ids, result_scores);

        int found_deleted = 0;
        for (int i = 0; i < nresults; i++) {
            if (result_ids[i] == 0) {
                found_deleted = 1;
                break;
            }
        }
        printf("  Search for deleted node's vector: %s (returned %d results)\n",
               found_deleted ? "FAIL (found deleted)" : "OK (not in results)", nresults);
        if (found_deleted) pass = 0;

        /* Un-delete for further tests */
        db.topo_array[0].flags &= ~LCVDB_FLAG_DELETED;
    }

    /* Final summary */
    printf("\n=========================================\n");
    printf("RESULT: %s\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("  Topo footprint: %zu bytes (%.1f KB) for %d nodes\n",
           (size_t)num_insert * sizeof(lcvdb_topo_t),
           (double)num_insert * sizeof(lcvdb_topo_t) / 1024.0, num_insert);
    printf("  Vec  footprint: %zu bytes (%.1f KB) for %d nodes\n",
           (size_t)num_insert * sizeof(lcvdb_vec_t),
           (double)num_insert * sizeof(lcvdb_vec_t) / 1024.0, num_insert);
    printf("  Total:          %.1f KB\n",
           (double)num_insert * (sizeof(lcvdb_topo_t) + sizeof(lcvdb_vec_t)) / 1024.0);

    free(vectors);
    free(topo_buf);
    free(vec_buf);
    return pass ? 0 : 1;
}
