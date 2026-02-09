/* ==========================================================================
 * L-Cache VDB — Test Harness
 * ==========================================================================
 * Exercises init, insert, and search on ARM AArch64.
 * Compile: aarch64-linux-gnu-gcc -O2 -o test_lcvdb test_lcvdb.c \
 *          init.S distance.S search.S build.S -lm
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

/* Generate a random 48D int8 vector */
static void rand_vector(int8_t *v) {
    for (int i = 0; i < LCVDB_VEC_DIM; i++) {
        v[i] = rand_i8();
    }
}

/* Reference dot product (for verification) */
static int32_t ref_dot(const int8_t *a, const int8_t *b) {
    int32_t sum = 0;
    for (int i = 0; i < LCVDB_VEC_DIM; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

int main(void) {
    printf("L-Cache VDB Test Harness\n");
    printf("========================\n\n");

    /* Allocate aligned memory */
    lcvdb_t db __attribute__((aligned(64)));
    void *node_buf = NULL;
    if (posix_memalign(&node_buf, 64, LCVDB_MAX_NODES * LCVDB_NODE_SIZE)) {
        fprintf(stderr, "Failed to allocate node buffer\n");
        return 1;
    }

    /* Initialize */
    lcvdb_init(&db, node_buf);
    printf("Initialized: node_count=%u, M=%u\n", db.node_count, db.M);

    /* Size report */
    printf("\nMemory layout:\n");
    printf("  sizeof(lcvdb_t):    %zu bytes (1 cache line)\n", sizeof(lcvdb_t));
    printf("  sizeof(lcvdb_node): %zu bytes (1 cache line)\n", sizeof(lcvdb_node_t));
    printf("  Node buffer:        %u bytes (%u nodes x %u bytes)\n",
           LCVDB_MAX_NODES * LCVDB_NODE_SIZE,
           LCVDB_MAX_NODES, LCVDB_NODE_SIZE);
    printf("  Total data:         %zu bytes (%.1f KB)\n",
           sizeof(lcvdb_t) + LCVDB_MAX_NODES * LCVDB_NODE_SIZE,
           (sizeof(lcvdb_t) + LCVDB_MAX_NODES * LCVDB_NODE_SIZE) / 1024.0);

    /* Test distance function */
    printf("\n--- Distance Function Test ---\n");
    int8_t va[LCVDB_VEC_DIM], vb[LCVDB_VEC_DIM];
    rand_vector(va);
    rand_vector(vb);

    int32_t neon_dot = lcvdb_dot_i8(va, vb);
    int32_t ref = ref_dot(va, vb);
    printf("  NEON dot:  %d\n", neon_dot);
    printf("  Ref dot:   %d\n", ref);
    printf("  Match:     %s\n", neon_dot == ref ? "YES" : "NO");

    /* Insert vectors */
    printf("\n--- Insert Test ---\n");
    int8_t vectors[LCVDB_MAX_NODES][LCVDB_VEC_DIM];
    int num_insert = 64;

    for (int i = 0; i < num_insert; i++) {
        rand_vector(vectors[i]);
        uint8_t id = lcvdb_insert(&db, vectors[i]);
        if (i < 5 || i == num_insert - 1) {
            printf("  Inserted node %u (layer=%u, neighbors=%u)\n",
                   id,
                   ((lcvdb_node_t *)node_buf)[id].max_layer,
                   ((lcvdb_node_t *)node_buf)[id].neighbor_count);
        } else if (i == 5) {
            printf("  ...\n");
        }
    }
    printf("  Total nodes: %u, entry_point: %u, max_level: %u\n",
           db.node_count, db.entry_point, db.max_level);

    /* Search test */
    printf("\n--- Search Test ---\n");
    int8_t query[LCVDB_VEC_DIM];
    rand_vector(query);

    uint8_t result_ids[8];
    int32_t result_scores[8];
    uint8_t k = 5;

    lcvdb_search(&db, query, k, result_ids, result_scores);

    printf("  Query results (k=%u):\n", k);
    for (int i = 0; i < k; i++) {
        int32_t verify = ref_dot(query, vectors[result_ids[i]]);
        printf("    #%d: node=%u  score=%d  (verify=%d)\n",
               i, result_ids[i], result_scores[i], verify);
    }

    /* Brute force verification */
    printf("\n  Brute force top-%u:\n", k);
    int32_t all_scores[LCVDB_MAX_NODES];
    for (int i = 0; i < num_insert; i++) {
        all_scores[i] = ref_dot(query, vectors[i]);
    }
    /* Simple selection of top-k */
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

    /* Final size summary */
    printf("\n========================\n");
    printf("TOTAL FOOTPRINT:\n");
    printf("  Code:  see 'size' output of compiled binary\n");
    printf("  Data:  %zu bytes (%.1f KB)\n",
           sizeof(lcvdb_t) + num_insert * LCVDB_NODE_SIZE,
           (sizeof(lcvdb_t) + num_insert * LCVDB_NODE_SIZE) / 1024.0);
    printf("  Stack: ~300 bytes (visited + beam + saved regs)\n");

    free(node_buf);
    return 0;
}
