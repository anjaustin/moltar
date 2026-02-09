/* Minimal debug test for search segfault */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "lcvdb.h"

int main(void) {
    lcvdb_t db __attribute__((aligned(64)));
    void *node_buf = NULL;
    posix_memalign(&node_buf, 64, LCVDB_MAX_NODES * LCVDB_NODE_SIZE);
    memset(node_buf, 0, LCVDB_MAX_NODES * LCVDB_NODE_SIZE);

    lcvdb_init(&db, node_buf);

    /* Insert just 3 nodes with known vectors */
    int8_t v0[48], v1[48], v2[48], query[48];
    memset(v0, 10, 48);   /* all 10s */
    memset(v1, -10, 48);  /* all -10s */
    memset(v2, 5, 48);    /* all 5s */
    memset(query, 10, 48); /* query = same as v0 */

    printf("Inserting node 0...\n");
    lcvdb_insert(&db, v0);
    printf("  node_count=%u entry=%u max_level=%u\n", db.node_count, db.entry_point, db.max_level);

    printf("Inserting node 1...\n");
    lcvdb_insert(&db, v1);
    printf("  node_count=%u entry=%u max_level=%u\n", db.node_count, db.entry_point, db.max_level);

    printf("Inserting node 2...\n");
    lcvdb_insert(&db, v2);
    printf("  node_count=%u entry=%u max_level=%u\n", db.node_count, db.entry_point, db.max_level);

    /* Check node structure */
    lcvdb_node_t *nodes = (lcvdb_node_t *)node_buf;
    for (int i = 0; i < 3; i++) {
        printf("Node %d: layer=%u, ncnt=%u, neighbors=[", i, nodes[i].max_layer, nodes[i].neighbor_count);
        for (int j = 0; j < nodes[i].neighbor_count; j++) {
            printf("%u%s", nodes[i].neighbors[j], j < nodes[i].neighbor_count - 1 ? "," : "");
        }
        printf("]\n");
    }

    printf("\nSearching (k=1)...\n");
    uint8_t result_ids[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    int32_t result_scores[8] = {0};

    lcvdb_search(&db, query, 1, result_ids, result_scores);
    printf("Result: id=%u score=%d\n", result_ids[0], result_scores[0]);

    printf("\nSearching (k=3)...\n");
    lcvdb_search(&db, query, 3, result_ids, result_scores);
    for (int i = 0; i < 3; i++) {
        printf("  #%d: id=%u score=%d\n", i, result_ids[i], result_scores[i]);
    }

    free(node_buf);
    return 0;
}
