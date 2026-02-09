/* Diagnostic test: dump graph connectivity and trace search */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "lcvdb.h"

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

static int32_t ref_dot(const int8_t *a, const int8_t *b) {
    int32_t sum = 0;
    for (int i = 0; i < LCVDB_VEC_DIM; i++)
        sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

int main(void) {
    lcvdb_t db __attribute__((aligned(64)));
    void *node_buf = NULL;
    posix_memalign(&node_buf, 64, LCVDB_MAX_NODES * LCVDB_NODE_SIZE);
    memset(node_buf, 0, LCVDB_MAX_NODES * LCVDB_NODE_SIZE);
    lcvdb_init(&db, node_buf);

    int8_t vectors[LCVDB_MAX_NODES][LCVDB_VEC_DIM];
    int num_insert = 64;

    for (int i = 0; i < num_insert; i++) {
        rand_vector(vectors[i]);
        lcvdb_insert(&db, vectors[i]);
    }

    lcvdb_node_t *nodes = (lcvdb_node_t *)node_buf;

    printf("=== GRAPH STRUCTURE ===\n");
    printf("entry_point=%u, max_level=%u, node_count=%u\n\n",
           db.entry_point, db.max_level, db.node_count);

    /* Check graph connectivity */
    int orphan_count = 0;
    int total_edges = 0;
    int max_nbrs = 0;
    int bidir_ok = 0, bidir_fail = 0;

    for (int i = 0; i < num_insert; i++) {
        int nc = nodes[i].neighbor_count;
        total_edges += nc;
        if (nc > max_nbrs) max_nbrs = nc;
        if (nc == 0 && i > 0) {
            orphan_count++;
            printf("  ORPHAN: node %d (layer=%u, 0 neighbors)\n", i, nodes[i].max_layer);
        }

        /* Check bidirectionality */
        for (int j = 0; j < nc; j++) {
            uint8_t nb = nodes[i].neighbors[j];
            int found = 0;
            for (int k = 0; k < nodes[nb].neighbor_count; k++) {
                if (nodes[nb].neighbors[k] == i) { found = 1; break; }
            }
            if (found) bidir_ok++;
            else bidir_fail++;
        }
    }

    printf("Total edges: %d (avg %.1f per node)\n", total_edges, (float)total_edges / num_insert);
    printf("Max neighbors: %d\n", max_nbrs);
    printf("Orphans: %d\n", orphan_count);
    printf("Bidirectional edges: %d ok, %d one-way\n", bidir_ok, bidir_fail);

    /* Check for duplicate neighbors */
    int dup_count = 0;
    for (int i = 0; i < num_insert; i++) {
        for (int j = 0; j < nodes[i].neighbor_count; j++) {
            for (int k = j + 1; k < nodes[i].neighbor_count; k++) {
                if (nodes[i].neighbors[j] == nodes[i].neighbors[k]) {
                    dup_count++;
                    printf("  DUPLICATE: node %d has neighbor %u twice (pos %d and %d)\n",
                           i, nodes[i].neighbors[j], j, k);
                }
            }
            /* Check self-loop */
            if (nodes[i].neighbors[j] == i) {
                printf("  SELF-LOOP: node %d points to itself at pos %d\n", i, j);
            }
        }
    }
    printf("Duplicate edges: %d\n", dup_count);

    /* Layer distribution */
    int layer_counts[LCVDB_MAX_LAYERS] = {0};
    for (int i = 0; i < num_insert; i++) {
        if (nodes[i].max_layer < LCVDB_MAX_LAYERS)
            layer_counts[nodes[i].max_layer]++;
    }
    printf("\nLayer distribution:\n");
    for (int l = 0; l < LCVDB_MAX_LAYERS; l++) {
        printf("  Layer %d: %d nodes\n", l, layer_counts[l]);
    }

    /* Now trace search for the same query as test_lcvdb.c */
    /* Regenerate the same query (rng state continues from inserts) */
    int8_t query[LCVDB_VEC_DIM];
    rand_vector(query);

    printf("\n=== BRUTE FORCE RANKING ===\n");
    int32_t all_scores[LCVDB_MAX_NODES];
    int sorted_ids[LCVDB_MAX_NODES];
    for (int i = 0; i < num_insert; i++) {
        all_scores[i] = ref_dot(query, vectors[i]);
        sorted_ids[i] = i;
    }
    /* Sort descending */
    for (int i = 0; i < num_insert - 1; i++) {
        for (int j = i + 1; j < num_insert; j++) {
            if (all_scores[sorted_ids[j]] > all_scores[sorted_ids[i]]) {
                int tmp = sorted_ids[i]; sorted_ids[i] = sorted_ids[j]; sorted_ids[j] = tmp;
            }
        }
    }
    for (int i = 0; i < 10; i++) {
        int id = sorted_ids[i];
        printf("  #%d: node=%d score=%d layer=%u ncnt=%u nbrs=[",
               i, id, all_scores[id], nodes[id].max_layer, nodes[id].neighbor_count);
        for (int j = 0; j < nodes[id].neighbor_count; j++)
            printf("%u%s", nodes[id].neighbors[j], j < nodes[id].neighbor_count - 1 ? "," : "");
        printf("]\n");
    }

    /* Manual search trace */
    printf("\n=== SEARCH TRACE ===\n");
    uint8_t ep = db.entry_point;
    printf("Start: entry_point=%u (score=%d)\n", ep, ref_dot(query, vectors[ep]));

    /* Phase 1: greedy descent */
    uint8_t current = ep;
    for (int layer = db.max_level; layer > 0; layer--) {
        int32_t best = ref_dot(query, vectors[current]);
        printf("\nLayer %d: start at node %u (score=%d)\n", layer, current, best);
        int improved = 1;
        while (improved) {
            improved = 0;
            for (int i = 0; i < nodes[current].neighbor_count; i++) {
                uint8_t nb = nodes[current].neighbors[i];
                int32_t s = ref_dot(query, vectors[nb]);
                printf("  eval neighbor %u: score=%d %s\n", nb, s, s > best ? "BETTER" : "");
                if (s > best) {
                    best = s;
                    current = nb;
                    improved = 1;
                }
            }
            if (improved) printf("  -> moved to node %u (score=%d)\n", current, best);
        }
        printf("  -> layer %d done, staying at node %u\n", layer, current);
    }

    /* Phase 2: beam search */
    printf("\nLayer 0 beam search: start at node %u\n", current);
    uint8_t visited[128] = {0};
    int32_t beam_scores[LCVDB_EF_SEARCH];
    uint8_t beam_ids[LCVDB_EF_SEARCH];
    int beam_count = 0;

    int32_t ep_score = ref_dot(query, vectors[current]);
    beam_scores[0] = ep_score;
    beam_ids[0] = current;
    beam_count = 1;
    visited[current / 8] |= (1 << (current % 8));

    int cursor = 0;
    int step = 0;
    while (cursor < beam_count) {
        uint8_t cand = beam_ids[cursor];
        printf("\n  Step %d: expand node %u (score=%d, beam_count=%d)\n",
               step++, cand, ref_dot(query, vectors[cand]), beam_count);

        for (int i = 0; i < nodes[cand].neighbor_count; i++) {
            uint8_t nb = nodes[cand].neighbors[i];
            if (visited[nb / 8] & (1 << (nb % 8))) {
                printf("    nbr %u: VISITED\n", nb);
                continue;
            }
            visited[nb / 8] |= (1 << (nb % 8));

            int32_t s = ref_dot(query, vectors[nb]);
            int inserted = 0;

            if (beam_count < LCVDB_EF_SEARCH) {
                int pos = beam_count;
                beam_scores[pos] = s;
                beam_ids[pos] = nb;
                while (pos > 0 && beam_scores[pos] > beam_scores[pos-1]) {
                    int32_t ts = beam_scores[pos]; beam_scores[pos] = beam_scores[pos-1]; beam_scores[pos-1] = ts;
                    uint8_t ti = beam_ids[pos]; beam_ids[pos] = beam_ids[pos-1]; beam_ids[pos-1] = ti;
                    pos--;
                }
                beam_count++;
                inserted = 1;
            } else if (s > beam_scores[beam_count - 1]) {
                int pos = beam_count - 1;
                beam_scores[pos] = s;
                beam_ids[pos] = nb;
                while (pos > 0 && beam_scores[pos] > beam_scores[pos-1]) {
                    int32_t ts = beam_scores[pos]; beam_scores[pos] = beam_scores[pos-1]; beam_scores[pos-1] = ts;
                    uint8_t ti = beam_ids[pos]; beam_ids[pos] = beam_ids[pos-1]; beam_ids[pos-1] = ti;
                    pos--;
                }
                inserted = 1;
            }

            printf("    nbr %u: score=%d %s\n", nb, s, inserted ? "INSERTED" : "rejected");
        }

        printf("    beam: [");
        for (int i = 0; i < beam_count; i++)
            printf("%u:%d%s", beam_ids[i], beam_scores[i], i < beam_count - 1 ? ", " : "");
        printf("]\n");

        cursor++;
    }

    printf("\n=== FINAL BEAM ===\n");
    for (int i = 0; i < beam_count; i++) {
        printf("  #%d: node=%u score=%d\n", i, beam_ids[i], beam_scores[i]);
    }

    /* Check: is node 48 reachable from entry point? */
    printf("\n=== REACHABILITY CHECK ===\n");
    int target_nodes[] = {48, 13, 0, 54, 35};
    for (int t = 0; t < 5; t++) {
        int target = target_nodes[t];
        /* BFS from entry point */
        uint8_t bfs_visited[256] = {0};
        uint8_t bfs_queue[256];
        int bfs_head = 0, bfs_tail = 0;
        bfs_queue[bfs_tail++] = ep;
        bfs_visited[ep] = 1;
        int found = 0;
        int hops = -1;
        uint8_t bfs_dist[256] = {0};

        while (bfs_head < bfs_tail && !found) {
            uint8_t n = bfs_queue[bfs_head++];
            for (int i = 0; i < nodes[n].neighbor_count; i++) {
                uint8_t nb = nodes[n].neighbors[i];
                if (!bfs_visited[nb]) {
                    bfs_visited[nb] = 1;
                    bfs_dist[nb] = bfs_dist[n] + 1;
                    bfs_queue[bfs_tail++] = nb;
                    if (nb == target) {
                        found = 1;
                        hops = bfs_dist[nb];
                        break;
                    }
                }
            }
        }

        int reachable_count = 0;
        for (int i = 0; i < num_insert; i++)
            if (bfs_visited[i]) reachable_count++;

        printf("  Node %d (score=%d): %s, hops=%d, total_reachable=%d/%d\n",
               target, all_scores[target],
               found ? "REACHABLE" : "UNREACHABLE", hops, reachable_count, num_insert);
    }

    free(node_buf);
    return 0;
}
