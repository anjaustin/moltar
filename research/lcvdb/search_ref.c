/* HNSW search — C control flow, NEON dot products via distance.S
 * Uses separate candidate and result lists (standard two-list approach).
 */
#include <string.h>
#include <stdint.h>
#include "lcvdb.h"

/* Use NEON dot product from distance.S instead of scalar C loop */
#define dot_i8_ref lcvdb_dot_i8

void lcvdb_search(const lcvdb_t *db, const int8_t *query,
                  uint8_t k, uint8_t *result_ids, int32_t *result_scores) {
    lcvdb_node_t *nodes = (lcvdb_node_t *)db->node_array;
    uint8_t visited[128];
    memset(visited, 0, sizeof(visited));

    /* Two separate lists (standard HNSW approach):
     *   candidates: nodes to expand, unordered, pick best each step
     *   results:    top-ef results sorted descending by score
     */
    #define MAX_CAND 256
    int32_t cand_scores[MAX_CAND];
    uint8_t cand_ids[MAX_CAND];
    int cand_count = 0;
    int cand_cursor = 0;

    int32_t res_scores[LCVDB_EF_SEARCH];
    uint8_t res_ids[LCVDB_EF_SEARCH];
    int res_count = 0;

    uint8_t ep = db->entry_point;
    uint8_t max_level = db->max_level;

    /* Phase 1: Greedy descent through upper layers */
    uint8_t current = ep;
    for (int layer = max_level; layer > 0; layer--) {
        int32_t best = dot_i8_ref(query, nodes[current].vector);
        int improved = 1;
        while (improved) {
            improved = 0;
            for (int i = 0; i < nodes[current].neighbor_count; i++) {
                uint8_t nb = nodes[current].neighbors[i];
                int32_t s = dot_i8_ref(query, nodes[nb].vector);
                if (s > best) {
                    best = s;
                    current = nb;
                    improved = 1;
                }
            }
        }
    }

    /* Phase 2: Beam search on layer 0 */
    int32_t ep_score = dot_i8_ref(query, nodes[current].vector);
    visited[current / 8] |= (1 << (current % 8));

    /* Seed both lists */
    cand_scores[0] = ep_score;
    cand_ids[0] = current;
    cand_count = 1;

    res_scores[0] = ep_score;
    res_ids[0] = current;
    res_count = 1;

    while (cand_cursor < cand_count) {
        /* Pick best unprocessed candidate */
        int best_idx = cand_cursor;
        int32_t best_score = cand_scores[cand_cursor];
        for (int i = cand_cursor + 1; i < cand_count; i++) {
            if (cand_scores[i] > best_score) {
                best_score = cand_scores[i];
                best_idx = i;
            }
        }

        /* Swap best to cursor position */
        if (best_idx != cand_cursor) {
            int32_t ts = cand_scores[cand_cursor];
            cand_scores[cand_cursor] = cand_scores[best_idx];
            cand_scores[best_idx] = ts;
            uint8_t ti = cand_ids[cand_cursor];
            cand_ids[cand_cursor] = cand_ids[best_idx];
            cand_ids[best_idx] = ti;
        }

        /* Early termination: best candidate worse than worst result */
        if (res_count >= LCVDB_EF_SEARCH && best_score <= res_scores[res_count - 1])
            break;

        uint8_t cand = cand_ids[cand_cursor];
        cand_cursor++;

        /* Expand: evaluate all neighbors */
        for (int i = 0; i < nodes[cand].neighbor_count; i++) {
            uint8_t nb = nodes[cand].neighbors[i];

            if (visited[nb / 8] & (1 << (nb % 8)))
                continue;
            visited[nb / 8] |= (1 << (nb % 8));

            int32_t s = dot_i8_ref(query, nodes[nb].vector);

            /* Add ALL unvisited neighbors to candidate pool.
             * The result list handles filtering — candidates just
             * need to be available for expansion. */
            if (cand_count < MAX_CAND) {
                cand_scores[cand_count] = s;
                cand_ids[cand_count] = nb;
                cand_count++;
            }

            /* Insert into results (sorted descending) */
            if (res_count < LCVDB_EF_SEARCH) {
                int pos = res_count;
                res_scores[pos] = s;
                res_ids[pos] = nb;
                while (pos > 0 && res_scores[pos] > res_scores[pos-1]) {
                    int32_t t = res_scores[pos]; res_scores[pos] = res_scores[pos-1]; res_scores[pos-1] = t;
                    uint8_t u = res_ids[pos]; res_ids[pos] = res_ids[pos-1]; res_ids[pos-1] = u;
                    pos--;
                }
                res_count++;
            } else if (s > res_scores[res_count - 1]) {
                int pos = res_count - 1;
                res_scores[pos] = s;
                res_ids[pos] = nb;
                while (pos > 0 && res_scores[pos] > res_scores[pos-1]) {
                    int32_t t = res_scores[pos]; res_scores[pos] = res_scores[pos-1]; res_scores[pos-1] = t;
                    uint8_t u = res_ids[pos]; res_ids[pos] = res_ids[pos-1]; res_ids[pos-1] = u;
                    pos--;
                }
            }
        }
    }

    /* Copy results */
    for (int i = 0; i < k && i < res_count; i++) {
        result_ids[i] = res_ids[i];
        result_scores[i] = res_scores[i];
    }
}
