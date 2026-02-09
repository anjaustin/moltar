/* L-Cache VDB — HNSW search (split storage, uint16 IDs)
 *
 * C control flow with NEON dot products via distance.S.
 * Uses separate candidate and result lists (standard two-list approach).
 * Skips deleted nodes (tombstone flag in topology).
 * Returns actual number of results found.
 */
#include <string.h>
#include <stdint.h>
#include "lcvdb.h"

/* Use NEON dot product from distance.S */
#define dot_i8_ref lcvdb_dot_i8

int lcvdb_search(const lcvdb_t *db, const int8_t *query,
                 int k, uint16_t *result_ids, int32_t *result_scores) {

    if (db->node_count == 0 || db->entry_point == LCVDB_INVALID_ID)
        return 0;

    lcvdb_topo_t *topo = db->topo_array;
    lcvdb_vec_t  *vecs = db->vec_array;

    /* Visited bitset — supports up to 65536 nodes (8 KB on stack) */
    uint8_t visited[8192];
    memset(visited, 0, (db->node_count + 7) / 8);

    #define VIS_SET(id)  (visited[(id) >> 3] |= (1 << ((id) & 7)))
    #define VIS_TEST(id) (visited[(id) >> 3] & (1 << ((id) & 7)))

    /* Two separate lists (standard HNSW approach):
     *   candidates: nodes to expand, unordered, pick best each step
     *   results:    top-ef results sorted descending by score
     */
    #define MAX_CAND 256
    int32_t cand_scores[MAX_CAND];
    uint16_t cand_ids[MAX_CAND];
    int cand_count = 0;
    int cand_cursor = 0;

    int32_t res_scores[LCVDB_EF_SEARCH];
    uint16_t res_ids[LCVDB_EF_SEARCH];
    int res_count = 0;

    uint16_t ep = db->entry_point;

    /* Phase 1: Greedy descent through upper layers */
    uint16_t current = ep;
    for (int layer = db->max_level; layer > 0; layer--) {
        int32_t best = dot_i8_ref(query, vecs[current].vector);
        int improved = 1;
        while (improved) {
            improved = 0;
            for (int i = 0; i < topo[current].neighbor_count; i++) {
                uint16_t nb = topo[current].neighbors[i];
                if (topo[nb].flags & LCVDB_FLAG_DELETED)
                    continue;
                int32_t s = dot_i8_ref(query, vecs[nb].vector);
                if (s > best) {
                    best = s;
                    current = nb;
                    improved = 1;
                }
            }
        }
    }

    /* Phase 2: Beam search on layer 0 */
    int32_t ep_score = dot_i8_ref(query, vecs[current].vector);
    VIS_SET(current);

    /* Seed both lists (skip if entry is deleted) */
    if (!(topo[current].flags & LCVDB_FLAG_DELETED)) {
        cand_scores[0] = ep_score;
        cand_ids[0] = current;
        cand_count = 1;

        res_scores[0] = ep_score;
        res_ids[0] = current;
        res_count = 1;
    } else {
        /* Still add to candidates for expansion even if deleted */
        cand_scores[0] = ep_score;
        cand_ids[0] = current;
        cand_count = 1;
    }

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
            uint16_t ti = cand_ids[cand_cursor];
            cand_ids[cand_cursor] = cand_ids[best_idx];
            cand_ids[best_idx] = ti;
        }

        /* Early termination: best candidate worse than worst result */
        if (res_count >= LCVDB_EF_SEARCH && best_score <= res_scores[res_count - 1])
            break;

        uint16_t cand = cand_ids[cand_cursor];
        cand_cursor++;

        /* Expand: evaluate all neighbors */
        for (int i = 0; i < topo[cand].neighbor_count; i++) {
            uint16_t nb = topo[cand].neighbors[i];

            if (VIS_TEST(nb))
                continue;
            VIS_SET(nb);

            int32_t s = dot_i8_ref(query, vecs[nb].vector);

            /* Add ALL unvisited neighbors to candidate pool */
            if (cand_count < MAX_CAND) {
                cand_scores[cand_count] = s;
                cand_ids[cand_count] = nb;
                cand_count++;
            }

            /* Skip deleted nodes for results */
            if (topo[nb].flags & LCVDB_FLAG_DELETED)
                continue;

            /* Insert into results (sorted descending) */
            if (res_count < LCVDB_EF_SEARCH) {
                int pos = res_count;
                res_scores[pos] = s;
                res_ids[pos] = nb;
                while (pos > 0 && res_scores[pos] > res_scores[pos - 1]) {
                    int32_t t = res_scores[pos];
                    res_scores[pos] = res_scores[pos - 1];
                    res_scores[pos - 1] = t;
                    uint16_t u = res_ids[pos];
                    res_ids[pos] = res_ids[pos - 1];
                    res_ids[pos - 1] = u;
                    pos--;
                }
                res_count++;
            } else if (s > res_scores[res_count - 1]) {
                int pos = res_count - 1;
                res_scores[pos] = s;
                res_ids[pos] = nb;
                while (pos > 0 && res_scores[pos] > res_scores[pos - 1]) {
                    int32_t t = res_scores[pos];
                    res_scores[pos] = res_scores[pos - 1];
                    res_scores[pos - 1] = t;
                    uint16_t u = res_ids[pos];
                    res_ids[pos] = res_ids[pos - 1];
                    res_ids[pos - 1] = u;
                    pos--;
                }
            }
        }
    }

    /* Copy results */
    int out_count = 0;
    for (int i = 0; i < k && i < res_count; i++) {
        result_ids[i] = res_ids[i];
        result_scores[i] = res_scores[i];
        out_count++;
    }

    return out_count;
}
