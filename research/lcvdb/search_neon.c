/* L-Cache VDB — NEON-optimized HNSW search (split storage, uint16 IDs)
 *
 * Uses inline assembly for dot products to eliminate function call overhead.
 * Query is preloaded into v0-v2 once and stays there for the entire search.
 * Each dot product is ~12 instructions inlined, vs ~20 instructions for bl+ret.
 *
 * Compiled with -ffixed-v0 -ffixed-v1 -ffixed-v2 to prevent the compiler
 * from clobbering the preloaded query.
 */
#include <string.h>
#include <stdint.h>
#include "lcvdb.h"

/* Inline NEON dot product: query in v0-v2 (preserved), candidate via pointer.
 * Equivalent to lcvdb_dot_i8_preloaded but zero call overhead. */
static inline int32_t dot_preloaded(const int8_t *cand) {
    int32_t result;
    __asm__ __volatile__ (
        /* Load candidate vector (48 bytes = 3 x 16) */
        "ld1    {v3.16b, v4.16b, v5.16b}, [%1]\n\t"
        /* Chunk 0: widen per-chunk to avoid int16 overflow */
        "smull  v6.8h, v0.8b, v3.8b\n\t"
        "smull2 v7.8h, v0.16b, v3.16b\n\t"
        "saddlp v8.4s, v6.8h\n\t"
        "sadalp v8.4s, v7.8h\n\t"
        /* Chunk 1 */
        "smull  v6.8h, v1.8b, v4.8b\n\t"
        "smull2 v7.8h, v1.16b, v4.16b\n\t"
        "sadalp v8.4s, v6.8h\n\t"
        "sadalp v8.4s, v7.8h\n\t"
        /* Chunk 2 */
        "smull  v6.8h, v2.8b, v5.8b\n\t"
        "smull2 v7.8h, v2.16b, v5.16b\n\t"
        "sadalp v8.4s, v6.8h\n\t"
        "sadalp v8.4s, v7.8h\n\t"
        /* Horizontal reduction */
        "addv   s9, v8.4s\n\t"
        "fmov   %w0, s9\n\t"
        : "=r" (result)
        : "r" (cand)
        : "v3", "v4", "v5", "v6", "v7", "v8", "v9", "memory"
    );
    return result;
}

/* Load query into v0-v2. Must be called once before dot_preloaded calls. */
static inline void load_query(const int8_t *query) {
    __asm__ __volatile__ (
        "ld1    {v0.16b, v1.16b, v2.16b}, [%0]\n\t"
        : /* no outputs */
        : "r" (query)
        : "v0", "v1", "v2", "memory"
    );
}

int lcvdb_search(const lcvdb_t *db, const int8_t *query,
                 int k, uint16_t *result_ids, int32_t *result_scores) {

    if (db->node_count == 0 || db->entry_point == LCVDB_INVALID_ID)
        return 0;

    lcvdb_topo_t *topo = db->topo_array;
    lcvdb_vec_t  *vecs = db->vec_array;

    /* Visited bitset — only clear bytes needed */
    uint8_t visited[8192];
    memset(visited, 0, (db->node_count + 7) / 8);

    #define VIS_SET(id)  (visited[(id) >> 3] |= (1 << ((id) & 7)))
    #define VIS_TEST(id) (visited[(id) >> 3] & (1 << ((id) & 7)))

    /* Two separate lists: candidates (expansion frontier) + results (top-ef) */
    #define MAX_CAND 256
    int32_t cand_scores[MAX_CAND];
    uint16_t cand_ids[MAX_CAND];
    int cand_count = 0;
    int cand_cursor = 0;

    int32_t res_scores[LCVDB_EF_SEARCH];
    uint16_t res_ids[LCVDB_EF_SEARCH];
    int res_count = 0;

    uint16_t ep = db->entry_point;

    /* Preload query into NEON registers v0-v2.
     * With -ffixed-v0/v1/v2, the compiler won't touch these. */
    load_query(query);

    /* Phase 1: Greedy descent through upper layers */
    uint16_t current = ep;
    for (int layer = db->max_level; layer > 0; layer--) {
        int32_t best = dot_preloaded(vecs[current].vector);
        int improved = 1;
        while (improved) {
            improved = 0;
            for (int i = 0; i < topo[current].neighbor_count; i++) {
                uint16_t nb = topo[current].neighbors[i];
                if (topo[nb].flags & LCVDB_FLAG_DELETED)
                    continue;
                int32_t s = dot_preloaded(vecs[nb].vector);
                if (s > best) {
                    best = s;
                    current = nb;
                    improved = 1;
                }
            }
        }
    }

    /* Phase 2: Beam search on layer 0 */
    int32_t ep_score = dot_preloaded(vecs[current].vector);
    VIS_SET(current);

    /* Seed both lists (skip deleted from results) */
    if (!(topo[current].flags & LCVDB_FLAG_DELETED)) {
        cand_scores[0] = ep_score;
        cand_ids[0] = current;
        cand_count = 1;
        res_scores[0] = ep_score;
        res_ids[0] = current;
        res_count = 1;
    } else {
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

        /* Expand: evaluate all neighbors of this candidate.
         * Inline dot product — no function call overhead. */
        for (int i = 0; i < topo[cand].neighbor_count; i++) {
            uint16_t nb = topo[cand].neighbors[i];

            if (VIS_TEST(nb))
                continue;
            VIS_SET(nb);

            /* Inline NEON dot product — query in v0-v2, candidate loaded */
            int32_t s = dot_preloaded(vecs[nb].vector);

            /* Add to candidate pool for further expansion */
            if (cand_count < MAX_CAND) {
                cand_scores[cand_count] = s;
                cand_ids[cand_count] = nb;
                cand_count++;
            }

            /* Skip deleted nodes for results */
            if (topo[nb].flags & LCVDB_FLAG_DELETED)
                continue;

            /* Insert into results (sorted descending by score) */
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
