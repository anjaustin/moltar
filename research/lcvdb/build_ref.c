/* L-Cache VDB — C reference insert with HNSW beam search + diversity heuristic (split storage)
 *
 * Proper HNSW insert (Algorithms 1+4 from Malkov & Yashunin 2018):
 *   - Topology array (lcvdb_topo_t, 64 bytes each): graph edges, metadata
 *   - Vector array (lcvdb_vec_t, 64 bytes each): embeddings
 *   - uint16 IDs (max 65534 nodes, 0xFFFF = invalid)
 *
 * Insert algorithm:
 *   1. Assign random HNSW layer via geometric distribution
 *   2. Copy vector into vec_array[new_id]
 *   3. Initialize topo_array[new_id] (neighbors, layer, payload)
 *   4. Greedy descent from entry point to target layer
 *   5. Beam search (ef_construction width) on layer 0 to find candidates
 *   6. Diversity selection (HNSW extended heuristic, Algorithm 4)
 *   7. Connect forward edges (new -> selected neighbors)
 *   8. Connect reverse edges (selected neighbors -> new), with pruning if full
 *   9. Update entry point if new node has higher layer
 *
 * Key fix over previous version: Phase 5 uses graph beam search (O(log N))
 * instead of brute-force O(N) scan. This produces higher-quality edges and
 * maintains graph connectivity at large N (was losing recall at N>=512).
 */
#include <string.h>
#include <stdint.h>
#include "lcvdb.h"

/* PRNG: xorshift32 */
static uint8_t lcvdb_assign_layer_ref(uint32_t *prng) {
    uint32_t s = *prng;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *prng = s;

    /* Geometric distribution: layer 0 with prob 7/8, etc. */
    uint8_t layer = 0;
    if ((s & 0x7) == 0) {
        layer++;
        if ((s & 0x38) == 0) {
            layer++;
            if ((s & 0x1C0) == 0) {
                layer++;
            }
        }
    }
    if (layer >= LCVDB_MAX_LAYERS)
        layer = LCVDB_MAX_LAYERS - 1;
    return layer;
}

/* Use NEON dot product from distance.S (same as search path) */
#define dot_i8 lcvdb_dot_i8

uint16_t lcvdb_insert(lcvdb_t *db, const int8_t *vector, uint32_t payload_id) {
    uint32_t new_id = db->node_count;
    if (new_id >= db->max_nodes)
        return LCVDB_INVALID_ID;

    lcvdb_topo_t *topo = db->topo_array;
    lcvdb_vec_t  *vecs = db->vec_array;

    /* Assign layer */
    uint8_t new_layer = lcvdb_assign_layer_ref(&db->prng_state);

    /* Copy vector into vec slot */
    memcpy(vecs[new_id].vector, vector, LCVDB_VEC_DIM);

    /* Initialize topology node */
    memset(&topo[new_id], 0, sizeof(lcvdb_topo_t));
    topo[new_id].max_layer = new_layer;
    topo[new_id].payload_id = payload_id;

    /* Increment node count */
    db->node_count = new_id + 1;

    /* First node — just set as entry point */
    if (new_id == 0) {
        db->entry_point = 0;
        db->max_level = new_layer;
        return 0;
    }

    uint16_t current = db->entry_point;

    /* Phase 1: Greedy descent through upper layers */
    for (int layer = (int)db->max_level; layer > (int)new_layer && layer > 0; layer--) {
        int32_t best = dot_i8(vector, vecs[current].vector);
        int improved = 1;
        while (improved) {
            improved = 0;
            for (int i = 0; i < topo[current].neighbor_count; i++) {
                uint16_t nb = topo[current].neighbors[i];
                int32_t s = dot_i8(vector, vecs[nb].vector);
                if (s > best) {
                    best = s;
                    current = nb;
                    improved = 1;
                }
            }
        }
    }

    /* Phase 2: Beam search on layer 0 (ef_construction width)
     *
     * Same algorithm as search_ref.c but with ef_construction beam width.
     * Produces top-ef_construction candidates sorted by similarity.
     * This is O(ef_construction * M * log N) vs the old O(N) brute force.
     *
     * Two lists:
     *   cand_*: expansion frontier (pick best unprocessed each step)
     *   res_*:  top-ef_construction results (sorted descending by score)
     */
    #define EFC LCVDB_EF_CONSTRUCT
    #define BEAM_MAX 256  /* max expansion candidates */

    /* Visited bitset — stack-allocated, supports up to max_nodes */
    uint8_t visited[(LCVDB_MAX_NODES + 8) / 8];
    memset(visited, 0, (new_id + 7) / 8);

    #define VIS_SET(id)  (visited[(id) >> 3] |= (1 << ((id) & 7)))
    #define VIS_TEST(id) (visited[(id) >> 3] & (1 << ((id) & 7)))

    int32_t  cand_scores[BEAM_MAX];
    uint16_t cand_ids[BEAM_MAX];
    int cand_count = 0;
    int cand_cursor = 0;

    int32_t  res_scores[EFC];
    uint16_t res_ids[EFC];
    int res_count = 0;

    /* Seed with greedy descent result */
    int32_t ep_score = dot_i8(vector, vecs[current].vector);
    VIS_SET(current);

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
            uint16_t ti = cand_ids[cand_cursor];
            cand_ids[cand_cursor] = cand_ids[best_idx];
            cand_ids[best_idx] = ti;
        }

        /* Early termination: best candidate worse than worst result */
        if (res_count >= EFC && best_score <= res_scores[res_count - 1])
            break;

        uint16_t cand = cand_ids[cand_cursor];
        cand_cursor++;

        /* Expand: evaluate all neighbors of this candidate */
        for (int i = 0; i < topo[cand].neighbor_count; i++) {
            uint16_t nb = topo[cand].neighbors[i];

            if (VIS_TEST(nb))
                continue;
            VIS_SET(nb);

            int32_t s = dot_i8(vector, vecs[nb].vector);

            /* Add to candidate pool for further expansion */
            if (cand_count < BEAM_MAX) {
                cand_scores[cand_count] = s;
                cand_ids[cand_count] = nb;
                cand_count++;
            }

            /* Insert into results (sorted descending by score) */
            if (res_count < EFC) {
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

    #undef VIS_SET
    #undef VIS_TEST

    /* Phase 3: Diversity selection (HNSW extended heuristic, Algorithm 4)
     * Keep candidate C only if dot(query, C) > dot(S, C) for ALL
     * already-selected neighbors S.
     * res_scores/res_ids are already sorted descending — iterate in order. */
    uint16_t sel_ids[LCVDB_M];
    int sel_count = 0;

    for (int i = 0; i < res_count && sel_count < LCVDB_M; i++) {
        uint16_t cid = res_ids[i];
        int32_t cscore = res_scores[i]; /* dot(query, C) */
        int reject = 0;

        for (int s = 0; s < sel_count; s++) {
            int32_t sc = dot_i8(vecs[sel_ids[s]].vector, vecs[cid].vector);
            if (sc >= cscore) {
                reject = 1;
                break;
            }
        }

        if (!reject) {
            sel_ids[sel_count] = cid;
            sel_count++;
        }
    }

    /* Phase 4: Connect forward edges (new node -> selected neighbors) */
    topo[new_id].neighbor_count = (uint8_t)sel_count;
    for (int i = 0; i < sel_count; i++) {
        topo[new_id].neighbors[i] = sel_ids[i];
    }

    /* Phase 5: Connect reverse edges (each selected neighbor -> new node)
     * If neighbor is full (M edges), shrink back to M using diversity heuristic.
     * With M=16, saturation happens later, reducing orphan risk. */
    for (int i = 0; i < sel_count; i++) {
        uint16_t nb = sel_ids[i];
        int nc = topo[nb].neighbor_count;

        if (nc < LCVDB_M) {
            /* Has room — append */
            topo[nb].neighbors[nc] = (uint16_t)new_id;
            topo[nb].neighbor_count = nc + 1;
        } else {
            /* Full — collect existing edges + new, re-select M via diversity.
             * Pool size = M + 1 (existing M edges + the new node). */
            int32_t pool_scores[LCVDB_M + 1];
            uint16_t pool_ids[LCVDB_M + 1];
            int pool_count = 0;

            for (int j = 0; j < LCVDB_M; j++) {
                uint16_t edge = topo[nb].neighbors[j];
                pool_scores[pool_count] = dot_i8(vecs[nb].vector, vecs[edge].vector);
                pool_ids[pool_count] = edge;
                pool_count++;
            }
            pool_scores[pool_count] = dot_i8(vecs[nb].vector, vecs[new_id].vector);
            pool_ids[pool_count] = (uint16_t)new_id;
            pool_count++;

            /* Sort descending by score */
            for (int a = 1; a < pool_count; a++) {
                int32_t ks = pool_scores[a];
                uint16_t ki = pool_ids[a];
                int b = a;
                while (b > 0 && pool_scores[b - 1] < ks) {
                    pool_scores[b] = pool_scores[b - 1];
                    pool_ids[b] = pool_ids[b - 1];
                    b--;
                }
                pool_scores[b] = ks;
                pool_ids[b] = ki;
            }

            /* Diversity-select M from pool (center = nb's vector) */
            uint16_t new_edges[LCVDB_M];
            int new_nc = 0;

            for (int p = 0; p < pool_count && new_nc < LCVDB_M; p++) {
                uint16_t pid = pool_ids[p];
                int32_t ps = pool_scores[p];
                int rej = 0;

                for (int q = 0; q < new_nc; q++) {
                    int32_t qs = dot_i8(vecs[new_edges[q]].vector, vecs[pid].vector);
                    if (qs >= ps) {
                        rej = 1;
                        break;
                    }
                }

                if (!rej)
                    new_edges[new_nc++] = pid;
            }

            /* Backfill if diversity was too aggressive */
            if (new_nc < LCVDB_M) {
                for (int p = 0; p < pool_count && new_nc < LCVDB_M; p++) {
                    int dup = 0;
                    for (int q = 0; q < new_nc; q++) {
                        if (new_edges[q] == pool_ids[p]) { dup = 1; break; }
                    }
                    if (!dup)
                        new_edges[new_nc++] = pool_ids[p];
                }
            }

            topo[nb].neighbor_count = (uint8_t)new_nc;
            for (int j = 0; j < new_nc; j++)
                topo[nb].neighbors[j] = new_edges[j];
        }
    }

    /* Phase 6: Update entry point if new node has higher layer */
    if (new_layer > db->max_level) {
        db->max_level = new_layer;
        db->entry_point = (uint16_t)new_id;
    }

    return (uint16_t)new_id;
}

void lcvdb_delete(lcvdb_t *db, uint16_t node_id) {
    if (node_id < db->node_count) {
        db->topo_array[node_id].flags |= LCVDB_FLAG_DELETED;
    }
}
