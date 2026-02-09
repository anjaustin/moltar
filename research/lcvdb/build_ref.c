/* L-Cache VDB — C reference insert with HNSW diversity heuristic (split storage)
 *
 * Direct C port of the assembly build.S, adapted for split storage layout:
 *   - Topology array (lcvdb_topo_t, 32 bytes each): graph edges, metadata
 *   - Vector array (lcvdb_vec_t, 64 bytes each): embeddings
 *   - uint16 IDs (max 65534 nodes, 0xFFFF = invalid)
 *
 * Insert algorithm:
 *   1. Assign random HNSW layer via geometric distribution
 *   2. Copy vector into vec_array[new_id]
 *   3. Initialize topo_array[new_id] (neighbors, layer, payload)
 *   4. Greedy descent from entry point to target layer
 *   5. Brute-force collect top-2M candidates at layer 0
 *   6. Diversity selection (HNSW extended heuristic, Algorithm 4)
 *   7. Connect forward edges (new -> selected neighbors)
 *   8. Connect reverse edges (selected neighbors -> new), with replacement if full
 *   9. Update entry point if new node has higher layer
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

/* Scalar dot product (used during build; search uses NEON from distance.S) */
static int32_t dot_i8(const int8_t *a, const int8_t *b) {
    int32_t sum = 0;
    for (int i = 0; i < LCVDB_VEC_DIM; i++)
        sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

#define CAND_MAX 16  /* 2*M candidates for diversity selection */

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

    /* Phase 2: Brute-force collect top-2M candidates at layer 0 */
    int32_t cand_scores[CAND_MAX];
    uint16_t cand_ids[CAND_MAX];
    int cand_count = 0;

    for (uint32_t i = 0; i < new_id; i++) {
        int32_t s = dot_i8(vector, vecs[i].vector);

        if (cand_count < CAND_MAX) {
            cand_scores[cand_count] = s;
            cand_ids[cand_count] = (uint16_t)i;
            cand_count++;
        } else {
            /* Find worst in pool */
            int worst_idx = 0;
            int32_t worst_score = cand_scores[0];
            for (int j = 1; j < cand_count; j++) {
                if (cand_scores[j] < worst_score) {
                    worst_score = cand_scores[j];
                    worst_idx = j;
                }
            }
            if (s > worst_score) {
                cand_scores[worst_idx] = s;
                cand_ids[worst_idx] = (uint16_t)i;
            }
        }
    }

    /* Phase 3: Sort candidates descending by score (insertion sort, small N) */
    for (int i = 1; i < cand_count; i++) {
        int32_t key_score = cand_scores[i];
        uint16_t key_id = cand_ids[i];
        int j = i;
        while (j > 0 && cand_scores[j - 1] < key_score) {
            cand_scores[j] = cand_scores[j - 1];
            cand_ids[j] = cand_ids[j - 1];
            j--;
        }
        cand_scores[j] = key_score;
        cand_ids[j] = key_id;
    }

    /* Phase 4: Diversity selection (HNSW extended heuristic)
     * Keep candidate C only if dot(query, C) > dot(S, C) for ALL
     * already-selected neighbors S. */
    int32_t sel_scores[LCVDB_M];
    uint16_t sel_ids[LCVDB_M];
    int sel_count = 0;

    for (int i = 0; i < cand_count && sel_count < LCVDB_M; i++) {
        uint16_t cid = cand_ids[i];
        int32_t cscore = cand_scores[i]; /* dot(query, C) */
        int reject = 0;

        for (int s = 0; s < sel_count; s++) {
            int32_t sc = dot_i8(vecs[sel_ids[s]].vector, vecs[cid].vector);
            if (sc >= cscore) {
                reject = 1;
                break;
            }
        }

        if (!reject) {
            sel_scores[sel_count] = cscore;
            sel_ids[sel_count] = cid;
            sel_count++;
        }
    }

    /* Phase 5: Connect forward edges (new node -> selected neighbors) */
    topo[new_id].neighbor_count = (uint8_t)sel_count;
    for (int i = 0; i < sel_count; i++) {
        topo[new_id].neighbors[i] = sel_ids[i];
    }

    /* Phase 6: Connect reverse edges (each selected neighbor -> new node) */
    for (int i = 0; i < sel_count; i++) {
        uint16_t nb = sel_ids[i];
        int nc = topo[nb].neighbor_count;

        if (nc < LCVDB_M) {
            /* Has room — append */
            topo[nb].neighbors[nc] = (uint16_t)new_id;
            topo[nb].neighbor_count = nc + 1;
        } else {
            /* Full — replace weakest edge if new node is stronger */
            int32_t new_score = dot_i8(vecs[nb].vector, vecs[new_id].vector);

            /* Find weakest existing edge of neighbor nb */
            int worst_idx = 0;
            int32_t worst_score = 0x7FFFFFFF;
            for (int j = 0; j < LCVDB_M; j++) {
                uint16_t edge = topo[nb].neighbors[j];
                int32_t es = dot_i8(vecs[nb].vector, vecs[edge].vector);
                if (es < worst_score) {
                    worst_score = es;
                    worst_idx = j;
                }
            }

            if (new_score > worst_score) {
                topo[nb].neighbors[worst_idx] = (uint16_t)new_id;
            }
        }
    }

    /* Phase 7: Update entry point if new node has higher layer */
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
