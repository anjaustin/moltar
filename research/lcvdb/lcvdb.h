/* ==========================================================================
 * L-Cache VDB — A Vector Database That Fits in L-Cache
 * ==========================================================================
 * HNSW search engine in NEON assembly.
 * 
 * Node layout: 64 bytes (1 cache line)
 *   [0..47]   int8 vector (48 dimensions)
 *   [48..55]  neighbor IDs (uint8 x 8, M=8)
 *   [56]      neighbor count (0-8)
 *   [57]      max_layer (0-3)
 *   [58..63]  reserved
 *
 * Memory budget (N=256):  16 KB code+data → fits in L1D
 * Memory budget (N=1024): 66 KB code+data → fits in L2
 * Memory budget (N=4096): 265 KB code+data → fits in L3
 * ========================================================================== */

#ifndef LCVDB_H
#define LCVDB_H

#include <stdint.h>

/* ---------- Configuration ---------- */
#define LCVDB_VEC_DIM       48      /* int8 dimensions per vector           */
#define LCVDB_NODE_SIZE     64      /* bytes per node (1 cache line)        */
#define LCVDB_M             8       /* max neighbors per node (layer 0)     */
#define LCVDB_M_UPPER       4       /* max neighbors per node (upper layers)*/
#define LCVDB_EF_SEARCH     8       /* beam width during search             */
#define LCVDB_MAX_LAYERS    4       /* maximum HNSW layers                  */
#define LCVDB_MAX_NODES     256     /* max nodes (uint8 IDs)                */

/* ---------- Node ---------- */
typedef struct __attribute__((aligned(64))) {
    int8_t   vector[LCVDB_VEC_DIM];     /* [0..47]  48D int8 vector         */
    uint8_t  neighbors[LCVDB_M];        /* [48..55] neighbor node IDs       */
    uint8_t  neighbor_count;            /* [56]     actual edge count       */
    uint8_t  max_layer;                 /* [57]     highest layer for node  */
    uint8_t  _reserved[6];              /* [58..63] pad to 64 bytes         */
} lcvdb_node_t;

_Static_assert(sizeof(lcvdb_node_t) == 64, "Node must be exactly 1 cache line");

/* ---------- Database ---------- 
 * AArch64 layout (pointers are 8 bytes):
 *   [0..1]   node_count (uint16)
 *   [2]      entry_point (uint8)
 *   [3]      max_level (uint8)
 *   [4..7]   padding (pointer alignment)
 *   [8..15]  node_array pointer
 *   [16..23] upper_layers pointer
 *   [24]     M
 *   [25]     M_upper
 *   [26..27] padding
 *   [28..31] prng_state
 *   [32..63] reserved (pad to 64 bytes)
 */
typedef struct __attribute__((aligned(64))) {
    uint16_t node_count;                /* [0..1]   current number of nodes */
    uint8_t  entry_point;               /* [2]      entry point node ID     */
    uint8_t  max_level;                 /* [3]      highest layer in graph  */
    uint32_t _pad0;                     /* [4..7]   alignment padding       */
    uint8_t  *node_array;               /* [8..15]  pointer to nodes        */
    uint8_t  *upper_layers;             /* [16..23] pointer to upper layers */
    uint8_t  M;                         /* [24]     neighbors per node      */
    uint8_t  M_upper;                   /* [25]     upper layer neighbors   */
    uint8_t  _pad1[2];                  /* [26..27] padding                 */
    uint32_t prng_state;                /* [28..31] xorshift32 state        */
    uint8_t  _reserved[32];             /* [32..63] pad to 64 bytes         */
} lcvdb_t;

_Static_assert(sizeof(lcvdb_t) == 64, "DB struct must be exactly 1 cache line");

/* ---------- API (implemented in NEON assembly) ---------- */

/* Initialize a new database. Caller provides aligned memory for nodes.
 * node_buf must be 64-byte aligned and at least LCVDB_MAX_NODES * 64 bytes.
 */
void lcvdb_init(lcvdb_t *db, void *node_buf);

/* Insert a vector. Returns new node ID, or 0xFF if full. */
uint8_t lcvdb_insert(lcvdb_t *db, const int8_t *vector);

/* Search for k nearest neighbors.
 * result_ids:    output array of k node IDs (uint8)
 * result_scores: output array of k dot-product scores (int32)
 */
void lcvdb_search(const lcvdb_t *db, const int8_t *query,
                  uint8_t k, uint8_t *result_ids, int32_t *result_scores);

/* Compute int8 dot product of two 48D vectors. */
int32_t lcvdb_dot_i8(const int8_t *a, const int8_t *b);

/* Preload query vector into NEON registers v0-v2. */
void lcvdb_load_query(const int8_t *query);

/* Dot product with query preloaded in v0-v2. */
int32_t lcvdb_dot_i8_preloaded(const int8_t *candidate);

/* Batch dot product: query (in v0-v2) vs 4 candidates.
 * Returns scores in w0-w3. */
void lcvdb_dot_i8_batch4(const int8_t *c0, const int8_t *c1,
                         const int8_t *c2, const int8_t *c3);

/* ---------- Memory Budget ---------- */
/*
 * Total memory for N nodes:
 *   Code:   1,964 bytes (verified by assembler)
 *   Nodes:  N * 64 bytes
 *   DB:     64 bytes
 *   Stack:  ~300 bytes (visited bitset + beam + saved regs)
 *
 *   N=64:    4,096 + 2,028 =   6.0 KB  → L1D (64 KB)
 *   N=128:   8,192 + 2,028 =  10.0 KB  → L1D
 *   N=256:  16,384 + 2,028 =  18.0 KB  → L1D
 *   N=512:  32,768 + 2,028 =  34.0 KB  → L1D
 *   N=1024: 65,536 + 2,028 =  66.0 KB  → L2 (256 KB)
 *   N=4096: 262,144 + 2,028 = 258.0 KB → L2/L3
 */

#endif /* LCVDB_H */
