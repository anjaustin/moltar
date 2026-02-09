/* ==========================================================================
 * L-Cache VDB — A Vector Database That Fits in L-Cache
 * ==========================================================================
 * HNSW search engine with split storage layout.
 *
 * Split storage: topology and vectors in separate arrays.
 *   - Topology array (graph structure) stays cache-hot during traversal
 *   - Vector array touched only for distance computation
 *   - Payload ID maps to user's external data
 *
 * Topology node: 64 bytes (1 cache line, M=16)
 *   [0..31]   neighbor IDs (uint16 x 16)
 *   [32]      neighbor count (uint8)
 *   [33]      max_layer (uint8)
 *   [34..35]  flags (uint16: bit 0 = deleted)
 *   [36..39]  payload_id (uint32: external ID)
 *   [40..63]  reserved (24 bytes)
 *
 * Vector slot: 64 bytes (1 cache line)
 *   [0..47]   int8 vector (48D)
 *   [48..63]  padding
 *
 * Memory budget (N=256):
 *   Topology: 64 * 256  =  16 KB → L1D
 *   Vectors:  64 * 256  =  16 KB → L1D
 *   Total:     32 KB + code + stack → L1D (64 KB)
 *
 * Memory budget (N=1024):
 *   Topology: 64 * 1024 =  64 KB → L2
 *   Vectors:  64 * 1024 =  64 KB → L2
 *   Total:    128 KB → L2 (256 KB)
 *
 * Memory budget (N=4096):
 *   Topology: 64 * 4096 = 256 KB → L2/L3
 *   Vectors:  64 * 4096 = 256 KB → L2/L3
 *   Total:    512 KB + code + stack
 * ========================================================================== */

#ifndef LCVDB_H
#define LCVDB_H

#include <stdint.h>

/* ---------- Configuration ---------- */
#define LCVDB_VEC_DIM       48      /* int8 dimensions per vector           */
#define LCVDB_VEC_SLOT      64      /* bytes per vector slot (padded)       */
#define LCVDB_TOPO_SIZE     64      /* bytes per topology node (1 cache line)*/
#define LCVDB_M             16      /* max neighbors per node (layer 0)     */
#define LCVDB_M_UPPER       8       /* max neighbors per node (upper layers)*/
#define LCVDB_EF_SEARCH     64      /* beam width during search             */
#define LCVDB_EF_CONSTRUCT  64      /* beam width during insert             */
#define LCVDB_MAX_LAYERS    4       /* maximum HNSW layers                  */
#define LCVDB_MAX_NODES     65535   /* max nodes (uint16 IDs, 0xFFFF=invalid)*/

/* ---------- Node flags ---------- */
#define LCVDB_FLAG_DELETED  0x0001

/* ---------- Invalid ID sentinel ---------- */
#define LCVDB_INVALID_ID    0xFFFF

/* ---------- Topology Node ----------
 * 64 bytes = 1 cache line. M=16 neighbors × 2 bytes = 32 bytes for IDs.
 *   [0..31]   neighbor IDs (uint16 x 16)
 *   [32]      neighbor count (uint8)
 *   [33]      max_layer (uint8)
 *   [34..35]  flags (uint16: bit 0 = deleted)
 *   [36..39]  payload_id (uint32: external ID)
 *   [40..63]  reserved (24 bytes)
 */
typedef struct __attribute__((aligned(64))) {
    uint16_t neighbors[LCVDB_M];        /* [0..31]  neighbor node IDs       */
    uint8_t  neighbor_count;            /* [32]     actual edge count       */
    uint8_t  max_layer;                 /* [33]     highest layer for node  */
    uint16_t flags;                     /* [34..35] bit flags               */
    uint32_t payload_id;                /* [36..39] external payload ID     */
    uint8_t  _reserved[24];            /* [40..63] pad to 64 bytes         */
} lcvdb_topo_t;

_Static_assert(sizeof(lcvdb_topo_t) == 64, "Topo node must be 1 cache line (64 bytes)");

/* ---------- Vector Slot ---------- */
typedef struct __attribute__((aligned(64))) {
    int8_t   vector[LCVDB_VEC_DIM];     /* [0..47]  48D int8 vector         */
    uint8_t  _padding[16];              /* [48..63] pad to cache line       */
} lcvdb_vec_t;

_Static_assert(sizeof(lcvdb_vec_t) == 64, "Vec slot must be 1 cache line");

/* ---------- Legacy Node (for old tests) ---------- */
#define LCVDB_NODE_SIZE     64
typedef struct __attribute__((aligned(64))) {
    int8_t   vector[LCVDB_VEC_DIM];
    uint8_t  neighbors[LCVDB_M];
    uint8_t  neighbor_count;
    uint8_t  max_layer;
    uint8_t  _reserved[6];
} lcvdb_node_t;

/* ---------- Database ----------
 * AArch64 layout (pointers are 8 bytes):
 *   [0..3]   node_count (uint32)
 *   [4..5]   entry_point (uint16)
 *   [6]      max_level (uint8)
 *   [7]      M (uint8)
 *   [8..15]  topo_array pointer
 *   [16..23] vec_array pointer
 *   [24..27] max_nodes (uint32)
 *   [28..31] prng_state (uint32)
 *   [32..63] reserved
 */
typedef struct __attribute__((aligned(64))) {
    uint32_t node_count;                /* [0..3]   current number of nodes */
    uint16_t entry_point;               /* [4..5]   entry point node ID     */
    uint8_t  max_level;                 /* [6]      highest layer in graph  */
    uint8_t  M;                         /* [7]      neighbors per node      */
    lcvdb_topo_t *topo_array;           /* [8..15]  topology nodes          */
    lcvdb_vec_t  *vec_array;            /* [16..23] vector slots            */
    uint32_t max_nodes;                 /* [24..27] capacity                */
    uint32_t prng_state;                /* [28..31] xorshift32 state        */
    uint8_t  _reserved[32];             /* [32..63] pad to 64 bytes         */
} lcvdb_t;

_Static_assert(sizeof(lcvdb_t) == 64, "DB struct must be exactly 1 cache line");

/* ---------- API ---------- */

/* Initialize a new database. Caller provides aligned memory.
 *   topo_buf: 64-byte aligned, at least max_nodes * 64 bytes
 *   vec_buf:  64-byte aligned, at least max_nodes * 64 bytes
 *   max_nodes: capacity (up to 65534; 65535 is reserved as invalid)
 */
void lcvdb_init(lcvdb_t *db, void *topo_buf, void *vec_buf, uint32_t max_nodes);

/* Insert a vector with optional payload. Returns new node ID, or 0xFFFF if full. */
uint16_t lcvdb_insert(lcvdb_t *db, const int8_t *vector, uint32_t payload_id);

/* Delete a node (tombstone). Node is skipped in search, slot not reused yet. */
void lcvdb_delete(lcvdb_t *db, uint16_t node_id);

/* Search for k nearest neighbors (skips deleted nodes).
 *   result_ids:    output array of k node IDs (uint16)
 *   result_scores: output array of k dot-product scores (int32)
 *   Returns actual number of results found.
 */
int lcvdb_search(const lcvdb_t *db, const int8_t *query,
                 int k, uint16_t *result_ids, int32_t *result_scores);

/* Compute int8 dot product of two 48D vectors. */
int32_t lcvdb_dot_i8(const int8_t *a, const int8_t *b);

/* Preload query vector into NEON registers v0-v2. */
void lcvdb_load_query(const int8_t *query);

/* Dot product with query preloaded in v0-v2. */
int32_t lcvdb_dot_i8_preloaded(const int8_t *candidate);

/* Batch dot product: query (in v0-v2) vs 4 candidates. */
void lcvdb_dot_i8_batch4(const int8_t *c0, const int8_t *c1,
                         const int8_t *c2, const int8_t *c3);

/* ---------- Memory Budget ---------- */
/*
 * Split storage: Topology (64B each) + Vectors (64B each) = 128B/node
 *
 *   N=256:    Topo=16KB + Vec=16KB  =  32 KB → L1D (64 KB)
 *   N=512:    Topo=32KB + Vec=32KB  =  64 KB → L1D/L2
 *   N=1024:   Topo=64KB + Vec=64KB  = 128 KB → L2 (256 KB)
 *   N=4096:   Topo=256KB + Vec=256KB = 512 KB → L2/L3
 *   N=16384:  Topo=1MB + Vec=1MB    =   2 MB → L3/DRAM
 *   N=65535:  Topo=4MB + Vec=4MB    =   8 MB → DRAM
 *
 * During search, only topology is traversed continuously.
 * Vectors loaded on-demand for distance computation.
 * Effective hot working set = topology only.
 */

#endif /* LCVDB_H */
