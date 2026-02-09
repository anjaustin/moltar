/* L-Cache VDB — C reference initialization (split storage) */
#include <string.h>
#include "lcvdb.h"

void lcvdb_init(lcvdb_t *db, void *topo_buf, void *vec_buf, uint32_t max_nodes) {
    memset(db, 0, sizeof(lcvdb_t));
    db->M = LCVDB_M;
    db->entry_point = LCVDB_INVALID_ID;
    db->topo_array = (lcvdb_topo_t *)topo_buf;
    db->vec_array = (lcvdb_vec_t *)vec_buf;
    db->max_nodes = max_nodes;
    db->prng_state = 0x12345678;

    /* Zero topology and vector arrays */
    memset(topo_buf, 0, max_nodes * sizeof(lcvdb_topo_t));
    memset(vec_buf, 0, max_nodes * sizeof(lcvdb_vec_t));
}
