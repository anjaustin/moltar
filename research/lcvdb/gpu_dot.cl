/* L-Cache VDB — OpenCL int8 dot product kernel for PowerVR BXM-8-256
 *
 * Each work item computes dot(query, nodes[gid]) for one node.
 * 48D int8 vectors, result is int32.
 *
 * PowerVR BXM-8-256: 128-wide SIMD, preferred int vec width = 16.
 * char16 = 16 x int8 = one SIMD-width load.
 * 48D = 3 x char16 loads per vector.
 */

__kernel void dot_batch(
    __global const char *query,       /* 48 bytes: the query vector */
    __global const char *nodes,       /* N * 64 bytes: node array (48B vec + 16B metadata) */
    __global int *scores,             /* N ints: output dot products */
    const int node_count              /* number of nodes */
) {
    int gid = get_global_id(0);
    if (gid >= node_count) return;

    /* Load query — 3 x 16 bytes = 48 dimensions */
    char16 q0 = vload16(0, query);
    char16 q1 = vload16(1, query);
    char16 q2 = vload16(2, query);

    /* Load node vector — stride is 64 bytes per node */
    __global const char *node_vec = nodes + gid * 64;
    char16 n0 = vload16(0, node_vec);
    char16 n1 = vload16(1, node_vec);
    char16 n2 = vload16(2, node_vec);

    /* Widening multiply: int8 * int8 -> int16, then accumulate to int32.
     * convert does sign-extension: char16 -> short16 -> int16.
     * We accumulate in int32 to avoid overflow (max per lane = 3 * 127^2 = 48387).
     */

    /* Chunk 0: dims 0-15 */
    short16 p0 = convert_short16(q0) * convert_short16(n0);
    /* Chunk 1: dims 16-31 */
    short16 p1 = convert_short16(q1) * convert_short16(n1);
    /* Chunk 2: dims 32-47 */
    short16 p2 = convert_short16(q2) * convert_short16(n2);

    /* Horizontal sum: short16 -> int
     * Widen to int and sum. Each short16 has 16 values.
     * Do pairwise: convert to int16 (16 x int32), then reduce.
     */
    int16 i0 = convert_int16(p0);
    int16 i1 = convert_int16(p1);
    int16 i2 = convert_int16(p2);

    /* Sum all three int16 vectors */
    int16 sum = i0 + i1 + i2;

    /* Horizontal reduction of int16 (16 x int32) -> single int32 */
    int8 h8 = sum.lo + sum.hi;
    int4 h4 = h8.lo + h8.hi;
    int2 h2 = h4.lo + h4.hi;
    int result = h2.x + h2.y;

    scores[gid] = result;
}

/* Variant: query in local memory for repeated use */
__kernel void dot_batch_local(
    __global const char *query,
    __global const char *nodes,
    __global int *scores,
    const int node_count
) {
    __local char lq[48];

    /* First 48 work items cooperatively load query into local mem */
    int lid = get_local_id(0);
    if (lid < 48)
        lq[lid] = query[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    int gid = get_global_id(0);
    if (gid >= node_count) return;

    char16 q0 = vload16(0, lq);
    char16 q1 = vload16(1, lq);
    char16 q2 = vload16(2, lq);

    __global const char *node_vec = nodes + gid * 64;
    char16 n0 = vload16(0, node_vec);
    char16 n1 = vload16(1, node_vec);
    char16 n2 = vload16(2, node_vec);

    short16 p0 = convert_short16(q0) * convert_short16(n0);
    short16 p1 = convert_short16(q1) * convert_short16(n1);
    short16 p2 = convert_short16(q2) * convert_short16(n2);

    int16 sum = convert_int16(p0) + convert_int16(p1) + convert_int16(p2);

    int8 h8 = sum.lo + sum.hi;
    int4 h4 = h8.lo + h8.hi;
    int2 h2 = h4.lo + h4.hi;
    scores[gid] = h2.x + h2.y;
}
