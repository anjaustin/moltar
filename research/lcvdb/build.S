// =============================================================================
// L-Cache VDB — HNSW Graph Build (Insert + Neighbor Selection)
// =============================================================================
// Insert a new vector into the HNSW graph with neighbor selection.
//
// Node layout (64 bytes):
//   [0..47]   int8 vector (48D)
//   [48..55]  neighbor IDs (uint8 x 8)
//   [56]      neighbor count
//   [57]      max_layer
//   [58..63]  reserved
// =============================================================================

    .arch armv8-a
    .text
    .align 4

    .equ NODE_SIZE,     64
    .equ VEC_OFF,       0
    .equ NBR_OFF,       48
    .equ NCNT_OFF,      56
    .equ MLYR_OFF,      57
    .equ M_MAX,         8
    .equ ML_FACTOR,     4      // 1/ln(M) ≈ 1/ln(8) ≈ 0.48, use integer approx

// =============================================================================
// lcvdb_assign_layer
// Assign a random layer to a new node using geometric distribution
// P(layer=l) = (1/M)^l, approximated with simple PRNG
//
// Parameters:
//   x0 = pointer to PRNG state (uint32)
//
// Returns:
//   w0 = assigned layer (0-3)
//
// Code size: ~40 bytes
// =============================================================================
    .global lcvdb_assign_layer
    .type lcvdb_assign_layer, %function
lcvdb_assign_layer:
    // xorshift32 PRNG
    ldr     w1, [x0]
    eor     w1, w1, w1, lsl #13
    eor     w1, w1, w1, lsr #17
    eor     w1, w1, w1, lsl #5
    str     w1, [x0]

    // Use top bits to assign layer
    // P(layer≥1) = 1/8, P(layer≥2) = 1/64, P(layer≥3) = 1/512
    mov     w0, #0
    tst     w1, #0x7               // bottom 3 bits == 0 → layer ≥ 1
    b.ne    .L_layer_done
    add     w0, w0, #1
    tst     w1, #0x38              // next 3 bits == 0 → layer ≥ 2
    b.ne    .L_layer_done
    add     w0, w0, #1
    tst     w1, #0x1C0             // next 3 bits == 0 → layer ≥ 3
    b.ne    .L_layer_done
    add     w0, w0, #1
.L_layer_done:
    ret
    .size lcvdb_assign_layer, . - lcvdb_assign_layer


// =============================================================================
// lcvdb_insert
// Insert a new vector into the HNSW graph
//
// Parameters:
//   x0 = pointer to lcvdb_t struct
//         [0]    uint16  node_count
//         [2]    uint8   entry_point_id
//         [3]    uint8   max_level
//         [4]    uint64  pointer to node array
//         [12]   uint64  pointer to upper layer table (unused for now)
//         [20]   uint8   M
//         [24]   uint32  prng_state
//   x1 = pointer to new vector (48 x int8)
//
// Returns:
//   w0 = new node ID (or 0xFF if full)
//
// Algorithm:
//   1. Assign layer to new node
//   2. Copy vector into node slot
//   3. If first node, just set as entry point
//   4. Greedy descent from entry to new node's layer
//   5. At each layer from new_layer down to 0:
//      a. Find M nearest neighbors via beam search
//      b. Connect new node to them (bidirectional)
//   6. If new_layer > max_level, update entry point
// =============================================================================
    .global lcvdb_insert
    .type lcvdb_insert, %function
lcvdb_insert:
    // Prologue
    stp     x29, x30, [sp, #-160]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8, d9, [sp, #96]

    mov     x19, x0                // x19 = db pointer
    mov     x20, x1                // x20 = new vector pointer

    // Get current node count
    ldrh    w21, [x19]             // w21 = node_count (= new node ID)
    cmp     w21, #255
    b.gt    .L_insert_full

    // Assign layer
    add     x0, x19, #28           // &prng_state
    bl      lcvdb_assign_layer
    mov     w22, w0                // w22 = new_layer

    // Get node array base
    ldr     x23, [x19, #8]        // x23 = node array

    // Calculate new node address
    ubfiz   x24, x21, #6, #8      // x24 = new_id * 64
    add     x24, x23, x24          // x24 = &nodes[new_id]

    // Copy vector into node (48 bytes)
    ld1     {v0.16b, v1.16b, v2.16b}, [x20]
    st1     {v0.16b, v1.16b, v2.16b}, [x24]

    // Initialize node metadata
    strb    wzr, [x24, #NCNT_OFF]  // neighbor_count = 0
    strb    w22, [x24, #MLYR_OFF]  // max_layer = new_layer
    // Zero out neighbor slots
    str     xzr, [x24, #NBR_OFF]

    // Increment node count
    add     w25, w21, #1
    strh    w25, [x19]

    // If this is the first node, set as entry point and done
    cbz     w21, .L_first_node

    // --- Find neighbors and connect ---
    // Preload query (new vector) into v0-v2
    // (already loaded above)

    // Get entry point
    ldrb    w26, [x19, #2]         // entry_point_id
    ldrb    w27, [x19, #3]         // current max_level (struct offsets match)

    // Phase 1: Greedy descent from entry to new_layer+1
    mov     w9, w27                // current layer = max_level
    mov     w20, w26               // current_node = entry_point

.L_insert_descent:
    cmp     w9, w22
    b.le    .L_insert_connect      // reached new_layer, start connecting

    // Greedy search at layer w9 (same as search.S greedy)
    ubfiz   x11, x20, #6, #8
    add     x11, x23, x11

    // Score current
    ld1     {v3.16b, v4.16b, v5.16b}, [x11]
    smull   v6.8h, v0.8b, v3.8b
    smull2  v7.8h, v0.16b, v3.16b
    smlal   v6.8h, v1.8b, v4.8b
    smlal2  v7.8h, v1.16b, v4.16b
    smlal   v6.8h, v2.8b, v5.8b
    smlal2  v7.8h, v2.16b, v5.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    addv    s9, v8.4s
    fmov    w12, s9                // best_score

.L_insert_greedy:
    add     x13, x11, #NBR_OFF
    ldrb    w14, [x11, #NCNT_OFF]
    mov     w15, #0
    mov     w16, #0                // improved flag

.L_insert_gnbr:
    cmp     w15, w14
    b.ge    .L_insert_gcheck

    ldrb    w17, [x13, x15]
    ubfiz   x11, x17, #6, #8
    add     x11, x23, x11

    ld1     {v3.16b, v4.16b, v5.16b}, [x11]
    smull   v6.8h, v0.8b, v3.8b
    smull2  v7.8h, v0.16b, v3.16b
    smlal   v6.8h, v1.8b, v4.8b
    smlal2  v7.8h, v1.16b, v4.16b
    smlal   v6.8h, v2.8b, v5.8b
    smlal2  v7.8h, v2.16b, v5.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    addv    s9, v8.4s
    fmov    w18, s9

    cmp     w18, w12
    b.le    .L_insert_gnext
    mov     w12, w18
    mov     w20, w17
    mov     w16, #1

.L_insert_gnext:
    add     w15, w15, #1
    b       .L_insert_gnbr

.L_insert_gcheck:
    cbnz    w16, .L_insert_grestart
    sub     w9, w9, #1
    b       .L_insert_descent

.L_insert_grestart:
    ubfiz   x11, x20, #6, #8
    add     x11, x23, x11
    b       .L_insert_greedy

    // Phase 2: Connect at each layer from new_layer down to 0
.L_insert_connect:
    // Simple neighbor selection: connect to the M closest nodes
    // found during a local search from current position
    
    // For each neighbor found, create bidirectional edge:
    //   new_node.neighbors[i] = neighbor_id
    //   neighbor.neighbors[j] = new_node_id (if not full)

    // Scan all existing nodes and find M closest (brute force for small N)
    // This is acceptable for N ≤ 256 (fits in L1)
    mov     w7, #0                 // candidate index
    // Use stack for top-M tracking
    // sp+112: scores[8] (int32 x 8)
    // sp+144: ids[8] (uint8 x 8)
    mov     w8, #0                 // found count
    mov     w28, #0x80000000       // INT32_MIN as worst score init

    // Initialize M worst scores
    mov     w9, #0
.L_init_topM:
    cmp     w9, #M_MAX
    b.ge    .L_scan_nodes
    str     w28, [sp, #112]
    add     w9, w9, #1
    b       .L_init_topM

.L_scan_nodes:
    cmp     w7, w21                // w21 = old node_count (before insert)
    b.ge    .L_connect_edges

    // Skip self
    cmp     w7, w21
    b.eq    .L_scan_next

    // Score node w7 against new vector (in v0-v2)
    ubfiz   x11, x7, #6, #8
    add     x11, x23, x11

    ld1     {v3.16b, v4.16b, v5.16b}, [x11]
    smull   v6.8h, v0.8b, v3.8b
    smull2  v7.8h, v0.16b, v3.16b
    smlal   v6.8h, v1.8b, v4.8b
    smlal2  v7.8h, v1.16b, v4.16b
    smlal   v6.8h, v2.8b, v5.8b
    smlal2  v7.8h, v2.16b, v5.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    addv    s9, v8.4s
    fmov    w18, s9                // w18 = score

    // Insert into top-M if better than worst
    cmp     w8, #M_MAX
    b.lt    .L_topM_insert         // not full yet, always insert

    // Find worst in top-M
    ldr     w9, [sp, #112]         // scores[0] (we keep sorted, worst last)
    // Actually, let's just do simple: find min in scores[0..M-1]
    mov     w10, #0                // worst_idx
    ldr     w11, [sp, #112]        // worst_score = scores[0]
    mov     w12, #1
.L_find_worst:
    cmp     w12, w8
    b.ge    .L_check_replace
    lsl     w13, w12, #2
    add     x14, sp, #112
    ldr     w15, [x14, x13]
    cmp     w15, w11
    b.ge    .L_fw_next
    mov     w11, w15
    mov     w10, w12
.L_fw_next:
    add     w12, w12, #1
    b       .L_find_worst

.L_check_replace:
    cmp     w18, w11
    b.le    .L_scan_next
    // Replace worst
    lsl     w13, w10, #2
    add     x14, sp, #112
    str     w18, [x14, x13]
    add     x14, sp, #144
    strb    w7, [x14, x10]
    b       .L_scan_next

.L_topM_insert:
    // Insert at position w8
    lsl     w13, w8, #2
    add     x14, sp, #112
    str     w18, [x14, x13]
    add     x14, sp, #144
    strb    w7, [x14, x8]
    add     w8, w8, #1
    b       .L_scan_next

.L_scan_next:
    add     w7, w7, #1
    b       .L_scan_nodes

.L_connect_edges:
    // w8 = number of neighbors found (≤ M_MAX)
    // Connect new node to its neighbors (bidirectional)

    // Set new node's neighbor list
    strb    w8, [x24, #NCNT_OFF]   // neighbor_count

    mov     w9, #0
.L_set_nbrs:
    cmp     w9, w8
    b.ge    .L_set_reverse

    add     x14, sp, #144
    ldrb    w10, [x14, x9]        // neighbor_id
    add     x15, x24, #NBR_OFF
    strb    w10, [x15, x9]        // new_node.neighbors[i] = neighbor_id

    add     w9, w9, #1
    b       .L_set_nbrs

.L_set_reverse:
    // Add reverse edges: for each neighbor, add new_node to their list.
    // If neighbor is full (M_MAX edges), replace its weakest edge
    // if new_node is closer to that neighbor than the weakest existing edge.
    // This prevents graph disconnection.
    mov     w9, #0
.L_rev_edge:
    cmp     w9, w8
    b.ge    .L_update_entry

    add     x14, sp, #144
    ldrb    w10, [x14, x9]        // neighbor_id

    // Get neighbor's node address
    ubfiz   x11, x10, #6, #8
    add     x11, x23, x11         // x11 = &nodes[neighbor_id]

    // Check if neighbor has room
    ldrb    w12, [x11, #NCNT_OFF]
    cmp     w12, #M_MAX
    b.ge    .L_rev_full            // neighbor full, try replace weakest

    // Has room -- just append
    add     x15, x11, #NBR_OFF
    strb    w21, [x15, x12]       // neighbor.neighbors[count] = new_id
    add     w12, w12, #1
    strb    w12, [x11, #NCNT_OFF]
    b       .L_rev_next

.L_rev_full:
    // Neighbor is full. Compute dot(neighbor_vec, new_node_vec).
    // If new_node is closer than neighbor's weakest existing edge, replace it.
    //
    // x11 = neighbor node addr, x24 = new node addr
    // v0-v2 = new node vector (already loaded as "query")
    //
    // dot(neighbor, new_node): neighbor vec at [x11], new_node vec in v0-v2
    ld1     {v3.16b, v4.16b, v5.16b}, [x11]   // neighbor vector
    smull   v6.8h, v0.8b, v3.8b
    smull2  v7.8h, v0.16b, v3.16b
    smlal   v6.8h, v1.8b, v4.8b
    smlal2  v7.8h, v1.16b, v4.16b
    smlal   v6.8h, v2.8b, v5.8b
    smlal2  v7.8h, v2.16b, v5.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    addv    s9, v8.4s
    fmov    w13, s9                // w13 = dot(neighbor, new_node)

    // Find weakest existing edge of this neighbor.
    // For each of M edges, compute dot(neighbor_vec, edge_target_vec).
    add     x15, x11, #NBR_OFF    // neighbor's neighbor list
    mov     w14, #0                // loop index
    mov     w16, #0                // worst_idx
    mov     w17, #0x7FFFFFFF       // worst_score = INT32_MAX

.L_rev_find_worst:
    cmp     w14, #M_MAX
    b.ge    .L_rev_check_replace

    ldrb    w18, [x15, x14]       // edge target id
    ubfiz   x6, x18, #6, #8
    add     x6, x23, x6           // edge target node addr

    // dot(neighbor_vec, edge_target_vec)
    // Reload neighbor vec from x11 each iteration (safe)
    ld1     {v3.16b, v4.16b, v5.16b}, [x11]    // neighbor vec
    ld1     {v13.16b, v14.16b, v15.16b}, [x6]   // edge target vec
    smull   v6.8h, v3.8b, v13.8b
    smull2  v7.8h, v3.16b, v13.16b
    smlal   v6.8h, v4.8b, v14.8b
    smlal2  v7.8h, v4.16b, v14.16b
    smlal   v6.8h, v5.8b, v15.8b
    smlal2  v7.8h, v5.16b, v15.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    addv    s9, v8.4s
    fmov    w6, s9                 // w6 = dot(neighbor, edge_target)

    cmp     w6, w17
    b.ge    .L_rev_fw_next
    mov     w17, w6                // new worst score
    mov     w16, w14               // new worst index

.L_rev_fw_next:
    add     w14, w14, #1
    b       .L_rev_find_worst

.L_rev_check_replace:
    // w13 = dot(neighbor, new_node)
    // w17 = dot(neighbor, weakest_existing_edge)
    // w16 = index of weakest edge
    cmp     w13, w17
    b.le    .L_rev_next            // new_node not closer than weakest, skip

    // Replace weakest edge with new_node
    strb    w21, [x15, w16, uxtw]  // neighbor.neighbors[worst_idx] = new_id

.L_rev_next:
    add     w9, w9, #1
    b       .L_rev_edge

.L_update_entry:
    // If new node's layer > max_level, update entry point
    ldrb    w9, [x19, #3]          // current max_level
    cmp     w22, w9
    b.le    .L_insert_ret

    strb    w22, [x19, #3]         // max_level = new_layer
    strb    w21, [x19, #2]         // entry_point = new_node_id

.L_insert_ret:
    mov     w0, w21                // return new node ID
    b       .L_insert_epilogue

.L_first_node:
    strb    w21, [x19, #2]         // entry_point = 0
    strb    w22, [x19, #3]         // max_level = new_layer
    mov     w0, w21
    b       .L_insert_epilogue

.L_insert_full:
    mov     w0, #0xFF              // error: full
    // fall through

.L_insert_epilogue:
    ldp     d8, d9, [sp, #96]
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #160
    ret
    .size lcvdb_insert, . - lcvdb_insert
