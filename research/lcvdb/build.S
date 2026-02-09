// =============================================================================
// L-Cache VDB — HNSW Graph Build (Insert + Neighbor Selection)
// =============================================================================
// Insert a new vector into the HNSW graph with diverse neighbor selection.
//
// Node layout (64 bytes):
//   [0..47]   int8 vector (48D)
//   [48..55]  neighbor IDs (uint8 x 8)
//   [56]      neighbor count
//   [57]      max_layer
//   [58..63]  reserved
//
// Neighbor selection uses the HNSW "extended heuristic" (Algorithm 4):
//   1. Collect 2*M closest candidates via brute-force scan
//   2. Iterate candidates best-first
//   3. Keep candidate C only if dot(query, C) > dot(S, C) for ALL
//      already-selected neighbors S (ensures spatial diversity)
//
// No multiplication beyond SMULL/SMLAL in dot products. All selection
// decisions are pure integer comparisons.
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
    .equ CAND_MAX,      16         // 2*M candidates for diversity selection

// Stack frame layout (total 320 bytes, 16-byte aligned):
//   [sp+0..15]      saved x29, x30
//   [sp+16..31]     saved x19, x20
//   [sp+32..47]     saved x21, x22
//   [sp+48..63]     saved x23, x24
//   [sp+64..79]     saved x25, x26
//   [sp+80..95]     saved x27, x28
//   [sp+96..111]    saved d8, d9
//   --- candidate pool (from brute-force scan) ---
//   [sp+112..175]   cand_scores[16] (int32 x 16 = 64 bytes)
//   [sp+176..191]   cand_ids[16] (uint8 x 16)
//   --- selected neighbors (diversity-filtered) ---
//   [sp+192..223]   sel_scores[8] (int32 x 8 = 32 bytes)
//   [sp+224..231]   sel_ids[8] (uint8 x 8)
//   [sp+232..319]   padding/spare

    .equ FRAME_SIZE,    320
    .equ CAND_SCORES,   112
    .equ CAND_IDS,      176
    .equ SEL_SCORES,    192
    .equ SEL_IDS,       224

// =============================================================================
// lcvdb_assign_layer
// =============================================================================
    .global lcvdb_assign_layer
    .type lcvdb_assign_layer, %function
lcvdb_assign_layer:
    ldr     w1, [x0]
    eor     w1, w1, w1, lsl #13
    eor     w1, w1, w1, lsr #17
    eor     w1, w1, w1, lsl #5
    str     w1, [x0]

    mov     w0, #0
    tst     w1, #0x7
    b.ne    .L_layer_done
    add     w0, w0, #1
    tst     w1, #0x38
    b.ne    .L_layer_done
    add     w0, w0, #1
    tst     w1, #0x1C0
    b.ne    .L_layer_done
    add     w0, w0, #1
.L_layer_done:
    ret
    .size lcvdb_assign_layer, . - lcvdb_assign_layer


// =============================================================================
// lcvdb_insert — Insert with diverse neighbor selection
// =============================================================================
    .global lcvdb_insert
    .type lcvdb_insert, %function
lcvdb_insert:
    // Prologue
    stp     x29, x30, [sp, #-FRAME_SIZE]!
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
    add     x0, x19, #28
    bl      lcvdb_assign_layer
    mov     w22, w0                // w22 = new_layer

    // Get node array base
    ldr     x23, [x19, #8]        // x23 = node array

    // Calculate new node address: new_id << 6
    ubfiz   x24, x21, #6, #8
    add     x24, x23, x24          // x24 = &nodes[new_id]

    // Copy vector into node (48 bytes)
    ld1     {v0.16b, v1.16b, v2.16b}, [x20]
    st1     {v0.16b, v1.16b, v2.16b}, [x24]

    // Initialize node metadata
    strb    wzr, [x24, #NCNT_OFF]
    strb    w22, [x24, #MLYR_OFF]
    str     xzr, [x24, #NBR_OFF]

    // Increment node count
    add     w25, w21, #1
    strh    w25, [x19]

    // First node? Just set as entry point.
    cbz     w21, .L_first_node

    // v0-v2 = new node vector (query for neighbor finding)
    // Get entry point
    ldrb    w26, [x19, #2]         // entry_point_id
    ldrb    w27, [x19, #3]         // current max_level

    // =====================================================================
    // Phase 1: Greedy descent from entry to new_layer
    // =====================================================================
    mov     w9, w27                // current layer
    mov     w20, w26               // current_node = entry_point

.L_insert_descent:
    cmp     w9, w22
    b.le    .L_insert_collect      // reached target layer

    ubfiz   x11, x20, #6, #8
    add     x11, x23, x11

    // Score current node (widen per-chunk to avoid int16 overflow)
    ld1     {v3.16b, v4.16b, v5.16b}, [x11]
    smull   v6.8h, v0.8b, v3.8b
    smull2  v7.8h, v0.16b, v3.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v1.8b, v4.8b
    smull2  v7.8h, v1.16b, v4.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v2.8b, v5.8b
    smull2  v7.8h, v2.16b, v5.16b
    sadalp  v8.4s, v6.8h
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
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v1.8b, v4.8b
    smull2  v7.8h, v1.16b, v4.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v2.8b, v5.8b
    smull2  v7.8h, v2.16b, v5.16b
    sadalp  v8.4s, v6.8h
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

    // =====================================================================
    // Phase 2: Collect top-2M (16) candidates via brute-force scan
    // =====================================================================
.L_insert_collect:
    mov     w7, #0                 // scan index
    mov     w8, #0                 // cand_count
    mov     w28, #0x80000000       // INT32_MIN

    // Initialize all 16 candidate scores to INT32_MIN
    mov     w9, #0
.L_init_cand:
    cmp     w9, #CAND_MAX
    b.ge    .L_scan_nodes
    lsl     w10, w9, #2
    add     x14, sp, #CAND_SCORES
    str     w28, [x14, x10]
    add     w9, w9, #1
    b       .L_init_cand

.L_scan_nodes:
    cmp     w7, w21                // w21 = node_count before insert
    b.ge    .L_diversity_select

    // Score node w7 against new vector (in v0-v2)
    ubfiz   x11, x7, #6, #8
    add     x11, x23, x11

    ld1     {v3.16b, v4.16b, v5.16b}, [x11]
    smull   v6.8h, v0.8b, v3.8b
    smull2  v7.8h, v0.16b, v3.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v1.8b, v4.8b
    smull2  v7.8h, v1.16b, v4.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v2.8b, v5.8b
    smull2  v7.8h, v2.16b, v5.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    addv    s9, v8.4s
    fmov    w18, s9                // w18 = dot(query, node[w7])

    // Insert into top-CAND_MAX pool
    cmp     w8, #CAND_MAX
    b.lt    .L_cand_insert

    // Pool full — find worst and maybe replace
    mov     w10, #0                // worst_idx
    add     x14, sp, #CAND_SCORES
    ldr     w11, [x14]             // worst_score = scores[0]
    mov     w12, #1
.L_cand_find_worst:
    cmp     w12, w8
    b.ge    .L_cand_check_replace
    lsl     w13, w12, #2
    ldr     w15, [x14, x13]
    cmp     w15, w11
    b.ge    .L_cand_fw_next
    mov     w11, w15
    mov     w10, w12
.L_cand_fw_next:
    add     w12, w12, #1
    b       .L_cand_find_worst

.L_cand_check_replace:
    cmp     w18, w11
    b.le    .L_scan_next
    lsl     w13, w10, #2
    add     x14, sp, #CAND_SCORES
    str     w18, [x14, x13]
    add     x14, sp, #CAND_IDS
    strb    w7, [x14, x10]
    b       .L_scan_next

.L_cand_insert:
    lsl     w13, w8, #2
    add     x14, sp, #CAND_SCORES
    str     w18, [x14, x13]
    add     x14, sp, #CAND_IDS
    strb    w7, [x14, x8]
    add     w8, w8, #1

.L_scan_next:
    add     w7, w7, #1
    b       .L_scan_nodes

    // =====================================================================
    // Phase 3: Diversity selection (HNSW extended heuristic)
    //
    // Sort candidates by score (descending), then iterate.
    // Keep candidate C if: for ALL already-selected neighbors S,
    //   dot(query, C) > dot(S, C)
    // i.e., C is closer to the query than to any selected neighbor.
    //
    // This uses only dot products (SMULL/SMLAL) and comparisons.
    // =====================================================================
.L_diversity_select:
    // w8 = total candidates collected
    mov     w25, w8                // w25 = cand_count (preserve)

    // --- Sort candidates descending by score (insertion sort) ---
    // Small N (<=16), simple is fine.
    mov     w9, #1                 // i = 1
.L_sort_outer:
    cmp     w9, w25
    b.ge    .L_sort_done

    // key_score = cand_scores[i], key_id = cand_ids[i]
    lsl     w10, w9, #2
    add     x14, sp, #CAND_SCORES
    ldr     w11, [x14, x10]       // key_score
    add     x15, sp, #CAND_IDS
    ldrb    w12, [x15, x9]        // key_id

    mov     w13, w9                // j = i
.L_sort_inner:
    cbz     w13, .L_sort_place
    sub     w14, w13, #1           // j-1
    lsl     w15, w14, #2
    add     x16, sp, #CAND_SCORES
    ldr     w17, [x16, x15]       // cand_scores[j-1]
    cmp     w17, w11
    b.ge    .L_sort_place          // scores[j-1] >= key, stop (descending)

    // Shift scores[j-1] -> scores[j]
    lsl     w18, w13, #2
    str     w17, [x16, x18]
    // Shift ids[j-1] -> ids[j]
    add     x16, sp, #CAND_IDS
    ldrb    w17, [x16, x14]
    strb    w17, [x16, x13]

    mov     w13, w14               // j = j-1
    b       .L_sort_inner

.L_sort_place:
    lsl     w14, w13, #2
    add     x16, sp, #CAND_SCORES
    str     w11, [x16, x14]       // cand_scores[j] = key_score
    add     x16, sp, #CAND_IDS
    strb    w12, [x16, x13]       // cand_ids[j] = key_id

    add     w9, w9, #1
    b       .L_sort_outer

.L_sort_done:
    // Candidates are now sorted descending by dot(query, candidate).
    // Now iterate and apply diversity filter.

    mov     w26, #0                // sel_count = 0
    mov     w9, #0                 // candidate index

.L_div_loop:
    cmp     w9, w25                // done with all candidates?
    b.ge    .L_div_done
    cmp     w26, #M_MAX            // selected M neighbors already?
    b.ge    .L_div_done

    // Get candidate C: score and id
    lsl     w10, w9, #2
    add     x14, sp, #CAND_SCORES
    ldr     w27, [x14, x10]       // w27 = dot(query, C)
    add     x14, sp, #CAND_IDS
    ldrb    w28, [x14, x9]        // w28 = C's node id

    // Load C's vector for diversity checks
    ubfiz   x11, x28, #6, #8
    add     x11, x23, x11         // x11 = &nodes[C]

    // Check against each already-selected neighbor S:
    //   If dot(S, C) >= dot(query, C) for ANY S, reject C.
    mov     w14, #0                // s_idx
    mov     w15, #0                // reject flag

.L_div_check:
    cmp     w14, w26               // checked all selected?
    b.ge    .L_div_accept_or_reject
    cbnz    w15, .L_div_accept_or_reject  // already rejected

    // Get selected neighbor S's node id
    add     x16, sp, #SEL_IDS
    ldrb    w17, [x16, x14]       // S's node id

    // Compute dot(S, C)
    ubfiz   x6, x17, #6, #8
    add     x6, x23, x6           // &nodes[S]

    // Load S vector and C vector, compute dot (overflow-safe)
    ld1     {v3.16b, v4.16b, v5.16b}, [x6]    // S vector
    ld1     {v13.16b, v14.16b, v15.16b}, [x11] // C vector
    smull   v6.8h, v3.8b, v13.8b
    smull2  v7.8h, v3.16b, v13.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v4.8b, v14.8b
    smull2  v7.8h, v4.16b, v14.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v5.8b, v15.8b
    smull2  v7.8h, v5.16b, v15.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    addv    s9, v8.4s
    fmov    w18, s9                // w18 = dot(S, C)

    // Diversity check: reject if dot(S, C) >= dot(query, C)
    cmp     w18, w27
    b.lt    .L_div_check_next
    mov     w15, #1                // reject

.L_div_check_next:
    add     w14, w14, #1
    b       .L_div_check

.L_div_accept_or_reject:
    cbnz    w15, .L_div_skip       // rejected

    // Accept: add C to selected list
    lsl     w14, w26, #2
    add     x16, sp, #SEL_SCORES
    str     w27, [x16, x14]       // sel_scores[sel_count] = dot(query, C)
    add     x16, sp, #SEL_IDS
    strb    w28, [x16, x26]       // sel_ids[sel_count] = C's id
    add     w26, w26, #1

.L_div_skip:
    add     w9, w9, #1
    b       .L_div_loop

.L_div_done:
    // w26 = number of selected diverse neighbors
    // sel_ids[0..w26-1] = the neighbor IDs
    mov     w8, w26                // w8 = final neighbor count

    // =====================================================================
    // Phase 4: Connect edges (same as before)
    // =====================================================================

    // Set new node's neighbor list
    strb    w8, [x24, #NCNT_OFF]

    mov     w9, #0
.L_set_nbrs:
    cmp     w9, w8
    b.ge    .L_set_reverse

    add     x14, sp, #SEL_IDS
    ldrb    w10, [x14, x9]        // neighbor_id
    add     x15, x24, #NBR_OFF
    strb    w10, [x15, x9]

    add     w9, w9, #1
    b       .L_set_nbrs

.L_set_reverse:
    // Reverse edges with replacement when full
    mov     w9, #0
.L_rev_edge:
    cmp     w9, w8
    b.ge    .L_update_entry

    add     x14, sp, #SEL_IDS
    ldrb    w10, [x14, x9]        // neighbor_id

    ubfiz   x11, x10, #6, #8
    add     x11, x23, x11         // x11 = &nodes[neighbor_id]

    ldrb    w12, [x11, #NCNT_OFF]
    cmp     w12, #M_MAX
    b.ge    .L_rev_full

    // Has room -- append
    add     x15, x11, #NBR_OFF
    strb    w21, [x15, x12]
    add     w12, w12, #1
    strb    w12, [x11, #NCNT_OFF]
    b       .L_rev_next

.L_rev_full:
    // Neighbor full. dot(neighbor, new_node) vs weakest existing edge.
    ld1     {v3.16b, v4.16b, v5.16b}, [x11]   // neighbor vector
    smull   v6.8h, v0.8b, v3.8b
    smull2  v7.8h, v0.16b, v3.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v1.8b, v4.8b
    smull2  v7.8h, v1.16b, v4.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v2.8b, v5.8b
    smull2  v7.8h, v2.16b, v5.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    addv    s9, v8.4s
    fmov    w13, s9                // w13 = dot(neighbor, new_node)

    add     x15, x11, #NBR_OFF
    mov     w14, #0
    mov     w16, #0                // worst_idx
    mov     w17, #0x7FFFFFFF       // worst_score

.L_rev_find_worst:
    cmp     w14, #M_MAX
    b.ge    .L_rev_check_replace

    ldrb    w18, [x15, x14]
    ubfiz   x6, x18, #6, #8
    add     x6, x23, x6

    ld1     {v3.16b, v4.16b, v5.16b}, [x11]
    ld1     {v13.16b, v14.16b, v15.16b}, [x6]
    smull   v6.8h, v3.8b, v13.8b
    smull2  v7.8h, v3.16b, v13.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v4.8b, v14.8b
    smull2  v7.8h, v4.16b, v14.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    smull   v6.8h, v5.8b, v15.8b
    smull2  v7.8h, v5.16b, v15.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h
    addv    s9, v8.4s
    fmov    w6, s9

    cmp     w6, w17
    b.ge    .L_rev_fw_next
    mov     w17, w6
    mov     w16, w14

.L_rev_fw_next:
    add     w14, w14, #1
    b       .L_rev_find_worst

.L_rev_check_replace:
    cmp     w13, w17
    b.le    .L_rev_next
    strb    w21, [x15, w16, uxtw]

.L_rev_next:
    add     w9, w9, #1
    b       .L_rev_edge

.L_update_entry:
    ldrb    w9, [x19, #3]
    cmp     w22, w9
    b.le    .L_insert_ret

    strb    w22, [x19, #3]
    strb    w21, [x19, #2]

.L_insert_ret:
    mov     w0, w21
    b       .L_insert_epilogue

.L_first_node:
    strb    w21, [x19, #2]
    strb    w22, [x19, #3]
    mov     w0, w21
    b       .L_insert_epilogue

.L_insert_full:
    mov     w0, #0xFF

.L_insert_epilogue:
    ldp     d8, d9, [sp, #96]
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #FRAME_SIZE
    ret
    .size lcvdb_insert, . - lcvdb_insert
