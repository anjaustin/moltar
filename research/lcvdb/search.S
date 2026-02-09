// =============================================================================
// L-Cache VDB — HNSW Search Core
// =============================================================================
// Complete HNSW search: greedy descent through upper layers + beam search
// on layer 0. All in AArch64 NEON assembly.
//
// Node layout (64 bytes = 1 cache line):
//   [0..47]   int8 vector (48D)
//   [48..55]  neighbor IDs (uint8 x 8, M=8)
//   [56]      neighbor count (0-8)
//   [57]      max_layer (0-3)
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
    .equ MAX_LAYERS,    4
    .equ EF_SEARCH,     8
    .equ MAX_K,         8

// Stack frame layout (total 512 bytes, 16-byte aligned):
//   [sp+0..15]     saved x29, x30
//   [sp+16..31]    saved x19, x20
//   [sp+32..47]    saved x21, x22
//   [sp+48..63]    saved x23, x24
//   [sp+64..79]    saved x25, x26
//   [sp+80..95]    saved x27, x28
//   [sp+96..111]   saved d8, d9
//   [sp+112..127]  saved d10, d11
//   [sp+128..255]  visited bitset (128 bytes, supports N≤1024)
//   [sp+256..287]  beam_scores (int32 x 8 = 32 bytes)
//   [sp+288..295]  beam_ids (uint8 x 8 = 8 bytes)
//   [sp+296]       beam_count (uint8)
//   [sp+297..511]  padding/spare

    .equ FRAME_SIZE,    512
    .equ VISITED_OFF,   128
    .equ BEAM_SCORES,   256
    .equ BEAM_IDS,      288
    .equ BEAM_COUNT,    296

// =============================================================================
// lcvdb_search
// Parameters:
//   x0 = pointer to lcvdb_t struct
//   x1 = pointer to query vector (48 x int8)
//   x2 = k (number of results wanted, ≤ 8)
//   x3 = pointer to result_ids output (uint8 x k)
//   x4 = pointer to result_scores output (int32 x k)
// =============================================================================
    .global lcvdb_search
    .type lcvdb_search, %function
lcvdb_search:
    // Prologue
    sub     sp, sp, #FRAME_SIZE
    stp     x29, x30, [sp]
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]

    // Parse lcvdb_t struct
    ldrh    w24, [x0]              // node_count
    ldrb    w27, [x0, #2]          // entry_point_id
    ldrb    w28, [x0, #3]          // max_level
    ldr     x19, [x0, #8]          // node array base

    // Save output pointers and k
    mov     x21, x2                // k
    mov     x22, x3                // result_ids
    mov     x23, x4                // result_scores

    // Zero visited bitset (128 bytes at sp+VISITED_OFF)
    add     x25, sp, #VISITED_OFF
    stp     xzr, xzr, [x25]
    stp     xzr, xzr, [x25, #16]
    stp     xzr, xzr, [x25, #32]
    stp     xzr, xzr, [x25, #48]
    stp     xzr, xzr, [x25, #64]
    stp     xzr, xzr, [x25, #80]
    stp     xzr, xzr, [x25, #96]
    stp     xzr, xzr, [x25, #112]

    // Beam base pointer
    add     x26, sp, #BEAM_SCORES

    // Preload query vector into v0-v2
    ld1     {v0.16b, v1.16b, v2.16b}, [x1]

    // Current position = entry point
    mov     w20, w27

    // =========================================================================
    // PHASE 1: Greedy descent through upper layers
    // =========================================================================
    cbz     w28, .L_beam_search

    mov     w9, w28                // w9 = current_layer

.L_greedy_layer:
    cmp     w9, #0
    b.le    .L_beam_search

    // Calculate current node address
    ubfiz   x11, x20, #6, #8
    add     x11, x19, x11

    // Score current node
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
    fmov    w12, s9                // w12 = best_score

.L_greedy_inner:
    add     x13, x11, #NBR_OFF
    ldrb    w14, [x11, #NCNT_OFF]
    mov     w15, #0
    mov     w16, #0                // improved flag

.L_greedy_nbr:
    cmp     w15, w14
    b.ge    .L_greedy_check

    ldrb    w17, [x13, x15]

    // Score neighbor
    ubfiz   x11, x17, #6, #8
    add     x11, x19, x11

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
    b.le    .L_greedy_next

    mov     w12, w18
    mov     w20, w17
    mov     w16, #1

.L_greedy_next:
    add     w15, w15, #1
    b       .L_greedy_nbr

.L_greedy_check:
    cbnz    w16, .L_greedy_restart
    sub     w9, w9, #1
    b       .L_greedy_layer

.L_greedy_restart:
    ubfiz   x11, x20, #6, #8
    add     x11, x19, x11
    b       .L_greedy_inner

    // =========================================================================
    // PHASE 2: Beam search on layer 0
    // =========================================================================
.L_beam_search:
    // Score entry point
    ubfiz   x11, x20, #6, #8
    add     x11, x19, x11

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
    fmov    w12, s9

    // Initialize beam
    str     w12, [x26]             // beam_scores[0]
    strb    w20, [x26, #32]        // beam_ids[0] (at BEAM_IDS - BEAM_SCORES = 32)
    mov     w8, #1
    strb    w8, [x26, #40]         // beam_count (at BEAM_COUNT - BEAM_SCORES = 40)

    // Mark entry as visited
    lsr     w9, w20, #3
    and     w10, w20, #7
    mov     w11, #1
    lsl     w11, w11, w10
    ldrb    w13, [x25, x9]
    orr     w13, w13, w11
    strb    w13, [x25, x9]

    mov     w6, #0                 // beam cursor

.L_beam_loop:
    ldrb    w8, [x26, #40]         // beam_count
    cmp     w6, w8
    b.ge    .L_beam_done

    // Load candidate
    add     x14, x26, #32
    ldrb    w20, [x14, x6]        // candidate id

    // Get candidate's neighbors
    ubfiz   x11, x20, #6, #8
    add     x11, x19, x11
    add     x13, x11, #NBR_OFF
    ldrb    w14, [x11, #NCNT_OFF]

    mov     w15, #0

.L_beam_nbr:
    cmp     w15, w14
    b.ge    .L_beam_advance

    ldrb    w17, [x13, x15]

    // Check visited
    lsr     w9, w17, #3
    and     w10, w17, #7
    ldrb    w11, [x25, x9]
    lsr     w11, w11, w10
    tbnz    w11, #0, .L_beam_skip

    // Mark visited
    ldrb    w11, [x25, x9]
    mov     w18, #1
    lsl     w18, w18, w10
    orr     w11, w11, w18
    strb    w11, [x25, x9]

    // Score neighbor
    ubfiz   x11, x17, #6, #8
    add     x11, x19, x11

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

    // Insert into beam
    ldrb    w8, [x26, #40]
    cmp     w8, #EF_SEARCH
    b.lt    .L_beam_insert

    // Beam full — check against worst
    sub     w9, w8, #1
    lsl     w10, w9, #2
    ldr     w11, [x26, x10]
    cmp     w18, w11
    b.le    .L_beam_skip

    sub     w8, w8, #1
    b       .L_beam_do_insert

.L_beam_insert:
    // Not full, insert at position w8

.L_beam_do_insert:
    lsl     w9, w8, #2
    str     w18, [x26, x9]
    add     x14, x26, #32
    strb    w17, [x14, x8]

    // Insertion sort: bubble up
.L_sort_loop:
    cbz     w8, .L_sort_done
    sub     w10, w8, #1
    lsl     w11, w10, #2
    ldr     w12, [x26, x11]
    lsl     w9, w8, #2
    ldr     w13, [x26, x9]
    cmp     w13, w12
    b.le    .L_sort_done

    str     w13, [x26, x11]
    str     w12, [x26, x9]
    add     x14, x26, #32
    ldrb    w12, [x14, x10]
    ldrb    w13, [x14, x8]
    strb    w13, [x14, x10]
    strb    w12, [x14, x8]

    mov     w8, w10
    b       .L_sort_loop

.L_sort_done:
    ldrb    w8, [x26, #40]
    cmp     w8, #EF_SEARCH
    b.ge    .L_beam_skip
    add     w8, w8, #1
    strb    w8, [x26, #40]

.L_beam_skip:
    add     w15, w15, #1
    b       .L_beam_nbr

.L_beam_advance:
    add     w6, w6, #1
    b       .L_beam_loop

.L_beam_done:
    // Copy top-k results to output
    mov     w8, #0

.L_copy_results:
    cmp     w8, w21
    b.ge    .L_epilogue

    lsl     w9, w8, #2
    ldr     w10, [x26, x9]
    str     w10, [x23, x9]        // result_scores[i]

    add     x14, x26, #32
    ldrb    w10, [x14, x8]
    strb    w10, [x22, x8]        // result_ids[i]

    add     w8, w8, #1
    b       .L_copy_results

.L_epilogue:
    ldp     d10, d11, [sp, #112]
    ldp     d8, d9, [sp, #96]
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp]
    add     sp, sp, #FRAME_SIZE
    ret
    .size lcvdb_search, . - lcvdb_search
