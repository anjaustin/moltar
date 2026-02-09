// =============================================================================
// L-Cache VDB — NEON Distance Functions
// =============================================================================
// int8 dot product for 48-dimensional vectors
// Each vector is 48 bytes (fits in 3 NEON registers)
//
// ARM AArch64 NEON assembly
// Cache-line aware: vectors are 48 bytes within a 64-byte cache line
// =============================================================================

    .arch armv8-a
    .text
    .align 4

// =============================================================================
// lcvdb_dot_i8
// Compute int8 dot product of two 48D vectors
//
// Parameters:
//   x0 = pointer to vector A (48 x int8)
//   x1 = pointer to vector B (48 x int8)
//
// Returns:
//   w0 = int32 dot product
//
// Registers used:
//   v0-v2: vector A (3 x 16 x int8)
//   v3-v5: vector B (3 x 16 x int8)
//   v6-v8: intermediate products (widened to int16)
//   v9:    accumulator
//
// Cycle estimate: ~12 cycles (fully pipelined)
// Code size: 72 bytes (18 instructions x 4 bytes)
// =============================================================================
    .global lcvdb_dot_i8
    .type lcvdb_dot_i8, %function
lcvdb_dot_i8:
    // Load vector A: 48 bytes = 3 x 16-byte loads
    ld1     {v0.16b, v1.16b, v2.16b}, [x0]

    // Load vector B: 48 bytes = 3 x 16-byte loads
    ld1     {v3.16b, v4.16b, v5.16b}, [x1]

    // Chunk 0 (dims 0-15): int8 x int8 -> int16
    smull   v6.8h, v0.8b, v3.8b        // dims 0-7
    smull2  v7.8h, v0.16b, v3.16b      // dims 8-15
    // Widen chunk 0 to int32 immediately (avoid int16 overflow)
    saddlp  v8.4s, v6.8h               // 8 x int16 -> 4 x int32
    sadalp  v8.4s, v7.8h               // accumulate

    // Chunk 1 (dims 16-31): int8 x int8 -> int16
    smull   v6.8h, v1.8b, v4.8b        // dims 16-23
    smull2  v7.8h, v1.16b, v4.16b      // dims 24-31
    // Widen and accumulate into int32
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h

    // Chunk 2 (dims 32-47): int8 x int8 -> int16
    smull   v6.8h, v2.8b, v5.8b        // dims 32-39
    smull2  v7.8h, v2.16b, v5.16b      // dims 40-47
    // Widen and accumulate into int32
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h

    // Horizontal reduction: 4 x int32 -> 1 x int32
    addv    s9, v8.4s                   // sum all 4 lanes
    fmov    w0, s9                      // move to general register

    ret
    .size lcvdb_dot_i8, . - lcvdb_dot_i8


// =============================================================================
// lcvdb_dot_i8_preloaded
// Compute dot product when query is already in v0-v2
// (Used in the search inner loop — query stays in registers)
//
// Parameters:
//   x0 = pointer to candidate vector (48 x int8)
//   Query assumed preloaded in v0, v1, v2
//
// Returns:
//   w0 = int32 dot product
//   v0-v2 preserved (query stays loaded)
//
// Code size: 48 bytes (12 instructions)
// =============================================================================
    .global lcvdb_dot_i8_preloaded
    .type lcvdb_dot_i8_preloaded, %function
lcvdb_dot_i8_preloaded:
    // Load candidate vector
    ld1     {v3.16b, v4.16b, v5.16b}, [x0]

    // Chunk 0: widen immediately to avoid int16 overflow
    smull   v6.8h, v0.8b, v3.8b
    smull2  v7.8h, v0.16b, v3.16b
    saddlp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h

    // Chunk 1
    smull   v6.8h, v1.8b, v4.8b
    smull2  v7.8h, v1.16b, v4.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h

    // Chunk 2
    smull   v6.8h, v2.8b, v5.8b
    smull2  v7.8h, v2.16b, v5.16b
    sadalp  v8.4s, v6.8h
    sadalp  v8.4s, v7.8h

    // Reduce
    addv    s9, v8.4s
    fmov    w0, s9

    ret
    .size lcvdb_dot_i8_preloaded, . - lcvdb_dot_i8_preloaded


// =============================================================================
// lcvdb_load_query
// Preload query vector into v0-v2 for repeated use
//
// Parameters:
//   x0 = pointer to query vector (48 x int8)
//
// Code size: 8 bytes (2 instructions)
// =============================================================================
    .global lcvdb_load_query
    .type lcvdb_load_query, %function
lcvdb_load_query:
    ld1     {v0.16b, v1.16b, v2.16b}, [x0]
    ret
    .size lcvdb_load_query, . - lcvdb_load_query


// =============================================================================
// lcvdb_dot_i8_batch4
// Compute dot product of query (in v0-v2) against 4 candidates
// For evaluating all neighbors of a node at once
//
// Parameters:
//   x0 = pointer to candidate 0 vector (48 x int8)
//   x1 = pointer to candidate 1 vector
//   x2 = pointer to candidate 2 vector
//   x3 = pointer to candidate 3 vector
//
// Returns:
//   w0 = dot product with candidate 0
//   w1 = dot product with candidate 1
//   w2 = dot product with candidate 2
//   w3 = dot product with candidate 3
//
// Code size: 100 bytes (25 instructions)
// =============================================================================
    .global lcvdb_dot_i8_batch4
    .type lcvdb_dot_i8_batch4, %function
lcvdb_dot_i8_batch4:
    // Save frame
    stp     x29, x30, [sp, #-16]!

    // Candidate 0 — widen per-chunk to avoid int16 overflow
    ld1     {v3.16b, v4.16b, v5.16b}, [x0]
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
    addv    s16, v8.4s              // result 0 in s16

    // Candidate 1
    ld1     {v3.16b, v4.16b, v5.16b}, [x1]
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
    addv    s17, v8.4s              // result 1 in s17

    // Candidate 2
    ld1     {v3.16b, v4.16b, v5.16b}, [x2]
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
    addv    s18, v8.4s              // result 2 in s18

    // Candidate 3
    ld1     {v3.16b, v4.16b, v5.16b}, [x3]
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
    addv    s19, v8.4s              // result 3 in s19

    // Move results to w0-w3
    fmov    w0, s16
    fmov    w1, s17
    fmov    w2, s18
    fmov    w3, s19

    ldp     x29, x30, [sp], #16
    ret
    .size lcvdb_dot_i8_batch4, . - lcvdb_dot_i8_batch4
