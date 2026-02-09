// =============================================================================
// L-Cache VDB — Initialization
// =============================================================================
// lcvdb_t layout (AArch64):
//   [0..1]   node_count (uint16)
//   [2]      entry_point (uint8)
//   [3]      max_level (uint8)
//   [4..7]   padding
//   [8..15]  node_array pointer
//   [16..23] upper_layers pointer
//   [24]     M
//   [25]     M_upper
//   [26..27] padding
//   [28..31] prng_state
//   [32..63] reserved
// =============================================================================

    .arch armv8-a
    .text
    .align 4

    .global lcvdb_init
    .type lcvdb_init, %function
lcvdb_init:
    // Zero the entire 64-byte struct
    stp     xzr, xzr, [x0]
    stp     xzr, xzr, [x0, #16]
    stp     xzr, xzr, [x0, #32]
    stp     xzr, xzr, [x0, #48]

    // Set node array pointer (offset 8)
    str     x1, [x0, #8]

    // Set M=8 and M_upper=4 (offsets 24, 25)
    mov     w2, #8
    strb    w2, [x0, #24]
    mov     w2, #4
    strb    w2, [x0, #25]

    // Initialize PRNG with a non-zero seed (offset 28)
    movz    w2, #0x5678
    movk    w2, #0x1234, lsl #16
    str     w2, [x0, #28]

    ret
    .size lcvdb_init, . - lcvdb_init
