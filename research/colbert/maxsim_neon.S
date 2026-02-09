/*
 * maxsim_neon.S — NEON-optimized ColBERT primitives for AArch64
 *
 * int32_t colbert_dot_i8(const int8_t *a, const int8_t *b)
 *   128D int8 dot product. Overflow-safe via per-chunk SMULL/SADALP widening.
 *
 * int32_t colbert_maxsim_i8(const int8_t *query, const int8_t *doc, int n_q, int n_d)
 *   Full MaxSim score: Σ_i max_j dot(q_i, d_j) for all query tokens i.
 */

    .arch armv8.2-a+dotprod
    .text

/* ────────────────────────────────────────────────────────────────
 * colbert_dot_i8: 128D int8 dot product
 *
 *   x0 = const int8_t *a   (128 bytes)
 *   x1 = const int8_t *b   (128 bytes)
 *   Returns: int32 dot product in w0
 *
 * Uses SDOT (signed dot product) — 4 lanes × 4 bytes per instruction.
 * 128D = 8 × 16-byte loads = 8 SDOT pairs.
 * ──────────────────────────────────────────────────────────────── */
    .global colbert_dot_i8
    .type   colbert_dot_i8, %function
    .p2align 4
colbert_dot_i8:
    /* Zero accumulator */
    movi    v0.4s, #0

    /* Load 128 bytes from a (v16..v23) and b (v24..v31), SDOT into v0 */
    ldp     q16, q17, [x0]
    ldp     q24, q25, [x1]
    sdot    v0.4s, v16.16b, v24.16b
    sdot    v0.4s, v17.16b, v25.16b

    ldp     q18, q19, [x0, #32]
    ldp     q26, q27, [x1, #32]
    sdot    v0.4s, v18.16b, v26.16b
    sdot    v0.4s, v19.16b, v27.16b

    ldp     q20, q21, [x0, #64]
    ldp     q28, q29, [x1, #64]
    sdot    v0.4s, v20.16b, v28.16b
    sdot    v0.4s, v21.16b, v29.16b

    ldp     q22, q23, [x0, #96]
    ldp     q30, q31, [x1, #96]
    sdot    v0.4s, v22.16b, v30.16b
    sdot    v0.4s, v23.16b, v31.16b

    /* Horizontal sum: v0.s[0]+v0.s[1]+v0.s[2]+v0.s[3] */
    addv    s0, v0.4s
    fmov    w0, s0
    ret
    .size   colbert_dot_i8, . - colbert_dot_i8

/* ────────────────────────────────────────────────────────────────
 * colbert_maxsim_i8: Full MaxSim scoring
 *
 *   x0 = const int8_t *query   [n_q × 128]
 *   x1 = const int8_t *doc     [n_d × 128]
 *   w2 = int n_q               (number of query tokens)
 *   w3 = int n_d               (number of document tokens)
 *   Returns: int32 MaxSim score in w0
 *
 * Algorithm:
 *   total = 0
 *   for each query token q_i:
 *     max_score = INT32_MIN
 *     for each doc token d_j:
 *       s = dot(q_i, d_j)
 *       if s > max_score: max_score = s
 *     total += max_score
 *   return total
 *
 * Register allocation:
 *   x0  = query base pointer (advances by 128 per query token)
 *   x1  = doc base pointer (reset each outer iteration)
 *   w2  = n_q counter (decremented)
 *   w3  = n_d (constant)
 *   x4  = doc base (saved)
 *   w5  = inner loop counter
 *   w6  = current dot product
 *   w7  = max_score for current query token
 *   w8  = total MaxSim score
 *   x9  = current doc pointer (inner loop)
 * ──────────────────────────────────────────────────────────────── */
    .global colbert_maxsim_i8
    .type   colbert_maxsim_i8, %function
    .p2align 4
colbert_maxsim_i8:
    /* Save callee-saved registers */
    stp     x19, x20, [sp, #-32]!
    stp     x21, x22, [sp, #16]

    mov     x4, x1              /* x4 = doc base (saved) */
    mov     w8, #0              /* w8 = total score */
    cbz     w2, .Lms_done       /* if n_q == 0, return 0 */
    cbz     w3, .Lms_done       /* if n_d == 0, return 0 */

    mov     w19, w2             /* w19 = n_q (outer counter) */
    mov     w20, w3             /* w20 = n_d (constant) */
    mov     x21, x0             /* x21 = query pointer */

.Lms_outer:
    /* For this query token, find max dot over all doc tokens */
    mov     w7, #1
    ror     w7, w7, #1          /* w7 = 0x80000000 = INT32_MIN */
    mov     x9, x4              /* x9 = doc pointer (reset to base) */
    mov     w5, w20             /* w5 = n_d (inner counter) */

.Lms_inner:
    /* Compute dot(query[i], doc[j]) using SDOT */
    movi    v0.4s, #0

    ldp     q16, q17, [x21]
    ldp     q24, q25, [x9]
    sdot    v0.4s, v16.16b, v24.16b
    sdot    v0.4s, v17.16b, v25.16b

    ldp     q18, q19, [x21, #32]
    ldp     q26, q27, [x9, #32]
    sdot    v0.4s, v18.16b, v26.16b
    sdot    v0.4s, v19.16b, v27.16b

    ldp     q20, q2, [x21, #64]
    ldp     q28, q29, [x9, #64]
    sdot    v0.4s, v20.16b, v28.16b
    sdot    v0.4s, v2.16b, v29.16b

    ldp     q22, q23, [x21, #96]
    ldp     q30, q31, [x9, #96]
    sdot    v0.4s, v22.16b, v30.16b
    sdot    v0.4s, v23.16b, v31.16b

    /* Horizontal sum */
    addv    s0, v0.4s
    fmov    w6, s0

    /* max_score = max(max_score, dot) */
    cmp     w6, w7
    csel    w7, w6, w7, gt

    /* Advance doc pointer, decrement inner counter */
    add     x9, x9, #128
    subs    w5, w5, #1
    b.ne    .Lms_inner

    /* total += max_score */
    add     w8, w8, w7

    /* Advance query pointer, decrement outer counter */
    add     x21, x21, #128
    subs    w19, w19, #1
    b.ne    .Lms_outer

.Lms_done:
    mov     w0, w8

    /* Restore callee-saved registers */
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #32
    ret
    .size   colbert_maxsim_i8, . - colbert_maxsim_i8
