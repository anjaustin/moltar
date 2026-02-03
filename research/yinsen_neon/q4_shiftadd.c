/*
 * q4_shiftadd.c - Q4_0 MatVec using Shift-Add (No Multiplication)
 *
 * Replaces w * x multiplication with:
 *   1. XOR + add for conditional negation (sign handling)
 *   2. Shift-add decomposition for magnitude (0-8)
 *
 * Q4_0 format: 4-bit weights, values 0-15, centered at 8
 *   effective weight = raw - 8, range [-8, +7]
 *
 * Copyright 2026 Trix Research
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ============================================================================
 * REFERENCE IMPLEMENTATION
 * ============================================================================ */

/*
 * Shift-add multiply for magnitude 0-8
 * Returns mag * x using only shifts and adds
 */
static inline int32_t shiftadd_mul(int32_t x, int mag) {
    switch (mag) {
        case 0: return 0;
        case 1: return x;
        case 2: return x << 1;
        case 3: return x + (x << 1);
        case 4: return x << 2;
        case 5: return x + (x << 2);
        case 6: return (x << 1) + (x << 2);
        case 7: return x + (x << 1) + (x << 2);  // or (x << 3) - x
        case 8: return x << 3;
        default: return 0;
    }
}

/*
 * q4_matvec_shiftadd_ref - Reference shift-add implementation
 *
 * For each 4-bit weight w (0-15):
 *   centered = w - 8  (range -8 to +7)
 *   sign = (centered < 0) ? 1 : 0
 *   magnitude = abs(centered)
 *   result += sign ? -shiftadd(x, mag) : +shiftadd(x, mag)
 */
void q4_matvec_shiftadd_ref(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,  /* Packed Q4: 2 weights per byte */
    int N,  /* Output channels */
    int K   /* Input channels (must be even) */
) {
    const int K_packed = K / 2;  /* 2 weights per byte */
    
    for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        const uint8_t* w_row = wgt + n * K_packed;
        
        for (int k = 0; k < K; k += 2) {
            uint8_t packed = w_row[k / 2];
            
            /* First weight (low nibble) */
            int w0 = (packed & 0x0F) - 8;  /* Center: -8 to +7 */
            int8_t a0 = act[k];
            int sign0 = (w0 < 0) ? 1 : 0;
            int mag0 = sign0 ? -w0 : w0;
            int32_t prod0 = shiftadd_mul(a0, mag0);
            acc += sign0 ? -prod0 : prod0;
            
            /* Second weight (high nibble) */
            int w1 = ((packed >> 4) & 0x0F) - 8;
            int8_t a1 = act[k + 1];
            int sign1 = (w1 < 0) ? 1 : 0;
            int mag1 = sign1 ? -w1 : w1;
            int32_t prod1 = shiftadd_mul(a1, mag1);
            acc += sign1 ? -prod1 : prod1;
        }
        out[n] = acc;
    }
}

/* ============================================================================
 * NEON IMPLEMENTATION - Bit Decomposition
 * ============================================================================ */

#ifdef __ARM_NEON

/*
 * q4_matvec_shiftadd_neon - NEON shift-add kernel
 *
 * Strategy: Decompose 4-bit magnitude into bits, selectively add shifted values
 *
 * For magnitude m = b2*4 + b1*2 + b0*1 (0-7 range, handle 8 separately):
 *   m * x = (b0 ? x : 0) + (b1 ? x<<1 : 0) + (b2 ? x<<2 : 0)
 *
 * Sign handling via XOR:
 *   -x = (x XOR 0xFF) + 1  (for int8)
 *   Conditional: (x XOR sign_mask) + sign_bit
 */
void q4_matvec_shiftadd_neon(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,
    int N,
    int K
) {
    const int K_packed = K / 2;
    
    /* Masks */
    const uint8x16_t mask_0F = vdupq_n_u8(0x0F);
    const uint8x16_t val_08 = vdupq_n_u8(8);
    
    for (int n = 0; n < N; n++) {
        int32x4_t acc = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const uint8_t* w_ptr = wgt + n * K_packed;
        
        /* Process 32 activations (16 packed weight bytes) per iteration */
        for (int k = 0; k < K; k += 32) {
            /* Load 16 packed bytes = 32 weights */
            uint8x16_t w_packed = vld1q_u8(w_ptr);
            w_ptr += 16;
            
            /* Load 32 activations as interleaved pairs:
             * VLD2 gives us:
             *   a_even: [a0, a2, a4, ...a30]  - 16 values
             *   a_odd:  [a1, a3, a5, ...a31]  - 16 values
             * This matches Q4 packing where low nibble -> even, high nibble -> odd
             */
            int8x16x2_t a_interleaved = vld2q_s8(a_ptr);
            a_ptr += 32;
            
            int8x16_t a_even = a_interleaved.val[0];  /* For low nibbles */
            int8x16_t a_odd = a_interleaved.val[1];   /* For high nibbles */
            
            /* Extract low and high nibbles */
            uint8x16_t w_even = vandq_u8(w_packed, mask_0F);   /* Low nibble -> even activations */
            uint8x16_t w_odd = vshrq_n_u8(w_packed, 4);        /* High nibble -> odd activations */
            
            /* Centered weight = w - 8
             * For w < 8: negative, magnitude = 8 - w
             * For w >= 8: non-negative, magnitude = w - 8
             */
            
            /* sign_mask: 0xFF if w < 8 (negative), 0x00 otherwise */
            uint8x16_t sign_even = vcltq_u8(w_even, val_08);
            uint8x16_t sign_odd = vcltq_u8(w_odd, val_08);
            
            /* magnitude = |w - 8| */
            uint8x16_t mag_even = vabdq_u8(w_even, val_08);
            uint8x16_t mag_odd = vabdq_u8(w_odd, val_08);
            
            /* Apply sign: signed_a = sign ? -a : a
             * CRITICAL: vneg_s8(-128) = -128 due to overflow!
             * We must widen to int16 BEFORE negation.
             */
            int16x8_t ae_lo_raw = vmovl_s8(vget_low_s8(a_even));
            int16x8_t ae_hi_raw = vmovl_s8(vget_high_s8(a_even));
            int16x8_t ao_lo_raw = vmovl_s8(vget_low_s8(a_odd));
            int16x8_t ao_hi_raw = vmovl_s8(vget_high_s8(a_odd));
            
            /* Expand sign masks to int16 properly */
            uint16x8_t sign_e_lo = vcgtq_u16(vmovl_u8(vget_low_u8(sign_even)), vdupq_n_u16(0));
            uint16x8_t sign_e_hi = vcgtq_u16(vmovl_u8(vget_high_u8(sign_even)), vdupq_n_u16(0));
            uint16x8_t sign_o_lo = vcgtq_u16(vmovl_u8(vget_low_u8(sign_odd)), vdupq_n_u16(0));
            uint16x8_t sign_o_hi = vcgtq_u16(vmovl_u8(vget_high_u8(sign_odd)), vdupq_n_u16(0));
            
            /* Negate in int16 (safe for -128) */
            int16x8_t neg_ae_lo = vnegq_s16(ae_lo_raw);
            int16x8_t neg_ae_hi = vnegq_s16(ae_hi_raw);
            int16x8_t neg_ao_lo = vnegq_s16(ao_lo_raw);
            int16x8_t neg_ao_hi = vnegq_s16(ao_hi_raw);
            
            /* Select based on sign */
            int16x8_t ae_lo = vbslq_s16(sign_e_lo, neg_ae_lo, ae_lo_raw);
            int16x8_t ae_hi = vbslq_s16(sign_e_hi, neg_ae_hi, ae_hi_raw);
            int16x8_t ao_lo = vbslq_s16(sign_o_lo, neg_ao_lo, ao_lo_raw);
            int16x8_t ao_hi = vbslq_s16(sign_o_hi, neg_ao_hi, ao_hi_raw);
            
            /* Bit decomposition for magnitude:
             * result = (b0 ? a : 0) + (b1 ? a*2 : 0) + (b2 ? a*4 : 0) + (b3 ? a*8 : 0)
             *
             * ae_lo, ae_hi, ao_lo, ao_hi are already computed above (signed_a in int16)
             */
            
            uint8x8_t me_lo = vget_low_u8(mag_even);
            uint8x8_t me_hi = vget_high_u8(mag_even);
            
            /* For each bit position, extract and expand to FULL 16-bit mask
             * vmovl_u8(0xFF) gives 0x00FF, not 0xFFFF!
             * Use vcgt to expand: if (x > 0) -> 0xFFFF, else 0x0000
             */
            /* Bit 0 */
            uint16x8_t b0_mask_e_lo = vcgtq_u16(vmovl_u8(vtst_u8(me_lo, vdup_n_u8(1))), vdupq_n_u16(0));
            uint16x8_t b0_mask_e_hi = vcgtq_u16(vmovl_u8(vtst_u8(me_hi, vdup_n_u8(1))), vdupq_n_u16(0));
            /* Bit 1 */
            uint16x8_t b1_mask_e_lo = vcgtq_u16(vmovl_u8(vtst_u8(me_lo, vdup_n_u8(2))), vdupq_n_u16(0));
            uint16x8_t b1_mask_e_hi = vcgtq_u16(vmovl_u8(vtst_u8(me_hi, vdup_n_u8(2))), vdupq_n_u16(0));
            /* Bit 2 */
            uint16x8_t b2_mask_e_lo = vcgtq_u16(vmovl_u8(vtst_u8(me_lo, vdup_n_u8(4))), vdupq_n_u16(0));
            uint16x8_t b2_mask_e_hi = vcgtq_u16(vmovl_u8(vtst_u8(me_hi, vdup_n_u8(4))), vdupq_n_u16(0));
            /* Bit 3 */
            uint16x8_t b3_mask_e_lo = vcgtq_u16(vmovl_u8(vtst_u8(me_lo, vdup_n_u8(8))), vdupq_n_u16(0));
            uint16x8_t b3_mask_e_hi = vcgtq_u16(vmovl_u8(vtst_u8(me_hi, vdup_n_u8(8))), vdupq_n_u16(0));
            
            /* Compute products using shifts and masked adds */
            int16x8_t prod_e_lo = vandq_s16(ae_lo, vreinterpretq_s16_u16(b0_mask_e_lo));
            prod_e_lo = vaddq_s16(prod_e_lo, vandq_s16(vshlq_n_s16(ae_lo, 1), vreinterpretq_s16_u16(b1_mask_e_lo)));
            prod_e_lo = vaddq_s16(prod_e_lo, vandq_s16(vshlq_n_s16(ae_lo, 2), vreinterpretq_s16_u16(b2_mask_e_lo)));
            prod_e_lo = vaddq_s16(prod_e_lo, vandq_s16(vshlq_n_s16(ae_lo, 3), vreinterpretq_s16_u16(b3_mask_e_lo)));
            
            int16x8_t prod_e_hi = vandq_s16(ae_hi, vreinterpretq_s16_u16(b0_mask_e_hi));
            prod_e_hi = vaddq_s16(prod_e_hi, vandq_s16(vshlq_n_s16(ae_hi, 1), vreinterpretq_s16_u16(b1_mask_e_hi)));
            prod_e_hi = vaddq_s16(prod_e_hi, vandq_s16(vshlq_n_s16(ae_hi, 2), vreinterpretq_s16_u16(b2_mask_e_hi)));
            prod_e_hi = vaddq_s16(prod_e_hi, vandq_s16(vshlq_n_s16(ae_hi, 3), vreinterpretq_s16_u16(b3_mask_e_hi)));
            
            /* Odd activations - ao_lo, ao_hi already computed above */
            uint8x8_t mo_lo = vget_low_u8(mag_odd);
            uint8x8_t mo_hi = vget_high_u8(mag_odd);
            
            uint16x8_t b0_mask_o_lo = vcgtq_u16(vmovl_u8(vtst_u8(mo_lo, vdup_n_u8(1))), vdupq_n_u16(0));
            uint16x8_t b0_mask_o_hi = vcgtq_u16(vmovl_u8(vtst_u8(mo_hi, vdup_n_u8(1))), vdupq_n_u16(0));
            uint16x8_t b1_mask_o_lo = vcgtq_u16(vmovl_u8(vtst_u8(mo_lo, vdup_n_u8(2))), vdupq_n_u16(0));
            uint16x8_t b1_mask_o_hi = vcgtq_u16(vmovl_u8(vtst_u8(mo_hi, vdup_n_u8(2))), vdupq_n_u16(0));
            uint16x8_t b2_mask_o_lo = vcgtq_u16(vmovl_u8(vtst_u8(mo_lo, vdup_n_u8(4))), vdupq_n_u16(0));
            uint16x8_t b2_mask_o_hi = vcgtq_u16(vmovl_u8(vtst_u8(mo_hi, vdup_n_u8(4))), vdupq_n_u16(0));
            uint16x8_t b3_mask_o_lo = vcgtq_u16(vmovl_u8(vtst_u8(mo_lo, vdup_n_u8(8))), vdupq_n_u16(0));
            uint16x8_t b3_mask_o_hi = vcgtq_u16(vmovl_u8(vtst_u8(mo_hi, vdup_n_u8(8))), vdupq_n_u16(0));
            
            int16x8_t prod_o_lo = vandq_s16(ao_lo, vreinterpretq_s16_u16(b0_mask_o_lo));
            prod_o_lo = vaddq_s16(prod_o_lo, vandq_s16(vshlq_n_s16(ao_lo, 1), vreinterpretq_s16_u16(b1_mask_o_lo)));
            prod_o_lo = vaddq_s16(prod_o_lo, vandq_s16(vshlq_n_s16(ao_lo, 2), vreinterpretq_s16_u16(b2_mask_o_lo)));
            prod_o_lo = vaddq_s16(prod_o_lo, vandq_s16(vshlq_n_s16(ao_lo, 3), vreinterpretq_s16_u16(b3_mask_o_lo)));
            
            int16x8_t prod_o_hi = vandq_s16(ao_hi, vreinterpretq_s16_u16(b0_mask_o_hi));
            prod_o_hi = vaddq_s16(prod_o_hi, vandq_s16(vshlq_n_s16(ao_hi, 1), vreinterpretq_s16_u16(b1_mask_o_hi)));
            prod_o_hi = vaddq_s16(prod_o_hi, vandq_s16(vshlq_n_s16(ao_hi, 2), vreinterpretq_s16_u16(b2_mask_o_hi)));
            prod_o_hi = vaddq_s16(prod_o_hi, vandq_s16(vshlq_n_s16(ao_hi, 3), vreinterpretq_s16_u16(b3_mask_o_hi)));
            
            /* Accumulate all into int32 */
            acc = vaddq_s32(acc, vpaddlq_s16(prod_e_lo));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_e_hi));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_o_lo));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_o_hi));
        }
        
        /* Horizontal sum */
        out[n] = vaddvq_s32(acc);
    }
}

#endif /* __ARM_NEON */

/* ============================================================================
 * STANDARD MULTIPLY REFERENCE (for comparison)
 * ============================================================================ */

void q4_matvec_multiply_ref(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,
    int N,
    int K
) {
    const int K_packed = K / 2;
    
    for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        const uint8_t* w_row = wgt + n * K_packed;
        
        for (int k = 0; k < K; k += 2) {
            uint8_t packed = w_row[k / 2];
            
            int w0 = (packed & 0x0F) - 8;
            int w1 = ((packed >> 4) & 0x0F) - 8;
            
            acc += w0 * act[k];
            acc += w1 * act[k + 1];
        }
        out[n] = acc;
    }
}
