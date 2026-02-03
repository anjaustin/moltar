/* Debug version of NEON kernel with per-iteration output */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arm_neon.h>

void q4_matvec_shiftadd_neon_debug(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ act,
    const uint8_t* __restrict__ wgt,
    int N,
    int K
) {
    const int K_packed = K / 2;
    const uint8x16_t mask_0F = vdupq_n_u8(0x0F);
    const uint8x16_t val_08 = vdupq_n_u8(8);
    
    for (int n = 0; n < N; n++) {
        int32x4_t acc = vdupq_n_s32(0);
        
        const int8_t* a_ptr = act;
        const uint8_t* w_ptr = wgt + n * K_packed;
        
        int iter = 0;
        for (int k = 0; k < K; k += 32) {
            uint8x16_t w_packed = vld1q_u8(w_ptr);
            w_ptr += 16;
            
            int8x16x2_t a_interleaved = vld2q_s8(a_ptr);
            a_ptr += 32;
            
            int8x16_t a_even = a_interleaved.val[0];
            int8x16_t a_odd = a_interleaved.val[1];
            
            uint8x16_t w_even = vandq_u8(w_packed, mask_0F);
            uint8x16_t w_odd = vshrq_n_u8(w_packed, 4);
            
            uint8x16_t sign_even = vcltq_u8(w_even, val_08);
            uint8x16_t sign_odd = vcltq_u8(w_odd, val_08);
            
            uint8x16_t mag_even = vabdq_u8(w_even, val_08);
            uint8x16_t mag_odd = vabdq_u8(w_odd, val_08);
            
            int8x16_t neg_a_even = vnegq_s8(a_even);
            int8x16_t neg_a_odd = vnegq_s8(a_odd);
            int8x16_t signed_a_even = vbslq_s8(sign_even, neg_a_even, a_even);
            int8x16_t signed_a_odd = vbslq_s8(sign_odd, neg_a_odd, a_odd);
            
            int16x8_t ae_lo = vmovl_s8(vget_low_s8(signed_a_even));
            int16x8_t ae_hi = vmovl_s8(vget_high_s8(signed_a_even));
            
            uint8x8_t me_lo = vget_low_u8(mag_even);
            uint8x8_t me_hi = vget_high_u8(mag_even);
            
            uint16x8_t b0_mask_e_lo = vcgtq_u16(vmovl_u8(vtst_u8(me_lo, vdup_n_u8(1))), vdupq_n_u16(0));
            uint16x8_t b1_mask_e_lo = vcgtq_u16(vmovl_u8(vtst_u8(me_lo, vdup_n_u8(2))), vdupq_n_u16(0));
            uint16x8_t b2_mask_e_lo = vcgtq_u16(vmovl_u8(vtst_u8(me_lo, vdup_n_u8(4))), vdupq_n_u16(0));
            uint16x8_t b3_mask_e_lo = vcgtq_u16(vmovl_u8(vtst_u8(me_lo, vdup_n_u8(8))), vdupq_n_u16(0));
            
            uint16x8_t b0_mask_e_hi = vcgtq_u16(vmovl_u8(vtst_u8(me_hi, vdup_n_u8(1))), vdupq_n_u16(0));
            uint16x8_t b1_mask_e_hi = vcgtq_u16(vmovl_u8(vtst_u8(me_hi, vdup_n_u8(2))), vdupq_n_u16(0));
            uint16x8_t b2_mask_e_hi = vcgtq_u16(vmovl_u8(vtst_u8(me_hi, vdup_n_u8(4))), vdupq_n_u16(0));
            uint16x8_t b3_mask_e_hi = vcgtq_u16(vmovl_u8(vtst_u8(me_hi, vdup_n_u8(8))), vdupq_n_u16(0));
            
            int16x8_t prod_e_lo = vandq_s16(ae_lo, vreinterpretq_s16_u16(b0_mask_e_lo));
            prod_e_lo = vaddq_s16(prod_e_lo, vandq_s16(vshlq_n_s16(ae_lo, 1), vreinterpretq_s16_u16(b1_mask_e_lo)));
            prod_e_lo = vaddq_s16(prod_e_lo, vandq_s16(vshlq_n_s16(ae_lo, 2), vreinterpretq_s16_u16(b2_mask_e_lo)));
            prod_e_lo = vaddq_s16(prod_e_lo, vandq_s16(vshlq_n_s16(ae_lo, 3), vreinterpretq_s16_u16(b3_mask_e_lo)));
            
            int16x8_t prod_e_hi = vandq_s16(ae_hi, vreinterpretq_s16_u16(b0_mask_e_hi));
            prod_e_hi = vaddq_s16(prod_e_hi, vandq_s16(vshlq_n_s16(ae_hi, 1), vreinterpretq_s16_u16(b1_mask_e_hi)));
            prod_e_hi = vaddq_s16(prod_e_hi, vandq_s16(vshlq_n_s16(ae_hi, 2), vreinterpretq_s16_u16(b2_mask_e_hi)));
            prod_e_hi = vaddq_s16(prod_e_hi, vandq_s16(vshlq_n_s16(ae_hi, 3), vreinterpretq_s16_u16(b3_mask_e_hi)));
            
            int16x8_t ao_lo = vmovl_s8(vget_low_s8(signed_a_odd));
            int16x8_t ao_hi = vmovl_s8(vget_high_s8(signed_a_odd));
            
            uint8x8_t mo_lo = vget_low_u8(mag_odd);
            uint8x8_t mo_hi = vget_high_u8(mag_odd);
            
            uint16x8_t b0_mask_o_lo = vcgtq_u16(vmovl_u8(vtst_u8(mo_lo, vdup_n_u8(1))), vdupq_n_u16(0));
            uint16x8_t b1_mask_o_lo = vcgtq_u16(vmovl_u8(vtst_u8(mo_lo, vdup_n_u8(2))), vdupq_n_u16(0));
            uint16x8_t b2_mask_o_lo = vcgtq_u16(vmovl_u8(vtst_u8(mo_lo, vdup_n_u8(4))), vdupq_n_u16(0));
            uint16x8_t b3_mask_o_lo = vcgtq_u16(vmovl_u8(vtst_u8(mo_lo, vdup_n_u8(8))), vdupq_n_u16(0));
            
            uint16x8_t b0_mask_o_hi = vcgtq_u16(vmovl_u8(vtst_u8(mo_hi, vdup_n_u8(1))), vdupq_n_u16(0));
            uint16x8_t b1_mask_o_hi = vcgtq_u16(vmovl_u8(vtst_u8(mo_hi, vdup_n_u8(2))), vdupq_n_u16(0));
            uint16x8_t b2_mask_o_hi = vcgtq_u16(vmovl_u8(vtst_u8(mo_hi, vdup_n_u8(4))), vdupq_n_u16(0));
            uint16x8_t b3_mask_o_hi = vcgtq_u16(vmovl_u8(vtst_u8(mo_hi, vdup_n_u8(8))), vdupq_n_u16(0));
            
            int16x8_t prod_o_lo = vandq_s16(ao_lo, vreinterpretq_s16_u16(b0_mask_o_lo));
            prod_o_lo = vaddq_s16(prod_o_lo, vandq_s16(vshlq_n_s16(ao_lo, 1), vreinterpretq_s16_u16(b1_mask_o_lo)));
            prod_o_lo = vaddq_s16(prod_o_lo, vandq_s16(vshlq_n_s16(ao_lo, 2), vreinterpretq_s16_u16(b2_mask_o_lo)));
            prod_o_lo = vaddq_s16(prod_o_lo, vandq_s16(vshlq_n_s16(ao_lo, 3), vreinterpretq_s16_u16(b3_mask_o_lo)));
            
            int16x8_t prod_o_hi = vandq_s16(ao_hi, vreinterpretq_s16_u16(b0_mask_o_hi));
            prod_o_hi = vaddq_s16(prod_o_hi, vandq_s16(vshlq_n_s16(ao_hi, 1), vreinterpretq_s16_u16(b1_mask_o_hi)));
            prod_o_hi = vaddq_s16(prod_o_hi, vandq_s16(vshlq_n_s16(ao_hi, 2), vreinterpretq_s16_u16(b2_mask_o_hi)));
            prod_o_hi = vaddq_s16(prod_o_hi, vandq_s16(vshlq_n_s16(ao_hi, 3), vreinterpretq_s16_u16(b3_mask_o_hi)));
            
            /* Print intermediate values */
            int32_t iter_sum_e_lo = vaddvq_s32(vpaddlq_s16(prod_e_lo));
            int32_t iter_sum_e_hi = vaddvq_s32(vpaddlq_s16(prod_e_hi));
            int32_t iter_sum_o_lo = vaddvq_s32(vpaddlq_s16(prod_o_lo));
            int32_t iter_sum_o_hi = vaddvq_s32(vpaddlq_s16(prod_o_hi));
            printf("Iter %d: e_lo=%d, e_hi=%d, o_lo=%d, o_hi=%d, total=%d\n",
                   iter, iter_sum_e_lo, iter_sum_e_hi, iter_sum_o_lo, iter_sum_o_hi,
                   iter_sum_e_lo + iter_sum_e_hi + iter_sum_o_lo + iter_sum_o_hi);
            
            acc = vaddq_s32(acc, vpaddlq_s16(prod_e_lo));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_e_hi));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_o_lo));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_o_hi));
            
            printf("  Running acc: %d\n", vaddvq_s32(acc));
            iter++;
        }
        
        out[n] = vaddvq_s32(acc);
    }
}

static uint32_t rng = 42;
static uint32_t xorshift() {
    uint32_t x = rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng = x;
    return x;
}

extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);

int main() {
    int K = 256;
    int K_packed = K / 2;
    
    uint8_t* wgt = (uint8_t*)malloc(K_packed);
    int8_t* act = (int8_t*)malloc(K);
    
    for (int i = 0; i < K_packed; i++) wgt[i] = xorshift() & 0xFF;
    for (int i = 0; i < K; i++) act[i] = (int8_t)((xorshift() % 256) - 128);
    
    int32_t out_mul, out_neon;
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, K);
    printf("Reference: %d\n\n", out_mul);
    
    printf("NEON Debug:\n");
    q4_matvec_shiftadd_neon_debug(&out_neon, act, wgt, 1, K);
    printf("\nFinal NEON: %d\n", out_neon);
    
    return 0;
}
