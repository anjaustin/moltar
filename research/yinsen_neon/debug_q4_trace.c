#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arm_neon.h>

extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);

static uint32_t rng = 42;
static uint32_t xorshift() {
    uint32_t x = rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng = x;
    return x;
}

int main() {
    int K = 256;
    int K_packed = K / 2;
    
    uint8_t* wgt = malloc(K_packed);
    int8_t* act = malloc(K);
    
    for (int i = 0; i < K_packed; i++) wgt[i] = xorshift() & 0xFF;
    for (int i = 0; i < K; i++) act[i] = (int8_t)((xorshift() % 256) - 128);
    
    int32_t out_mul;
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, K);
    printf("Reference: %d\n", out_mul);
    
    /* Manual NEON trace */
    const uint8x16_t mask_0F = vdupq_n_u8(0x0F);
    const uint8x16_t val_08 = vdupq_n_u8(8);
    
    int32x4_t acc = vdupq_n_s32(0);
    
    const int8_t* a_ptr = act;
    const uint8_t* w_ptr = wgt;
    
    int iteration = 0;
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
        
        /* Just print first element values */
        if (iteration == 0) {
            printf("Iter 0, elem 0:\n");
            printf("  w_packed[0] = 0x%02x\n", vgetq_lane_u8(w_packed, 0));
            printf("  w_even = %d, w_odd = %d\n", vgetq_lane_u8(w_even, 0), vgetq_lane_u8(w_odd, 0));
            printf("  a_even = %d, a_odd = %d\n", vgetq_lane_s8(a_even, 0), vgetq_lane_s8(a_odd, 0));
            printf("  sign_even = 0x%02x, sign_odd = 0x%02x\n", vgetq_lane_u8(sign_even, 0), vgetq_lane_u8(sign_odd, 0));
            printf("  mag_even = %d, mag_odd = %d\n", vgetq_lane_u8(mag_even, 0), vgetq_lane_u8(mag_odd, 0));
            printf("  signed_a_even = %d, signed_a_odd = %d\n", vgetq_lane_s8(signed_a_even, 0), vgetq_lane_s8(signed_a_odd, 0));
            
            /* What should the product be? */
            int w0 = (wgt[0] & 0xF) - 8;
            int w1 = ((wgt[0] >> 4) & 0xF) - 8;
            printf("  Manual: w0=%d, w1=%d\n", w0, w1);
            printf("  Manual products: %d*%d=%d, %d*%d=%d\n", 
                   w0, act[0], w0*act[0], w1, act[1], w1*act[1]);
        }
        
        iteration++;
    }
    
    return 0;
}
