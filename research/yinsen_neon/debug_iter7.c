#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arm_neon.h>

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
    
    uint8_t* wgt = (uint8_t*)malloc(K_packed);
    int8_t* act = (int8_t*)malloc(K);
    
    for (int i = 0; i < K_packed; i++) wgt[i] = xorshift() & 0xFF;
    for (int i = 0; i < K; i++) act[i] = (int8_t)((xorshift() % 256) - 128);
    
    int iter = 7;
    int k_start = iter * 32;
    int w_start = iter * 16;
    
    printf("Iteration 7 (k=%d..%d):\n", k_start, k_start + 31);
    
    /* Scalar calculation */
    int scalar_sum = 0;
    for (int i = 0; i < 16; i++) {
        int w0 = (wgt[w_start + i] & 0xF) - 8;
        int w1 = ((wgt[w_start + i] >> 4) & 0xF) - 8;
        int a0 = act[k_start + 2*i];
        int a1 = act[k_start + 2*i + 1];
        scalar_sum += w0 * a0 + w1 * a1;
    }
    printf("Scalar sum: %d\n\n", scalar_sum);
    
    /* NEON path */
    const uint8x16_t mask_0F = vdupq_n_u8(0x0F);
    const uint8x16_t val_08 = vdupq_n_u8(8);
    
    uint8x16_t w_packed = vld1q_u8(wgt + w_start);
    int8x16x2_t a_interleaved = vld2q_s8(act + k_start);
    
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
    
    /* Store to arrays for examination */
    uint8_t arr_w_even[16], arr_w_odd[16], arr_mag_even[16], arr_mag_odd[16];
    int8_t arr_a_even[16], arr_a_odd[16], arr_signed_ae[16], arr_signed_ao[16];
    
    vst1q_u8(arr_w_even, w_even);
    vst1q_u8(arr_w_odd, w_odd);
    vst1q_u8(arr_mag_even, mag_even);
    vst1q_u8(arr_mag_odd, mag_odd);
    vst1q_s8(arr_a_even, a_even);
    vst1q_s8(arr_a_odd, a_odd);
    vst1q_s8(arr_signed_ae, signed_a_even);
    vst1q_s8(arr_signed_ao, signed_a_odd);
    
    printf("NEON element check:\n");
    for (int i = 0; i < 16; i++) {
        int w_e = arr_w_even[i];
        int w_o = arr_w_odd[i];
        int a_e = arr_a_even[i];
        int a_o = arr_a_odd[i];
        int signed_ae = arr_signed_ae[i];
        int signed_ao = arr_signed_ao[i];
        int mag_e = arr_mag_even[i];
        int mag_o = arr_mag_odd[i];
        
        /* Scalar expected */
        int scalar_w_e = w_e - 8;
        int scalar_w_o = w_o - 8;
        int expected_e = scalar_w_e * a_e;
        int expected_o = scalar_w_o * a_o;
        
        /* NEON approach: signed_a * mag */
        int neon_e = signed_ae * mag_e;
        int neon_o = signed_ao * mag_o;
        
        if (expected_e != neon_e || expected_o != neon_o) {
            printf("  i=%d: MISMATCH!\n", i);
            printf("    w_e=%d (c=%d), a_e=%d, signed=%d, mag=%d\n", w_e, scalar_w_e, a_e, signed_ae, mag_e);
            printf("    expected=%d, neon=%d\n", expected_e, neon_e);
            printf("    w_o=%d (c=%d), a_o=%d, signed=%d, mag=%d\n", w_o, scalar_w_o, a_o, signed_ao, mag_o);
            printf("    expected=%d, neon=%d\n", expected_o, neon_o);
        }
    }
    
    return 0;
}
