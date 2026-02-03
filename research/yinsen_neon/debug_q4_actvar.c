#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
#ifdef __ARM_NEON
extern void q4_matvec_shiftadd_neon(int32_t*, const int8_t*, const uint8_t*, int, int);
#endif

int main() {
    /* Test varying activations */
    printf("Testing varying activations:\n");
    
    uint8_t wgt[16];  /* 32 weights */
    int8_t act[32];
    int32_t out_mul, out_neon;
    
    /* Weight = 8 (centered = 0) - should give 0 */
    memset(wgt, 0x88, 16);
    for (int i = 0; i < 32; i++) act[i] = i - 16;  /* -16 to +15 */
    
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
    printf("w=8 (0): mul=%d, neon=%d %s\n", out_mul, out_neon, out_mul == out_neon ? "OK" : "FAIL");
#endif

    /* Weight = 9 (centered = +1) */
    memset(wgt, 0x99, 16);
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
    printf("w=9 (+1): mul=%d, neon=%d %s\n", out_mul, out_neon, out_mul == out_neon ? "OK" : "FAIL");
#endif

    /* Weight = 7 (centered = -1) */
    memset(wgt, 0x77, 16);
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
    printf("w=7 (-1): mul=%d, neon=%d %s\n", out_mul, out_neon, out_mul == out_neon ? "OK" : "FAIL");
#endif

    /* Random-ish activations */
    static uint32_t rng = 42;
    for (int i = 0; i < 32; i++) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        act[i] = (int8_t)((rng % 256) - 128);
    }
    
    memset(wgt, 0x55, 16);  /* w=5, centered=-3 */
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
    printf("w=5 (-3), random acts: mul=%d, neon=%d %s\n", out_mul, out_neon, out_mul == out_neon ? "OK" : "FAIL");
#endif

    printf("\nNow with random weights AND random activations:\n");
    for (int i = 0; i < 16; i++) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        wgt[i] = rng & 0xFF;
    }
    for (int i = 0; i < 32; i++) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        act[i] = (int8_t)((rng % 256) - 128);
    }
    
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
    printf("random all: mul=%d, neon=%d %s\n", out_mul, out_neon, out_mul == out_neon ? "OK" : "FAIL");
#endif
    
    return 0;
}
