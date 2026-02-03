#include <stdio.h>
#include <stdint.h>
#include <string.h>

extern void q4_matvec_shiftadd_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
#ifdef __ARM_NEON
extern void q4_matvec_shiftadd_neon(int32_t*, const int8_t*, const uint8_t*, int, int);
#endif

int main() {
    /* Test with exactly 32 elements (minimum for NEON kernel) */
    uint8_t wgt[16];  /* 16 bytes = 32 weights */
    int8_t act[32];
    int32_t out_mul, out_shift, out_neon;
    
    /* Initialize: all weights = 9 (centered = +1), all acts = 1 */
    memset(wgt, 0x99, 16);  /* 9 in both nibbles */
    memset(act, 1, 32);
    
    printf("Test 1: all weights=9 (+1 centered), all acts=1\n");
    printf("Expected: 32 * (1 * 1) = 32\n");
    
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
    printf("Multiply ref: %d\n", out_mul);
    
    q4_matvec_shiftadd_ref(&out_shift, act, wgt, 1, 32);
    printf("Shift-add ref: %d\n", out_shift);
    
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
    printf("NEON shift-add: %d\n", out_neon);
#endif
    
    /* Test 2: weights = 0xAA (10,10 = +2,+2 centered), acts = 5 */
    printf("\nTest 2: all weights=10 (+2 centered), all acts=5\n");
    memset(wgt, 0xAA, 16);
    memset(act, 5, 32);
    printf("Expected: 32 * (2 * 5) = 320\n");
    
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
    printf("Multiply ref: %d\n", out_mul);
    
    q4_matvec_shiftadd_ref(&out_shift, act, wgt, 1, 32);
    printf("Shift-add ref: %d\n", out_shift);
    
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
    printf("NEON shift-add: %d\n", out_neon);
#endif

    /* Test 3: weights = 0x55 (5,5 = -3,-3 centered), acts = 10 */
    printf("\nTest 3: all weights=5 (-3 centered), all acts=10\n");
    memset(wgt, 0x55, 16);
    memset(act, 10, 32);
    printf("Expected: 32 * (-3 * 10) = -960\n");
    
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
    printf("Multiply ref: %d\n", out_mul);
    
    q4_matvec_shiftadd_ref(&out_shift, act, wgt, 1, 32);
    printf("Shift-add ref: %d\n", out_shift);
    
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
    printf("NEON shift-add: %d\n", out_neon);
#endif
    
    return 0;
}
