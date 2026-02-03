#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void q4_matvec_shiftadd_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
#ifdef __ARM_NEON
extern void q4_matvec_shiftadd_neon(int32_t*, const int8_t*, const uint8_t*, int, int);
#endif

int main() {
    /* Test with 64 elements (2 iterations of NEON kernel) */
    uint8_t wgt[32];  /* 32 bytes = 64 weights */
    int8_t act[64];
    int32_t out_mul, out_shift, out_neon;
    
    /* Simple pattern */
    for (int i = 0; i < 32; i++) wgt[i] = 0x55;  /* All 5 -> -3 centered */
    for (int i = 0; i < 64; i++) act[i] = (int8_t)(i + 1);
    
    printf("Test: weights all 5 (-3), acts 1..64\n");
    printf("Expected: sum of (-3) * (1 + 2 + ... + 64) = -3 * 2080 = -6240\n");
    
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 64);
    printf("Multiply ref: %d\n", out_mul);
    
    q4_matvec_shiftadd_ref(&out_shift, act, wgt, 1, 64);
    printf("Shift-add ref: %d\n", out_shift);
    
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 64);
    printf("NEON shift-add: %d\n", out_neon);
#endif
    
    /* Test with 1024 elements */
    printf("\nTest: 1024 elements, simple pattern\n");
    uint8_t* wgt2 = (uint8_t*)malloc(512);
    int8_t* act2 = (int8_t*)malloc(1024);
    
    for (int i = 0; i < 512; i++) wgt2[i] = 0xAA;  /* All 10 -> +2 centered */
    for (int i = 0; i < 1024; i++) act2[i] = 1;
    
    printf("Expected: 1024 * (2 * 1) = 2048\n");
    
    q4_matvec_multiply_ref(&out_mul, act2, wgt2, 1, 1024);
    printf("Multiply ref: %d\n", out_mul);
    
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act2, wgt2, 1, 1024);
    printf("NEON shift-add: %d\n", out_neon);
#endif

    /* Test: varying weights */
    printf("\nTest: varying weights\n");
    for (int i = 0; i < 512; i++) wgt2[i] = i % 16 | ((i % 16) << 4);
    for (int i = 0; i < 1024; i++) act2[i] = 1;
    
    q4_matvec_multiply_ref(&out_mul, act2, wgt2, 1, 1024);
    printf("Multiply ref: %d\n", out_mul);
    
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(&out_neon, act2, wgt2, 1, 1024);
    printf("NEON shift-add: %d\n", out_neon);
#endif
    
    return 0;
}
