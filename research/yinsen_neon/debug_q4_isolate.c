#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
#ifdef __ARM_NEON
extern void q4_matvec_shiftadd_neon(int32_t*, const int8_t*, const uint8_t*, int, int);
#endif

int main() {
    /* Test each weight value 0-15 individually */
    printf("Testing each weight value with act=10:\n");
    
    for (int w = 0; w <= 15; w++) {
        uint8_t wgt[16];  /* 32 weights */
        int8_t act[32];
        int32_t out_mul, out_neon;
        
        memset(wgt, w | (w << 4), 16);  /* All same weight */
        memset(act, 10, 32);
        
        int centered = w - 8;
        int expected = 32 * centered * 10;
        
        q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
        
#ifdef __ARM_NEON
        q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
        
        if (out_mul != out_neon) {
            printf("w=%2d (centered=%3d): expected=%6d, mul=%6d, neon=%6d  FAIL\n",
                   w, centered, expected, out_mul, out_neon);
        }
#endif
    }
    
    printf("\nNow with act=-10:\n");
    for (int w = 0; w <= 15; w++) {
        uint8_t wgt[16];
        int8_t act[32];
        int32_t out_mul, out_neon;
        
        memset(wgt, w | (w << 4), 16);
        memset(act, -10, 32);
        
        int centered = w - 8;
        int expected = 32 * centered * (-10);
        
        q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
        
#ifdef __ARM_NEON
        q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
        
        if (out_mul != out_neon) {
            printf("w=%2d (centered=%3d): expected=%6d, mul=%6d, neon=%6d  FAIL\n",
                   w, centered, expected, out_mul, out_neon);
        }
#endif
    }
    
    printf("Done!\n");
    return 0;
}
