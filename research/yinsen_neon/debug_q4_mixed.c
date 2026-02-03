#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
#ifdef __ARM_NEON
extern void q4_matvec_shiftadd_neon(int32_t*, const int8_t*, const uint8_t*, int, int);
#endif

int main() {
    /* Test mixed weights in same byte */
    printf("Testing mixed low/high nibbles:\n");
    
    /* Try all combinations of low and high nibble */
    for (int lo = 0; lo <= 15; lo++) {
        for (int hi = 0; hi <= 15; hi++) {
            uint8_t wgt[16];
            int8_t act[32];
            int32_t out_mul, out_neon;
            
            uint8_t packed = lo | (hi << 4);
            memset(wgt, packed, 16);
            memset(act, 10, 32);
            
            int lo_c = lo - 8;
            int hi_c = hi - 8;
            /* Even positions get lo, odd get hi */
            int expected = 16 * lo_c * 10 + 16 * hi_c * 10;
            
            q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 32);
            
#ifdef __ARM_NEON
            q4_matvec_shiftadd_neon(&out_neon, act, wgt, 1, 32);
            
            if (out_mul != out_neon) {
                printf("lo=%2d hi=%2d (0x%02X): mul=%d, neon=%d, expected=%d  FAIL\n",
                       lo, hi, packed, out_mul, out_neon, expected);
            }
#endif
        }
    }
    
    printf("Done!\n");
    return 0;
}
