#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
#ifdef __ARM_NEON
extern void q4_matvec_shiftadd_neon(int32_t*, const int8_t*, const uint8_t*, int, int);
#endif

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
    /* Test progressively larger K */
    int N = 1;
    int Ks[] = {32, 64, 128, 256, 512, 1024};
    
    for (int ki = 0; ki < 6; ki++) {
        int K = Ks[ki];
        int K_packed = K / 2;
        
        uint8_t* wgt = malloc(N * K_packed);
        int8_t* act = malloc(K);
        int32_t out_mul, out_neon;
        
        rng = 42;  /* Reset for consistency */
        for (int i = 0; i < N * K_packed; i++) wgt[i] = xorshift() & 0xFF;
        for (int i = 0; i < K; i++) act[i] = (int8_t)((xorshift() % 256) - 128);
        
        q4_matvec_multiply_ref(&out_mul, act, wgt, N, K);
#ifdef __ARM_NEON
        q4_matvec_shiftadd_neon(&out_neon, act, wgt, N, K);
        
        if (out_mul != out_neon) {
            printf("K=%4d: mul=%d, neon=%d  FAIL (diff=%d)\n", K, out_mul, out_neon, out_neon - out_mul);
        } else {
            printf("K=%4d: %d OK\n", K, out_mul);
        }
#endif
        
        free(wgt);
        free(act);
    }
    
    return 0;
}
