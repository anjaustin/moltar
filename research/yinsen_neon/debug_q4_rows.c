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
    int N = 4;
    int K = 32;  /* Minimum for NEON */
    int K_packed = K / 2;
    
    printf("Testing N=%d, K=%d\n", N, K);
    
    uint8_t* wgt = malloc(N * K_packed);
    int8_t* act = malloc(K);
    int32_t* out_mul = malloc(N * sizeof(int32_t));
    int32_t* out_neon = malloc(N * sizeof(int32_t));
    
    for (int i = 0; i < N * K_packed; i++) wgt[i] = xorshift() & 0xFF;
    for (int i = 0; i < K; i++) act[i] = (int8_t)((xorshift() % 256) - 128);
    
    q4_matvec_multiply_ref(out_mul, act, wgt, N, K);
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(out_neon, act, wgt, N, K);
    
    for (int n = 0; n < N; n++) {
        if (out_mul[n] != out_neon[n]) {
            printf("Row %d: mul=%d, neon=%d  FAIL\n", n, out_mul[n], out_neon[n]);
        } else {
            printf("Row %d: %d OK\n", n, out_mul[n]);
        }
    }
#endif
    
    /* Now with K=64 */
    printf("\nTesting N=%d, K=64\n", N);
    K = 64;
    K_packed = K / 2;
    free(wgt); free(act); free(out_mul); free(out_neon);
    wgt = malloc(N * K_packed);
    act = malloc(K);
    out_mul = malloc(N * sizeof(int32_t));
    out_neon = malloc(N * sizeof(int32_t));
    
    rng = 42;
    for (int i = 0; i < N * K_packed; i++) wgt[i] = xorshift() & 0xFF;
    for (int i = 0; i < K; i++) act[i] = (int8_t)((xorshift() % 256) - 128);
    
    q4_matvec_multiply_ref(out_mul, act, wgt, N, K);
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(out_neon, act, wgt, N, K);
    
    for (int n = 0; n < N; n++) {
        if (out_mul[n] != out_neon[n]) {
            printf("Row %d: mul=%d, neon=%d  FAIL\n", n, out_mul[n], out_neon[n]);
        } else {
            printf("Row %d: %d OK\n", n, out_mul[n]);
        }
    }
#endif

    return 0;
}
