#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void q4_matvec_shiftadd_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
#ifdef __ARM_NEON
extern void q4_matvec_shiftadd_neon(int32_t*, const int8_t*, const uint8_t*, int, int);
#endif

static uint32_t xorshift(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

int main() {
    int N = 4;
    int K = 1024;
    int K_packed = K / 2;
    
    uint8_t* wgt = (uint8_t*)malloc(N * K_packed);
    int8_t* act = (int8_t*)malloc(K);
    int32_t* out_mul = (int32_t*)malloc(N * sizeof(int32_t));
    int32_t* out_neon = (int32_t*)malloc(N * sizeof(int32_t));
    
    /* Same RNG as benchmark */
    uint32_t rng = 42;
    for (int i = 0; i < N * K_packed; i++) {
        wgt[i] = xorshift(&rng) & 0xFF;
    }
    for (int i = 0; i < K; i++) {
        act[i] = (int8_t)((xorshift(&rng) % 256) - 128);
    }
    
    printf("Testing N=%d, K=%d\n", N, K);
    
    q4_matvec_multiply_ref(out_mul, act, wgt, N, K);
    
#ifdef __ARM_NEON
    q4_matvec_shiftadd_neon(out_neon, act, wgt, N, K);
    
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (out_mul[i] != out_neon[i]) {
            printf("Row %d: expected %d, got %d (diff %d)\n", 
                   i, out_mul[i], out_neon[i], out_neon[i] - out_mul[i]);
            errors++;
        }
    }
    printf("Errors: %d/%d\n", errors, N);
#endif
    
    return 0;
}
