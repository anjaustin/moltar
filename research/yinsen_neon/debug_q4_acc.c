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

/* Compute scalar for one iteration to verify */
int scalar_iter(const uint8_t* wgt, const int8_t* act) {
    int sum = 0;
    for (int i = 0; i < 16; i++) {
        int w0 = (wgt[i] & 0xF) - 8;
        int w1 = ((wgt[i] >> 4) & 0xF) - 8;
        sum += w0 * act[2*i];
        sum += w1 * act[2*i + 1];
    }
    return sum;
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
    printf("Reference total: %d\n\n", out_mul);
    
    /* Check each iteration */
    int total = 0;
    for (int iter = 0; iter < K / 32; iter++) {
        int iter_sum = scalar_iter(wgt + iter * 16, act + iter * 32);
        total += iter_sum;
        printf("Iter %d: scalar sum = %d, running total = %d\n", iter, iter_sum, total);
    }
    printf("\nFinal scalar total: %d\n", total);
    
    return 0;
}
