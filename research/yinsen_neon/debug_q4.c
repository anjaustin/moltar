#include <stdio.h>
#include <stdint.h>

extern void q4_matvec_shiftadd_ref(int32_t*, const int8_t*, const uint8_t*, int, int);
extern void q4_matvec_multiply_ref(int32_t*, const int8_t*, const uint8_t*, int, int);

int main() {
    /* Small test: 1 output, 4 inputs */
    uint8_t wgt[2] = {0x53, 0xA7};  /* 4 weights packed: 3,5,7,10 -> centered: -5,-3,-1,+2 */
    int8_t act[4] = {10, 20, 30, 40};
    int32_t out_mul, out_shift;
    
    printf("Weights (raw): %d, %d, %d, %d\n", 
           wgt[0] & 0xF, (wgt[0] >> 4) & 0xF, 
           wgt[1] & 0xF, (wgt[1] >> 4) & 0xF);
    printf("Weights (centered): %d, %d, %d, %d\n",
           (wgt[0] & 0xF) - 8, ((wgt[0] >> 4) & 0xF) - 8,
           (wgt[1] & 0xF) - 8, ((wgt[1] >> 4) & 0xF) - 8);
    printf("Activations: %d, %d, %d, %d\n", act[0], act[1], act[2], act[3]);
    
    printf("\nExpected: (%d)*%d + (%d)*%d + (%d)*%d + (%d)*%d = %d\n",
           (wgt[0] & 0xF) - 8, act[0],
           ((wgt[0] >> 4) & 0xF) - 8, act[1],
           (wgt[1] & 0xF) - 8, act[2],
           ((wgt[1] >> 4) & 0xF) - 8, act[3],
           ((wgt[0] & 0xF) - 8) * act[0] +
           (((wgt[0] >> 4) & 0xF) - 8) * act[1] +
           ((wgt[1] & 0xF) - 8) * act[2] +
           (((wgt[1] >> 4) & 0xF) - 8) * act[3]);
    
    q4_matvec_multiply_ref(&out_mul, act, wgt, 1, 4);
    printf("Multiply ref: %d\n", out_mul);
    
    q4_matvec_shiftadd_ref(&out_shift, act, wgt, 1, 4);
    printf("Shift-add ref: %d\n", out_shift);
    
    return 0;
}
