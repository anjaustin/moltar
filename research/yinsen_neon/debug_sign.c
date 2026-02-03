#include <stdio.h>
#include <stdint.h>
#include <arm_neon.h>

int main() {
    /* Test sign handling */
    int8_t a = 10;
    uint8_t w = 5;  /* centered = -3 */
    uint8_t val_8 = 8;
    
    printf("a = %d, w = %d, centered = %d\n", a, w, (int)w - 8);
    printf("Expected: %d * %d = %d\n", (int)w - 8, a, ((int)w - 8) * a);
    
    /* Check vclt */
    uint8x8_t w_vec = vdup_n_u8(w);
    uint8x8_t val8_vec = vdup_n_u8(8);
    uint8x8_t sign = vclt_u8(w_vec, val8_vec);
    printf("vclt(5, 8) mask: 0x%02x (expect 0xFF)\n", vget_lane_u8(sign, 0));
    
    /* Check magnitude */
    uint8x8_t mag = vabd_u8(w_vec, val8_vec);
    printf("|5 - 8| = %d (expect 3)\n", vget_lane_u8(mag, 0));
    
    /* Check negation */
    int8x8_t a_vec = vdup_n_s8(a);
    int8x8_t neg_a = vneg_s8(a_vec);
    printf("neg(10) = %d (expect -10)\n", vget_lane_s8(neg_a, 0));
    
    /* Check vbsl - sign is 0xFF for negative */
    int8x8_t signed_a = vbsl_s8(sign, neg_a, a_vec);
    printf("vbsl(0xFF, -10, 10) = %d (expect -10)\n", vget_lane_s8(signed_a, 0));
    
    /* Now the shift-add part */
    int16x8_t a16 = vmovl_s8(signed_a);
    printf("After vmovl: %d\n", vgetq_lane_s16(a16, 0));
    
    /* Magnitude = 3 = 0b011, bits: b0=1, b1=1, b2=0, b3=0 */
    uint8x8_t b0_mask = vtst_u8(mag, vdup_n_u8(1));
    uint8x8_t b1_mask = vtst_u8(mag, vdup_n_u8(2));
    printf("bit0 mask: 0x%02x (expect 0xFF)\n", vget_lane_u8(b0_mask, 0));
    printf("bit1 mask: 0x%02x (expect 0xFF)\n", vget_lane_u8(b1_mask, 0));
    
    /* Expand to 16-bit */
    uint16x8_t b0_16 = vmovl_u8(b0_mask);
    uint16x8_t b1_16 = vmovl_u8(b1_mask);
    printf("b0_16: 0x%04x\n", vgetq_lane_u16(b0_16, 0));
    printf("b1_16: 0x%04x\n", vgetq_lane_u16(b1_16, 0));
    
    /* This is the problem! vmovl_u8 zero-extends, so 0xFF -> 0x00FF, not 0xFFFF */
    /* We need the full mask! */
    
    /* Try vcgt to create proper mask */
    uint16x8_t proper_mask = vcgtq_u16(b0_16, vdupq_n_u16(0));
    printf("proper mask: 0x%04x (expect 0xFFFF)\n", vgetq_lane_u16(proper_mask, 0));
    
    /* Product with proper mask */
    int16x8_t prod = vandq_s16(a16, vreinterpretq_s16_u16(proper_mask));
    printf("(-10) & 0xFFFF = %d (expect -10)\n", vgetq_lane_s16(prod, 0));
    
    /* a*2 with shift */
    int16x8_t a2 = vshlq_n_s16(a16, 1);
    printf("(-10) << 1 = %d (expect -20)\n", vgetq_lane_s16(a2, 0));
    
    /* a*3 = a + a*2 */
    int16x8_t sum = vaddq_s16(prod, vandq_s16(a2, vreinterpretq_s16_u16(vcgtq_u16(b1_16, vdupq_n_u16(0)))));
    printf("Result: %d (expect -30)\n", vgetq_lane_s16(sum, 0));
    
    return 0;
}
