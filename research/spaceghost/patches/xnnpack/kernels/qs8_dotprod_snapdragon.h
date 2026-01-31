#ifndef XNNPACK_QS8_DOTPROD_SNAPDRAGON_H
#define XNNPACK_QS8_DOTPROD_SNAPDRAGON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Snapdragon 480 optimized kernels using dot product instructions
// These kernels leverage UDOT/SDOT instructions available in ARMv8.2-A +dotprod

/**
 * GEMM microkernel for Snapdragon 480 with dot product acceleration
 *
 * Optimized for Cortex-A76 microarchitecture with:
 * - UDOT/SDOT instructions for quantized operations
 * - Optimal register allocation
 * - L3 cache prefetching
 * - Big core (A76) specific optimizations
 */
void xnn_qs8_gemm_minmax_ukernel_1x8__snapdragon480_dotprod_ld128(
    size_t mr, size_t nr, size_t k,
    const int8_t* a, size_t a_stride,
    const int8_t* w, size_t w_stride,
    const float* bias, float* c, size_t c_stride,
    const union xnn_qs8_conv_minmax_params params[restrict static 1]);

void xnn_qs8_gemm_minmax_ukernel_2x8__snapdragon480_dotprod_ld128(
    size_t mr, size_t nr, size_t k,
    const int8_t* a, size_t a_stride,
    const int8_t* w, size_t w_stride,
    const float* bias, float* c, size_t c_stride,
    const union xnn_qs8_conv_minmax_params params[restrict static 1]);

void xnn_qs8_gemm_minmax_ukernel_4x8__snapdragon480_dotprod_ld128(
    size_t mr, size_t nr, size_t k,
    const int8_t* a, size_t a_stride,
    const int8_t* w, size_t w_stride,
    const float* bias, float* c, size_t c_stride,
    const union xnn_qs8_conv_minmax_params params[restrict static 1]);

/**
 * Convolution microkernels optimized for Snapdragon 480
 */
void xnn_qs8_conv_minmax_rndnu_ukernel_1x8__snapdragon480_dotprod(
    size_t mr, size_t nr, size_t kc, size_t ks,
    const int8_t** a, const void* w, float* c, size_t cm_stride, size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict static 1]);

void xnn_qs8_conv_minmax_rndnu_ukernel_2x8__snapdragon480_dotprod(
    size_t mr, size_t nr, size_t kc, size_t ks,
    const int8_t** a, const void* w, float* c, size_t cm_stride, size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict static 1]);

/**
 * Depthwise convolution kernels for Snapdragon 480
 */
void xnn_qs8_dwconv_minmax_rndnu_ukernel_4x8__snapdragon480_dotprod(
    size_t mr, size_t nr, size_t kc, size_t ks,
    const int8_t** a, const void* w, float* c, size_t cm_stride, size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict static 1]);

void xnn_qs8_dwconv_minmax_rndnu_ukernel_8x8__snapdragon480_dotprod(
    size_t mr, size_t nr, size_t kc, size_t ks,
    const int8_t** a, const void* w, float* c, size_t cm_stride, size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict static 1]);

/**
 * Global average pooling optimized for Snapdragon 480
 */
void xnn_qs8_gavgpool_minmax_rndnu_ukernel_7x__snapdragon480_dotprod(
    size_t mr, size_t nr, size_t kc,
    const int8_t* a, const int8_t* zero, const int8_t* multiplier,
    int8_t* c, size_t c_stride,
    const union xnn_qs8_conv_minmax_params params[restrict static 1]);

void xnn_qs8_gavgpool_minmax_rndnu_ukernel_7p7x__snapdragon480_dotprod(
    size_t mr, size_t nr, size_t kc,
    const int8_t* a, const int8_t* zero, const int8_t* multiplier,
    int8_t* c, size_t c_stride,
    const union xnn_qs8_conv_minmax_params params[restrict static 1]);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // XNNPACK_QS8_DOTPROD_SNAPDRAGON_H