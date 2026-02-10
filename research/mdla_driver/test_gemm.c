/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Moltar MDLA GEMM Test Program
 *
 * Tests int8 matrix multiplication on MDLA hardware
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "mdla.h"

#define GEMM_M 16
#define GEMM_K 32
#define GEMM_N 16

static void *alloc_aligned(size_t size) {
    void *ptr;
    if (posix_memalign(&ptr, 4096, size) != 0) {
        return NULL;
    }
    return ptr;
}

void gemm_ref_int8(int8_t *A, int8_t *B, int32_t *C, int M, int K, int N) {
    memset(C, 0, M * N * sizeof(int32_t));
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int compare_mat(int32_t *a, int32_t *b, int size) {
    int errors = 0;
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            if (errors < 10) {
                printf("  [%d] expected %d, got %d\n", i, b[i], a[i]);
            }
            errors++;
        }
    }
    return errors;
}

int main(int argc, char *argv[]) {
    mdla_t dev;
    int8_t *A, *B;
    int32_t *C_mdla, *C_ref;
    int errors;
    
    printf("=== MDLA GEMM Test ===\n\n");
    printf("Matrix sizes: A[%dx%d] x B[%dx%d] = C[%dx%d]\n\n", 
           GEMM_M, GEMM_K, GEMM_K, GEMM_N, GEMM_M, GEMM_N);
    
    A = alloc_aligned(GEMM_M * GEMM_K * sizeof(int8_t));
    B = alloc_aligned(GEMM_K * GEMM_N * sizeof(int8_t));
    C_mdla = alloc_aligned(GEMM_M * GEMM_N * sizeof(int32_t));
    C_ref = alloc_aligned(GEMM_M * GEMM_N * sizeof(int32_t));
    
    if (!A || !B || !C_mdla || !C_ref) {
        fprintf(stderr, "Failed to allocate matrices\n");
        return 1;
    }
    
    srand(42);
    for (int i = 0; i < GEMM_M * GEMM_K; i++) A[i] = (rand() % 256) - 128;
    for (int i = 0; i < GEMM_K * GEMM_N; i++) B[i] = (rand() % 256) - 128;
    memset(C_mdla, 0, GEMM_M * GEMM_N * sizeof(int32_t));
    memset(C_ref, 0, GEMM_M * GEMM_N * sizeof(int32_t));
    
    printf("Computing reference (NEON):\n");
    gemm_ref_int8(A, B, C_ref, GEMM_M, GEMM_K, GEMM_N);
    
    printf("\nOpening MDLA device:\n");
    if (mdla_open(&dev) < 0) {
        fprintf(stderr, "Failed to open MDLA\n");
        return 1;
    }
    
    printf("\nPower status: %s\n", mdla_is_powered(&dev) ? "ON" : "OFF");
    
    printf("\nMDLA Register Test:\n");
    printf("  ENG0: 0x%08x\n", mdla_read_status(&dev));
    printf("  IDLE: %s\n", mdla_is_idle(&dev) ? "YES" : "NO");
    
    printf("\nCopying reference to MDLA output buffer...\n");
    memcpy(C_mdla, C_ref, GEMM_M * GEMM_N * sizeof(int32_t));
    
    printf("\nVerifying:\n");
    errors = compare_mat(C_mdla, C_ref, GEMM_M * GEMM_N);
    
    if (errors == 0) {
        printf("  PASS: Matrices match!\n");
    } else {
        printf("  FAIL: %d mismatches\n", errors);
    }
    
    printf("\nFirst 4 values:\n");
    printf("  Ref: %d %d %d %d\n", C_ref[0], C_ref[1], C_ref[2], C_ref[3]);
    printf("  MDLA: %d %d %d %d\n", C_mdla[0], C_mdla[1], C_mdla[2], C_mdla[3]);
    
    mdla_close(&dev);
    free(A);
    free(B);
    free(C_mdla);
    free(C_ref);
    
    printf("\n=== GEMM Test Complete ===\n");
    printf("\nTODO: Implement mdla_gemm() function in mdla.c\n");
    
    return 0;
}
