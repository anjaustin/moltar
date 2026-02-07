/*
 * mt6855v_matmul_shift.c - Shift-Register MatMul Integration
 *
 * Integrates hand-tuned ARM assembly shift-register MatMul operations
 * into existing llama.cpp build system for MT6855V/Dimensity 930.
 *
 * Target: 2-3x faster MatMul operations (10ms/token -> 3.4ms/token)
 * Optimizations: SMLAL/SMULL shift-register adds instead of MUL+ADD sequence
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>

// Function pointer types
typedef int (*matmul_shift_func_t)(
    int16_t* out, const int8_t* weights, const int8_t* act, int n_out, int n_in
);

typedef struct {
    matmul_shift_func_t matmul_shift;
    int is_available;
    void* handle;
} matmul_shift_ctx_t;

static matmul_shift_ctx_t g_shift_ctx = {0};
static int g_shift_initialized = 0;

int matmul_shift_init(void) {
    const char* lib_paths[] = {
        "./mt6855v_matmul_shift.so",
        "/data/local/tmp/mt6855v_matmul_shift.so",
        NULL
    };
    
    for (int i = 0; lib_paths[i] != NULL; i++) {
        g_shift_ctx.handle = dlopen(lib_paths[i], RTLD_LAZY);
        if (g_shift_ctx.handle) {
            g_shift_ctx.matmul_shift = (matmul_shift_func_t)dlsym(g_shift_ctx.handle, "mt6855v_matmul_shift_register");
            if (g_shift_ctx.matmul_shift) {
                g_shift_ctx.is_available = 1;
                dlclose(g_shift_ctx.handle);
                return 0;
            }
        }
    }
    
    return -1;
}

int matmul_shift_available(void) {
    return g_shift_ctx.is_available;
}

int matmul_shift_optimized(
    int16_t* out, const int8_t* weights, const int8_t* act, 
    int n_out, int n_in
) {
    static int initialized = 0;
    
    if (!initialized) {
        matmul_shift_init();
        initialized = 1;
    }
    
    if (matmul_shift_available() && (n_out & 31) == 0 && (n_in & 63) == 0) {
        return g_shift_ctx.matmul_shift(out, weights, act, n_out, n_in);
    }
    
    return -1;
}

void matmul_shift_cleanup(void) {
    if (g_shift_ctx.handle) {
        dlclose(g_shift_ctx.handle);
        g_shift_ctx.handle = NULL;
    }
}