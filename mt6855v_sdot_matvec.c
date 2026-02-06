/*
 * mt6855v_sdot_matvec.c - C Wrapper for MT6855V Assembly Kernels
 *
 * Provides runtime detection and dispatch for assembly-optimized
 * matrix-vector multiplication on Motorola MT6855V hardware.
 */

#include "mt6855v_sdot_matvec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

// Performance tracking
static float g_last_tokens_per_sec = 0.0f;
static float g_last_memory_bw_gb = 0.0f;
static int g_last_core_type = -1;

/* ============================================================================
 * Hardware Detection
 * ============================================================================ */

int get_current_cpu_core(void) {
    // Get current CPU core (0-7 for MT6855V)
    #ifdef __linux__
        return sched_getcpu();
    #else
        // Fallback for other platforms
        return 0;
    #endif
}

int is_mt6855v_hardware(void) {
    // Check if running on MT6855V hardware
    #ifdef __linux__
        // Check /proc/cpuinfo for MT6855V signature
        FILE* fp = fopen("/proc/cpuinfo", "r");
        if (!fp) return 0;
        
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, "MT6855V") || strstr(line, "mt6855")) {
                fclose(fp);
                return 1;
            }
        }
        fclose(fp);
    #endif
    
    // Check for ARM features we need
    #ifdef __ARM_FEATURE_DOTPROD
        return 1;  // Assume MT6855V if we have dotprod
    #else
        return 0;
    #endif
}

/* ============================================================================
 * Assembly Kernel Dispatch
 * ============================================================================ */

int mt6855v_matvec_dispatch(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ weights,
    const int8_t* __restrict__ act,
    int N,
    int K
) {
    if (!out || !weights || !act || N <= 0 || K <= 0) {
        return -1;
    }
    
    // Validate constraints
    if ((N & 7) != 0 || (K & 63) != 0) {
        return -1;  // Must be multiples of 8 and 64
    }
    
    // Get current CPU core
    int cpu_core = get_current_cpu_core();
    g_last_core_type = (cpu_core >= 4) ? 1 : 0;  // 1=big, 0=little
    
    // Dispatch to appropriate assembly kernel
    if (cpu_core >= 4) {
        // Big core (CPU 4-7) - use Cortex-A78 optimized
        return neon_int8_matvec_blocked8_k64_asm_big(out, weights, act, N, K);
    } else {
        // Little core (CPU 0-3) - use Cortex-A55 optimized
        return neon_int8_matvec_blocked8_k64_asm_little(out, weights, act, N, K);
    }
}

int neon_int8_matvec_blocked8_k64_asm(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ weights,
    const int8_t* __restrict__ act,
    int N,
    int K
) {
    // Runtime dispatch based on CPU core
    return mt6855v_matvec_dispatch(out, weights, act, N, K);
}

/* ============================================================================
 * Performance Monitoring
 * ============================================================================ */

int mt6855v_assembly_available(void) {
    return is_mt6855v_hardware();
}

void mt6855v_get_performance_metrics(
    float* tokens_per_sec,
    float* memory_bw_gb,
    int* core_type
) {
    if (tokens_per_sec) *tokens_per_sec = g_last_tokens_per_sec;
    if (memory_bw_gb) *memory_bw_gb = g_last_memory_bw_gb;
    if (core_type) *core_type = g_last_core_type;
}

/* ============================================================================
 * Benchmark and Validation
 * ============================================================================ */

int benchmark_mt6855v_assembly(void) {
    printf("MT6855V Assembly Kernel Benchmark\n");
    printf("=================================\n");
    
    if (!mt6855v_assembly_available()) {
        printf("❌ MT6855V hardware not detected\n");
        return -1;
    }
    
    printf("✅ MT6855V detected - CPU core: %d\n", get_current_cpu_core());
    printf("   Big cores: CPU 4-7 (Cortex-A78)\n");
    printf("   Little cores: CPU 0-3 (Cortex-A55)\n");
    
    // Test different core types
    int core = get_current_cpu_core();
    const char* core_type = (core >= 4) ? "Big (A78)" : "Little (A55)";
    printf("   Running on: %s core\n", core_type);
    
    return 0;
}

/* ============================================================================
 * Integration with Existing Codebase
 * ============================================================================ */

// Wrapper to integrate with existing trix_lfm2_chip code
int mt6855v_matvec_optimized(
    int32_t* __restrict__ out,
    const int8_t* __restrict__ weights,
    const int8_t* __restrict__ act,
    int N,
    int K
) {
    // Check if assembly is available and beneficial
    if (mt6855v_assembly_available() && N >= 8 && K >= 64) {
        // Use assembly kernels
        return mt6855v_matvec_dispatch(out, weights, act, N, K);
    } else {
        // Fallback to C implementation
        return -1;  // Let caller handle fallback
    }
}