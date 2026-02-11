/*
 * gpu_bench.c - OpenCL GPU compute benchmark
 * Tests: SGEMM (FP32), HGEMM (FP16), vector add, reduction
 * Cross-compile with Android NDK, run on device
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* ---- Minimal OpenCL typedefs ---- */
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef intptr_t cl_platform_id;
typedef intptr_t cl_device_id;
typedef void*    cl_context;
typedef void*    cl_command_queue;
typedef void*    cl_program;
typedef void*    cl_kernel;
typedef void*    cl_mem;
typedef void*    cl_event;
typedef uint64_t cl_device_type;
typedef uint64_t cl_mem_flags;
typedef uint32_t cl_bool;
typedef uint64_t cl_command_queue_properties;
typedef uint32_t cl_profiling_info;
typedef uint64_t cl_bitfield;
typedef uint32_t cl_context_properties_tag;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL          0xFFFFFFFF
#define CL_DEVICE_TYPE_GPU          (1 << 2)
#define CL_MEM_READ_WRITE           (1 << 0)
#define CL_MEM_WRITE_ONLY           (1 << 1)
#define CL_MEM_READ_ONLY            (1 << 2)
#define CL_MEM_COPY_HOST_PTR        (1 << 5)
#define CL_QUEUE_PROFILING_ENABLE   (1 << 1)
#define CL_PROFILING_COMMAND_START  0x1282
#define CL_PROFILING_COMMAND_END    0x1283
#define CL_PROGRAM_BUILD_LOG        0x1183
#define CL_TRUE                     1
#define CL_FALSE                    0

/* Context properties */
#define CL_CONTEXT_PLATFORM         0x1084

/* Function pointer types */
typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*pfn_clCreateContext)(const intptr_t*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*pfn_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_program (*pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_kernel (*pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (*pfn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (*pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*pfn_clFinish)(cl_command_queue);
typedef cl_int (*pfn_clWaitForEvents)(cl_uint, const cl_event*);
typedef cl_int (*pfn_clGetEventProfilingInfo)(cl_event, cl_profiling_info, size_t, void*, size_t*);
typedef cl_int (*pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (*pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (*pfn_clReleaseProgram)(cl_program);
typedef cl_int (*pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*pfn_clReleaseContext)(cl_context);
typedef cl_int (*pfn_clReleaseEvent)(cl_event);
typedef cl_int (*pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);

/* Global function pointers */
static pfn_clGetPlatformIDs          fn_clGetPlatformIDs;
static pfn_clGetDeviceIDs            fn_clGetDeviceIDs;
static pfn_clCreateContext           fn_clCreateContext;
static pfn_clCreateCommandQueue      fn_clCreateCommandQueue;
static pfn_clCreateProgramWithSource fn_clCreateProgramWithSource;
static pfn_clBuildProgram            fn_clBuildProgram;
static pfn_clCreateKernel            fn_clCreateKernel;
static pfn_clCreateBuffer            fn_clCreateBuffer;
static pfn_clSetKernelArg            fn_clSetKernelArg;
static pfn_clEnqueueNDRangeKernel    fn_clEnqueueNDRangeKernel;
static pfn_clEnqueueReadBuffer       fn_clEnqueueReadBuffer;
static pfn_clEnqueueWriteBuffer      fn_clEnqueueWriteBuffer;
static pfn_clFinish                  fn_clFinish;
static pfn_clWaitForEvents           fn_clWaitForEvents;
static pfn_clGetEventProfilingInfo   fn_clGetEventProfilingInfo;
static pfn_clReleaseMemObject        fn_clReleaseMemObject;
static pfn_clReleaseKernel           fn_clReleaseKernel;
static pfn_clReleaseProgram          fn_clReleaseProgram;
static pfn_clReleaseCommandQueue     fn_clReleaseCommandQueue;
static pfn_clReleaseContext          fn_clReleaseContext;
static pfn_clReleaseEvent            fn_clReleaseEvent;
static pfn_clGetProgramBuildInfo     fn_clGetProgramBuildInfo;

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static double get_event_time_ms(cl_event ev) {
    cl_ulong start, end;
    fn_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    fn_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    return (end - start) / 1000000.0;
}

static void print_build_log(cl_program prog, cl_device_id dev) {
    size_t log_size;
    fn_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 1) {
        char *log = malloc(log_size + 1);
        fn_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        printf("  Build log:\n%s\n", log);
        free(log);
    }
}

/* ---- OpenCL Kernel Sources ---- */

/* SGEMM: C = A * B, naive but tests raw throughput */
static const char *sgemm_src =
"__kernel void sgemm(\n"
"    __global const float* A, __global const float* B, __global float* C,\n"
"    const int M, const int N, const int K) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += A[row * K + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n";

/* Tiled SGEMM with local memory */
static const char *sgemm_tiled_src =
"#define TS 16\n"
"__kernel void sgemm_tiled(\n"
"    __global const float* A, __global const float* B, __global float* C,\n"
"    const int M, const int N, const int K) {\n"
"    __local float Asub[TS][TS];\n"
"    __local float Bsub[TS][TS];\n"
"    int row = get_local_id(0);\n"
"    int col = get_local_id(1);\n"
"    int globalRow = TS * get_group_id(0) + row;\n"
"    int globalCol = TS * get_group_id(1) + col;\n"
"    float sum = 0.0f;\n"
"    int numTiles = (K + TS - 1) / TS;\n"
"    for (int t = 0; t < numTiles; t++) {\n"
"        int tiledRow = TS * t + row;\n"
"        int tiledCol = TS * t + col;\n"
"        Asub[row][col] = (globalRow < M && tiledCol < K) ? A[globalRow * K + tiledCol] : 0.0f;\n"
"        Bsub[row][col] = (tiledRow < N && globalCol < K) ? B[tiledRow * N + globalCol] : 0.0f;\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        for (int k = 0; k < TS; k++) {\n"
"            sum += Asub[row][k] * Bsub[k][col];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (globalRow < M && globalCol < N) {\n"
"        C[globalRow * N + globalCol] = sum;\n"
"    }\n"
"}\n";

/* FP16 SGEMM using half precision */
static const char *hgemm_src =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void hgemm(\n"
"    __global const half* A, __global const half* B, __global half* C,\n"
"    const int M, const int N, const int K) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    if (row < M && col < N) {\n"
"        half sum = (half)0.0f;\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += A[row * K + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n";

/* Vector add (bandwidth test) */
static const char *vecadd_src =
"__kernel void vecadd(\n"
"    __global const float* A, __global const float* B, __global float* C, const int N) {\n"
"    int i = get_global_id(0);\n"
"    if (i < N) C[i] = A[i] + B[i];\n"
"}\n";

/* Reduction (sum) */
static const char *reduce_src =
"__kernel void reduce(\n"
"    __global const float* input, __global float* output,\n"
"    __local float* scratch, const int N) {\n"
"    int gid = get_global_id(0);\n"
"    int lid = get_local_id(0);\n"
"    int group_size = get_local_size(0);\n"
"    scratch[lid] = (gid < N) ? input[gid] : 0.0f;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int s = group_size / 2; s > 0; s >>= 1) {\n"
"        if (lid < s) scratch[lid] += scratch[lid + s];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (lid == 0) output[get_group_id(0)] = scratch[0];\n"
"}\n";

int main(int argc, char **argv) {
    void *lib = NULL;
    const char *paths[] = {
        "libOpenCL.so", "/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so", "/vendor/lib64/mt6855/libPVROCL.so", NULL
    };
    for (int i = 0; paths[i]; i++) {
        lib = dlopen(paths[i], RTLD_NOW);
        if (lib) break;
    }
    if (!lib) { fprintf(stderr, "Failed to load OpenCL: %s\n", dlerror()); return 1; }

    /* Load all functions */
    #define LOAD(name) fn_##name = (pfn_##name)dlsym(lib, #name); \
        if (!fn_##name) { fprintf(stderr, "Missing: " #name "\n"); return 1; }
    LOAD(clGetPlatformIDs); LOAD(clGetDeviceIDs);
    LOAD(clCreateContext); LOAD(clCreateCommandQueue);
    LOAD(clCreateProgramWithSource); LOAD(clBuildProgram);
    LOAD(clCreateKernel); LOAD(clCreateBuffer);
    LOAD(clSetKernelArg); LOAD(clEnqueueNDRangeKernel);
    LOAD(clEnqueueReadBuffer); LOAD(clEnqueueWriteBuffer);
    LOAD(clFinish); LOAD(clWaitForEvents);
    LOAD(clGetEventProfilingInfo); LOAD(clReleaseMemObject);
    LOAD(clReleaseKernel); LOAD(clReleaseProgram);
    LOAD(clReleaseCommandQueue); LOAD(clReleaseContext);
    LOAD(clReleaseEvent); LOAD(clGetProgramBuildInfo);
    #undef LOAD

    /* Setup */
    cl_int err;
    cl_platform_id plat;
    cl_device_id dev;
    fn_clGetPlatformIDs(1, &plat, NULL);
    fn_clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL);

    intptr_t ctx_props[] = { CL_CONTEXT_PLATFORM, (intptr_t)plat, 0 };
    cl_context ctx = fn_clCreateContext(ctx_props, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateContext failed: %d\n", err); return 1; }

    cl_command_queue queue = fn_clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateCommandQueue failed: %d\n", err); return 1; }

    printf("=== GPU Compute Benchmark ===\n");
    printf("Device: PowerVR BXM-8-256\n\n");

    /* ---- Benchmark 1: SGEMM (naive) ---- */
    {
        const int M = 512, N = 512, K = 512;
        size_t sz = M * N * sizeof(float);
        float *hA = malloc(sz), *hB = malloc(sz), *hC = malloc(sz);
        srand(42);
        for (int i = 0; i < M*K; i++) hA[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K*N; i++) hB[i] = (float)(rand() % 100) / 100.0f;

        cl_mem dA = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M*K*sizeof(float), hA, &err);
        cl_mem dB = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K*N*sizeof(float), hB, &err);
        cl_mem dC = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, &err);

        cl_program prog = fn_clCreateProgramWithSource(ctx, 1, &sgemm_src, NULL, &err);
        err = fn_clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
        if (err != CL_SUCCESS) { printf("SGEMM build failed: %d\n", err); print_build_log(prog, dev); }
        else {
            cl_kernel kern = fn_clCreateKernel(prog, "sgemm", &err);
            fn_clSetKernelArg(kern, 0, sizeof(cl_mem), &dA);
            fn_clSetKernelArg(kern, 1, sizeof(cl_mem), &dB);
            fn_clSetKernelArg(kern, 2, sizeof(cl_mem), &dC);
            fn_clSetKernelArg(kern, 3, sizeof(int), &M);
            fn_clSetKernelArg(kern, 4, sizeof(int), &N);
            fn_clSetKernelArg(kern, 5, sizeof(int), &K);

            size_t global[2] = {M, N};
            size_t local[2] = {16, 16};

            /* Warmup */
            cl_event ev;
            fn_clEnqueueNDRangeKernel(queue, kern, 2, NULL, global, local, 0, NULL, NULL);
            fn_clFinish(queue);

            /* Timed runs */
            int runs = 5;
            double total_ms = 0;
            for (int r = 0; r < runs; r++) {
                fn_clEnqueueNDRangeKernel(queue, kern, 2, NULL, global, local, 0, NULL, &ev);
                fn_clWaitForEvents(1, &ev);
                total_ms += get_event_time_ms(ev);
                fn_clReleaseEvent(ev);
            }
            double avg_ms = total_ms / runs;
            double flops = 2.0 * M * N * K;  /* 2*M*N*K FLOPs for matmul */
            double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

            /* Verify */
            fn_clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, sz, hC, 0, NULL, NULL);
            float expected = 0;
            for (int k = 0; k < K; k++) expected += hA[k] * hB[k * N];
            float diff = fabsf(hC[0] - expected);

            printf("[1] SGEMM Naive %dx%dx%d\n", M, N, K);
            printf("    Time:    %.2f ms (avg of %d runs)\n", avg_ms, runs);
            printf("    GFLOPS:  %.2f\n", gflops);
            printf("    Verify:  C[0,0]=%.4f expected=%.4f diff=%.6f %s\n\n",
                   hC[0], expected, diff, diff < 0.01f ? "OK" : "MISMATCH");

            fn_clReleaseKernel(kern);
        }
        fn_clReleaseProgram(prog);
        fn_clReleaseMemObject(dA); fn_clReleaseMemObject(dB); fn_clReleaseMemObject(dC);
        free(hA); free(hB); free(hC);
    }

    /* ---- Benchmark 2: SGEMM Tiled ---- */
    {
        const int M = 512, N = 512, K = 512;
        size_t sz = M * N * sizeof(float);
        float *hA = malloc(sz), *hB = malloc(sz), *hC = malloc(sz);
        srand(42);
        for (int i = 0; i < M*K; i++) hA[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K*N; i++) hB[i] = (float)(rand() % 100) / 100.0f;

        cl_mem dA = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M*K*sizeof(float), hA, &err);
        cl_mem dB = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K*N*sizeof(float), hB, &err);
        cl_mem dC = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, &err);

        cl_program prog = fn_clCreateProgramWithSource(ctx, 1, &sgemm_tiled_src, NULL, &err);
        err = fn_clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
        if (err != CL_SUCCESS) { printf("SGEMM Tiled build failed: %d\n", err); print_build_log(prog, dev); }
        else {
            cl_kernel kern = fn_clCreateKernel(prog, "sgemm_tiled", &err);
            fn_clSetKernelArg(kern, 0, sizeof(cl_mem), &dA);
            fn_clSetKernelArg(kern, 1, sizeof(cl_mem), &dB);
            fn_clSetKernelArg(kern, 2, sizeof(cl_mem), &dC);
            fn_clSetKernelArg(kern, 3, sizeof(int), &M);
            fn_clSetKernelArg(kern, 4, sizeof(int), &N);
            fn_clSetKernelArg(kern, 5, sizeof(int), &K);

            size_t global[2] = {M, N};
            size_t local[2] = {16, 16};

            fn_clEnqueueNDRangeKernel(queue, kern, 2, NULL, global, local, 0, NULL, NULL);
            fn_clFinish(queue);

            int runs = 5;
            double total_ms = 0;
            cl_event ev;
            for (int r = 0; r < runs; r++) {
                fn_clEnqueueNDRangeKernel(queue, kern, 2, NULL, global, local, 0, NULL, &ev);
                fn_clWaitForEvents(1, &ev);
                total_ms += get_event_time_ms(ev);
                fn_clReleaseEvent(ev);
            }
            double avg_ms = total_ms / runs;
            double flops = 2.0 * M * N * K;
            double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

            fn_clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, sz, hC, 0, NULL, NULL);
            float expected = 0;
            for (int k = 0; k < K; k++) expected += hA[k] * hB[k * N];
            float diff = fabsf(hC[0] - expected);

            printf("[2] SGEMM Tiled (16x16) %dx%dx%d\n", M, N, K);
            printf("    Time:    %.2f ms (avg of %d runs)\n", avg_ms, runs);
            printf("    GFLOPS:  %.2f\n", gflops);
            printf("    Verify:  C[0,0]=%.4f expected=%.4f diff=%.6f %s\n\n",
                   hC[0], expected, diff, diff < 0.01f ? "OK" : "MISMATCH");

            fn_clReleaseKernel(kern);
        }
        fn_clReleaseProgram(prog);
        fn_clReleaseMemObject(dA); fn_clReleaseMemObject(dB); fn_clReleaseMemObject(dC);
        free(hA); free(hB); free(hC);
    }

    /* ---- Benchmark 3: FP16 HGEMM ---- */
    {
        const int M = 512, N = 512, K = 512;
        /* Use uint16_t for half storage, convert from float */
        size_t sz_half = M * N * sizeof(uint16_t);
        uint16_t *hA = malloc(M * K * sizeof(uint16_t));
        uint16_t *hB = malloc(K * N * sizeof(uint16_t));
        uint16_t *hC = malloc(sz_half);

        /* Simple float-to-half conversion (IEEE 754) */
        srand(42);
        for (int i = 0; i < M*K; i++) {
            float f = (float)(rand() % 100) / 100.0f;
            /* Use compiler builtin or manual conversion */
            union { float f; uint32_t u; } fu = { .f = f };
            uint32_t sign = (fu.u >> 16) & 0x8000;
            int32_t exp = ((fu.u >> 23) & 0xFF) - 127 + 15;
            uint32_t frac = (fu.u >> 13) & 0x3FF;
            if (exp <= 0) hA[i] = sign;
            else if (exp >= 31) hA[i] = sign | 0x7C00;
            else hA[i] = sign | (exp << 10) | frac;
        }
        for (int i = 0; i < K*N; i++) {
            float f = (float)(rand() % 100) / 100.0f;
            union { float f; uint32_t u; } fu = { .f = f };
            uint32_t sign = (fu.u >> 16) & 0x8000;
            int32_t exp = ((fu.u >> 23) & 0xFF) - 127 + 15;
            uint32_t frac = (fu.u >> 13) & 0x3FF;
            if (exp <= 0) hB[i] = sign;
            else if (exp >= 31) hB[i] = sign | 0x7C00;
            else hB[i] = sign | (exp << 10) | frac;
        }

        cl_mem dA = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M*K*sizeof(uint16_t), hA, &err);
        cl_mem dB = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K*N*sizeof(uint16_t), hB, &err);
        cl_mem dC = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz_half, NULL, &err);

        cl_program prog = fn_clCreateProgramWithSource(ctx, 1, &hgemm_src, NULL, &err);
        err = fn_clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("[3] FP16 HGEMM %dx%dx%d\n", M, N, K);
            printf("    Build FAILED (err=%d)\n", err);
            print_build_log(prog, dev);
            printf("\n");
        } else {
            cl_kernel kern = fn_clCreateKernel(prog, "hgemm", &err);
            fn_clSetKernelArg(kern, 0, sizeof(cl_mem), &dA);
            fn_clSetKernelArg(kern, 1, sizeof(cl_mem), &dB);
            fn_clSetKernelArg(kern, 2, sizeof(cl_mem), &dC);
            fn_clSetKernelArg(kern, 3, sizeof(int), &M);
            fn_clSetKernelArg(kern, 4, sizeof(int), &N);
            fn_clSetKernelArg(kern, 5, sizeof(int), &K);

            size_t global[2] = {M, N};
            size_t local[2] = {16, 16};

            fn_clEnqueueNDRangeKernel(queue, kern, 2, NULL, global, local, 0, NULL, NULL);
            fn_clFinish(queue);

            int runs = 5;
            double total_ms = 0;
            cl_event ev;
            for (int r = 0; r < runs; r++) {
                fn_clEnqueueNDRangeKernel(queue, kern, 2, NULL, global, local, 0, NULL, &ev);
                fn_clWaitForEvents(1, &ev);
                total_ms += get_event_time_ms(ev);
                fn_clReleaseEvent(ev);
            }
            double avg_ms = total_ms / runs;
            double flops = 2.0 * M * N * K;
            double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

            printf("[3] FP16 HGEMM %dx%dx%d\n", M, N, K);
            printf("    Time:    %.2f ms (avg of %d runs)\n", avg_ms, runs);
            printf("    GFLOPS:  %.2f\n\n", gflops);

            fn_clReleaseKernel(kern);
        }
        fn_clReleaseProgram(prog);
        fn_clReleaseMemObject(dA); fn_clReleaseMemObject(dB); fn_clReleaseMemObject(dC);
        free(hA); free(hB); free(hC);
    }

    /* ---- Benchmark 4: Vector Add (bandwidth) ---- */
    {
        const int N = 4 * 1024 * 1024;  /* 4M elements = 48 MB total */
        size_t sz = N * sizeof(float);
        float *hA = malloc(sz), *hB = malloc(sz), *hC = malloc(sz);
        for (int i = 0; i < N; i++) { hA[i] = 1.0f; hB[i] = 2.0f; }

        cl_mem dA = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz, hA, &err);
        cl_mem dB = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz, hB, &err);
        cl_mem dC = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, &err);

        cl_program prog = fn_clCreateProgramWithSource(ctx, 1, &vecadd_src, NULL, &err);
        err = fn_clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
        if (err != CL_SUCCESS) { printf("VecAdd build failed: %d\n", err); print_build_log(prog, dev); }
        else {
            cl_kernel kern = fn_clCreateKernel(prog, "vecadd", &err);
            fn_clSetKernelArg(kern, 0, sizeof(cl_mem), &dA);
            fn_clSetKernelArg(kern, 1, sizeof(cl_mem), &dB);
            fn_clSetKernelArg(kern, 2, sizeof(cl_mem), &dC);
            fn_clSetKernelArg(kern, 3, sizeof(int), &N);

            size_t global_sz = N;
            size_t local_sz = 256;

            fn_clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_sz, &local_sz, 0, NULL, NULL);
            fn_clFinish(queue);

            int runs = 10;
            double total_ms = 0;
            cl_event ev;
            for (int r = 0; r < runs; r++) {
                fn_clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_sz, &local_sz, 0, NULL, &ev);
                fn_clWaitForEvents(1, &ev);
                total_ms += get_event_time_ms(ev);
                fn_clReleaseEvent(ev);
            }
            double avg_ms = total_ms / runs;
            double bytes = 3.0 * sz;  /* 2 reads + 1 write */
            double gb_s = (bytes / (avg_ms / 1000.0)) / 1e9;

            fn_clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, sizeof(float), hC, 0, NULL, NULL);

            printf("[4] Vector Add (4M floats, 48 MB)\n");
            printf("    Time:    %.2f ms (avg of %d runs)\n", avg_ms, runs);
            printf("    BW:      %.2f GB/s\n", gb_s);
            printf("    Verify:  C[0]=%.1f (expected 3.0) %s\n\n",
                   hC[0], hC[0] == 3.0f ? "OK" : "MISMATCH");

            fn_clReleaseKernel(kern);
        }
        fn_clReleaseProgram(prog);
        fn_clReleaseMemObject(dA); fn_clReleaseMemObject(dB); fn_clReleaseMemObject(dC);
        free(hA); free(hB); free(hC);
    }

    /* ---- Benchmark 5: Reduction ---- */
    {
        const int N = 4 * 1024 * 1024;
        size_t sz = N * sizeof(float);
        float *hInput = malloc(sz);
        for (int i = 0; i < N; i++) hInput[i] = 1.0f;

        cl_mem dInput = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz, hInput, &err);
        int wg_size = 256;
        int num_groups = (N + wg_size - 1) / wg_size;
        float *hOutput = malloc(num_groups * sizeof(float));
        cl_mem dOutput = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, num_groups * sizeof(float), NULL, &err);

        cl_program prog = fn_clCreateProgramWithSource(ctx, 1, &reduce_src, NULL, &err);
        err = fn_clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
        if (err != CL_SUCCESS) { printf("Reduce build failed: %d\n", err); print_build_log(prog, dev); }
        else {
            cl_kernel kern = fn_clCreateKernel(prog, "reduce", &err);
            fn_clSetKernelArg(kern, 0, sizeof(cl_mem), &dInput);
            fn_clSetKernelArg(kern, 1, sizeof(cl_mem), &dOutput);
            fn_clSetKernelArg(kern, 2, wg_size * sizeof(float), NULL);  /* local */
            fn_clSetKernelArg(kern, 3, sizeof(int), &N);

            size_t global_sz = (size_t)num_groups * wg_size;
            size_t local_sz = wg_size;

            fn_clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_sz, &local_sz, 0, NULL, NULL);
            fn_clFinish(queue);

            int runs = 10;
            double total_ms = 0;
            cl_event ev;
            for (int r = 0; r < runs; r++) {
                fn_clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_sz, &local_sz, 0, NULL, &ev);
                fn_clWaitForEvents(1, &ev);
                total_ms += get_event_time_ms(ev);
                fn_clReleaseEvent(ev);
            }
            double avg_ms = total_ms / runs;

            fn_clEnqueueReadBuffer(queue, dOutput, CL_TRUE, 0, num_groups * sizeof(float), hOutput, 0, NULL, NULL);
            double sum = 0;
            for (int i = 0; i < num_groups; i++) sum += hOutput[i];

            printf("[5] Reduction (4M floats)\n");
            printf("    Time:    %.2f ms (avg of %d runs)\n", avg_ms, runs);
            printf("    Verify:  sum=%.0f (expected %d) %s\n\n",
                   sum, N, (int)sum == N ? "OK" : "MISMATCH");

            fn_clReleaseKernel(kern);
        }
        fn_clReleaseProgram(prog);
        fn_clReleaseMemObject(dInput); fn_clReleaseMemObject(dOutput);
        free(hInput); free(hOutput);
    }

    /* ---- Benchmark 6: Larger SGEMM 1024x1024 ---- */
    {
        const int M = 1024, N = 1024, K = 1024;
        size_t szA = M * K * sizeof(float);
        size_t szB = K * N * sizeof(float);
        size_t szC = M * N * sizeof(float);
        float *hA = malloc(szA), *hB = malloc(szB), *hC = malloc(szC);
        srand(42);
        for (int i = 0; i < M*K; i++) hA[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K*N; i++) hB[i] = (float)(rand() % 100) / 100.0f;

        cl_mem dA = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, szA, hA, &err);
        cl_mem dB = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, szB, hB, &err);
        cl_mem dC = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, szC, NULL, &err);

        cl_program prog = fn_clCreateProgramWithSource(ctx, 1, &sgemm_tiled_src, NULL, &err);
        err = fn_clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
        if (err != CL_SUCCESS) { printf("SGEMM 1024 build failed: %d\n", err); print_build_log(prog, dev); }
        else {
            cl_kernel kern = fn_clCreateKernel(prog, "sgemm_tiled", &err);
            fn_clSetKernelArg(kern, 0, sizeof(cl_mem), &dA);
            fn_clSetKernelArg(kern, 1, sizeof(cl_mem), &dB);
            fn_clSetKernelArg(kern, 2, sizeof(cl_mem), &dC);
            fn_clSetKernelArg(kern, 3, sizeof(int), &M);
            fn_clSetKernelArg(kern, 4, sizeof(int), &N);
            fn_clSetKernelArg(kern, 5, sizeof(int), &K);

            size_t global[2] = {M, N};
            size_t local[2] = {16, 16};

            fn_clEnqueueNDRangeKernel(queue, kern, 2, NULL, global, local, 0, NULL, NULL);
            fn_clFinish(queue);

            int runs = 3;
            double total_ms = 0;
            cl_event ev;
            for (int r = 0; r < runs; r++) {
                fn_clEnqueueNDRangeKernel(queue, kern, 2, NULL, global, local, 0, NULL, &ev);
                fn_clWaitForEvents(1, &ev);
                total_ms += get_event_time_ms(ev);
                fn_clReleaseEvent(ev);
            }
            double avg_ms = total_ms / runs;
            double flops = 2.0 * M * N * K;
            double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

            printf("[6] SGEMM Tiled 1024x1024x1024\n");
            printf("    Time:    %.2f ms (avg of %d runs)\n", avg_ms, runs);
            printf("    GFLOPS:  %.2f\n\n", gflops);

            fn_clReleaseKernel(kern);
        }
        fn_clReleaseProgram(prog);
        fn_clReleaseMemObject(dA); fn_clReleaseMemObject(dB); fn_clReleaseMemObject(dC);
        free(hA); free(hB); free(hC);
    }

    fn_clReleaseCommandQueue(queue);
    fn_clReleaseContext(ctx);
    dlclose(lib);

    printf("=== Benchmark Complete ===\n");
    return 0;
}
