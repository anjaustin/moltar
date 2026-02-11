/*
 * workloads.c - Practical GPU workload benchmarks
 * Tests real-world use cases: image processing, crypto, FFT, FP16 vectorized
 * Cross-compile with Android NDK for aarch64
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* ---- Minimal OpenCL typedefs (same as gpu_bench.c) ---- */
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

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL          0xFFFFFFFF
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

/* Globals */
static pfn_clGetPlatformIDs fn_clGetPlatformIDs;
static pfn_clGetDeviceIDs fn_clGetDeviceIDs;
static pfn_clCreateContext fn_clCreateContext;
static pfn_clCreateCommandQueue fn_clCreateCommandQueue;
static pfn_clCreateProgramWithSource fn_clCreateProgramWithSource;
static pfn_clBuildProgram fn_clBuildProgram;
static pfn_clCreateKernel fn_clCreateKernel;
static pfn_clCreateBuffer fn_clCreateBuffer;
static pfn_clSetKernelArg fn_clSetKernelArg;
static pfn_clEnqueueNDRangeKernel fn_clEnqueueNDRangeKernel;
static pfn_clEnqueueReadBuffer fn_clEnqueueReadBuffer;
static pfn_clFinish fn_clFinish;
static pfn_clWaitForEvents fn_clWaitForEvents;
static pfn_clGetEventProfilingInfo fn_clGetEventProfilingInfo;
static pfn_clReleaseMemObject fn_clReleaseMemObject;
static pfn_clReleaseKernel fn_clReleaseKernel;
static pfn_clReleaseProgram fn_clReleaseProgram;
static pfn_clReleaseCommandQueue fn_clReleaseCommandQueue;
static pfn_clReleaseContext fn_clReleaseContext;
static pfn_clReleaseEvent fn_clReleaseEvent;
static pfn_clGetProgramBuildInfo fn_clGetProgramBuildInfo;

static double get_event_ms(cl_event ev) {
    cl_ulong s, e;
    fn_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
    fn_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL);
    return (e - s) / 1e6;
}

static void print_build_log(cl_program prog, cl_device_id dev) {
    size_t sz;
    fn_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    if (sz > 1) {
        char *log = malloc(sz + 1);
        fn_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz, log, NULL);
        log[sz] = '\0';
        fprintf(stderr, "BUILD LOG:\n%s\n", log);
        free(log);
    }
}

/* Helper: build kernel, returns NULL on failure */
static cl_kernel build_kernel(cl_context ctx, cl_device_id dev,
                              const char *src, const char *name, const char *opts) {
    cl_int err;
    cl_program prog = fn_clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "CreateProgram failed: %d\n", err); return NULL; }
    err = fn_clBuildProgram(prog, 1, &dev, opts ? opts : "", NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Build '%s' failed: %d\n", name, err);
        print_build_log(prog, dev);
        fn_clReleaseProgram(prog);
        return NULL;
    }
    cl_kernel k = fn_clCreateKernel(prog, name, &err);
    /* NOTE: leaking program handle for simplicity */
    return k;
}

/* Helper: time N runs, return avg ms */
static double bench_kernel(cl_command_queue q, cl_kernel k,
                           cl_uint dims, const size_t *global, const size_t *local,
                           int warmup, int runs) {
    for (int i = 0; i < warmup; i++) {
        fn_clEnqueueNDRangeKernel(q, k, dims, NULL, global, local, 0, NULL, NULL);
    }
    fn_clFinish(q);

    double total = 0;
    for (int i = 0; i < runs; i++) {
        cl_event ev;
        fn_clEnqueueNDRangeKernel(q, k, dims, NULL, global, local, 0, NULL, &ev);
        fn_clWaitForEvents(1, &ev);
        total += get_event_ms(ev);
        fn_clReleaseEvent(ev);
    }
    return total / runs;
}

/* ============================================================
 * KERNEL SOURCES
 * ============================================================ */

/* 1. Image 3x3 convolution (Gaussian blur on 1080p grayscale) */
static const char *conv3x3_src =
"__kernel void conv3x3(\n"
"    __global const float* in, __global float* out,\n"
"    const int W, const int H,\n"
"    __constant float* filter) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x < 1 || x >= W-1 || y < 1 || y >= H-1) {\n"
"        if (x < W && y < H) out[y*W+x] = in[y*W+x];\n"
"        return;\n"
"    }\n"
"    float sum = 0.0f;\n"
"    for (int dy = -1; dy <= 1; dy++)\n"
"        for (int dx = -1; dx <= 1; dx++)\n"
"            sum += in[(y+dy)*W + (x+dx)] * filter[(dy+1)*3 + (dx+1)];\n"
"    out[y*W+x] = sum;\n"
"}\n";

/* 2. Image 5x5 convolution (stronger blur / edge detect) */
static const char *conv5x5_src =
"__kernel void conv5x5(\n"
"    __global const float* in, __global float* out,\n"
"    const int W, const int H,\n"
"    __constant float* filter) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x < 2 || x >= W-2 || y < 2 || y >= H-2) {\n"
"        if (x < W && y < H) out[y*W+x] = in[y*W+x];\n"
"        return;\n"
"    }\n"
"    float sum = 0.0f;\n"
"    for (int dy = -2; dy <= 2; dy++)\n"
"        for (int dx = -2; dx <= 2; dx++)\n"
"            sum += in[(y+dy)*W + (x+dx)] * filter[(dy+2)*5 + (dx+2)];\n"
"    out[y*W+x] = sum;\n"
"}\n";

/* 3. RGB to YUV color space conversion (4K frame, 4 channels) */
static const char *rgb2yuv_src =
"__kernel void rgb2yuv(\n"
"    __global const uchar4* rgb, __global uchar4* yuv, const int N) {\n"
"    int i = get_global_id(0);\n"
"    if (i >= N) return;\n"
"    float r = (float)rgb[i].x;\n"
"    float g = (float)rgb[i].y;\n"
"    float b = (float)rgb[i].z;\n"
"    float Y = 0.299f * r + 0.587f * g + 0.114f * b;\n"
"    float U = -0.169f * r - 0.331f * g + 0.500f * b + 128.0f;\n"
"    float V = 0.500f * r - 0.419f * g - 0.081f * b + 128.0f;\n"
"    yuv[i] = (uchar4)(convert_uchar_sat(Y), convert_uchar_sat(U),\n"
"                       convert_uchar_sat(V), rgb[i].w);\n"
"}\n";

/* 4. Histogram (256-bin grayscale histogram) */
static const char *histogram_src =
"__kernel void histogram(\n"
"    __global const uchar* img, __global volatile uint* bins,\n"
"    const int N) {\n"
"    int i = get_global_id(0);\n"
"    if (i >= N) return;\n"
"    atomic_inc(&bins[img[i]]);\n"
"}\n";

/* 5. SHA-256 (single block hash, batch many messages) */
static const char *sha256_src =
"#define RR(x,n) (((x)>>(n))|((x)<<(32-(n))))\n"
"#define CH(x,y,z) (((x)&(y))^((~(x))&(z)))\n"
"#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))\n"
"#define S0(x) (RR(x,2)^RR(x,13)^RR(x,22))\n"
"#define S1(x) (RR(x,6)^RR(x,11)^RR(x,25))\n"
"#define s0(x) (RR(x,7)^RR(x,18)^((x)>>3))\n"
"#define s1(x) (RR(x,17)^RR(x,19)^((x)>>10))\n"
"\n"
"__constant uint K[64] = {\n"
"    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,\n"
"    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,\n"
"    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,\n"
"    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,\n"
"    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,\n"
"    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,\n"
"    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,\n"
"    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2\n"
"};\n"
"\n"
"__kernel void sha256_batch(\n"
"    __global const uint* messages,  /* N * 16 uint words (64-byte blocks) */\n"
"    __global uint* hashes,          /* N * 8 uint words */\n"
"    const int N) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= N) return;\n"
"    \n"
"    __global const uint* msg = messages + idx * 16;\n"
"    __global uint* hash = hashes + idx * 8;\n"
"    \n"
"    uint W[64];\n"
"    for (int i = 0; i < 16; i++) W[i] = msg[i];\n"
"    for (int i = 16; i < 64; i++) W[i] = s1(W[i-2]) + W[i-7] + s0(W[i-15]) + W[i-16];\n"
"    \n"
"    uint a=0x6a09e667, b=0xbb67ae85, c=0x3c6ef372, d=0xa54ff53a;\n"
"    uint e=0x510e527f, f=0x9b05688c, g=0x1f83d9ab, h=0x5be0cd19;\n"
"    \n"
"    for (int i = 0; i < 64; i++) {\n"
"        uint T1 = h + S1(e) + CH(e,f,g) + K[i] + W[i];\n"
"        uint T2 = S0(a) + MAJ(a,b,c);\n"
"        h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;\n"
"    }\n"
"    \n"
"    hash[0]=a+0x6a09e667; hash[1]=b+0xbb67ae85;\n"
"    hash[2]=c+0x3c6ef372; hash[3]=d+0xa54ff53a;\n"
"    hash[4]=e+0x510e527f; hash[5]=f+0x9b05688c;\n"
"    hash[6]=g+0x1f83d9ab; hash[7]=h+0x5be0cd19;\n"
"}\n";

/* 6. Radix-2 FFT butterfly (in-place, one stage) */
static const char *fft_src =
"__kernel void fft_butterfly(\n"
"    __global float2* data, const int stage, const int N) {\n"
"    int gid = get_global_id(0);\n"
"    int half_size = 1 << stage;\n"
"    int size = half_size << 1;\n"
"    int group = gid / half_size;\n"
"    int j = gid % half_size;\n"
"    int i0 = group * size + j;\n"
"    int i1 = i0 + half_size;\n"
"    if (i1 >= N) return;\n"
"    \n"
"    float angle = -2.0f * M_PI_F * (float)j / (float)size;\n"
"    float2 w = (float2)(cos(angle), sin(angle));\n"
"    float2 a = data[i0];\n"
"    float2 b = data[i1];\n"
"    float2 wb = (float2)(w.x*b.x - w.y*b.y, w.x*b.y + w.y*b.x);\n"
"    data[i0] = a + wb;\n"
"    data[i1] = a - wb;\n"
"}\n";

/* 7. Vectorized FP16 dot product (using half8) */
static const char *fp16_dot_src =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void fp16_dot(\n"
"    __global const half8* A, __global const half8* B,\n"
"    __global float* partial, const int N8) {\n"
"    int gid = get_global_id(0);\n"
"    int lid = get_local_id(0);\n"
"    int gs = get_local_size(0);\n"
"    \n"
"    float sum = 0.0f;\n"
"    for (int i = gid; i < N8; i += get_global_size(0)) {\n"
"        half8 a = A[i];\n"
"        half8 b = B[i];\n"
"        half8 ab = a * b;\n"
"        sum += (float)ab.s0 + (float)ab.s1 + (float)ab.s2 + (float)ab.s3\n"
"             + (float)ab.s4 + (float)ab.s5 + (float)ab.s6 + (float)ab.s7;\n"
"    }\n"
"    \n"
"    __local float scratch[256];\n"
"    scratch[lid] = sum;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int s = gs/2; s > 0; s >>= 1) {\n"
"        if (lid < s) scratch[lid] += scratch[lid + s];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (lid == 0) partial[get_group_id(0)] = scratch[0];\n"
"}\n";

/* 8. FP16 SGEMM with half4 vectorization */
static const char *fp16_matmul_src =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#define TS 16\n"
"__kernel void fp16_matmul(\n"
"    __global const half* A, __global const half* B, __global half* C,\n"
"    const int M, const int N, const int K) {\n"
"    __local half Asub[TS][TS];\n"
"    __local half Bsub[TS][TS];\n"
"    int row = get_local_id(0);\n"
"    int col = get_local_id(1);\n"
"    int gRow = TS * get_group_id(0) + row;\n"
"    int gCol = TS * get_group_id(1) + col;\n"
"    half sum = (half)0.0f;\n"
"    int nT = (K + TS - 1) / TS;\n"
"    for (int t = 0; t < nT; t++) {\n"
"        int tR = TS*t+row, tC = TS*t+col;\n"
"        Asub[row][col] = (gRow<M && tC<K) ? A[gRow*K+tC] : (half)0.0f;\n"
"        Bsub[row][col] = (tR<K && gCol<N) ? B[tR*N+gCol] : (half)0.0f;\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        for (int k = 0; k < TS; k++) sum += Asub[row][k] * Bsub[k][col];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (gRow < M && gCol < N) C[gRow*N+gCol] = sum;\n"
"}\n";

/* 9. Sobel edge detection (classic CV workload) */
static const char *sobel_src =
"__kernel void sobel(\n"
"    __global const uchar* in, __global uchar* out,\n"
"    const int W, const int H) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x < 1 || x >= W-1 || y < 1 || y >= H-1) {\n"
"        if (x < W && y < H) out[y*W+x] = 0;\n"
"        return;\n"
"    }\n"
"    int gx = -in[(y-1)*W+(x-1)] + in[(y-1)*W+(x+1)]\n"
"           - 2*in[y*W+(x-1)]     + 2*in[y*W+(x+1)]\n"
"           - in[(y+1)*W+(x-1)] + in[(y+1)*W+(x+1)];\n"
"    int gy = -in[(y-1)*W+(x-1)] - 2*in[(y-1)*W+x] - in[(y-1)*W+(x+1)]\n"
"           + in[(y+1)*W+(x-1)] + 2*in[(y+1)*W+x] + in[(y+1)*W+(x+1)];\n"
"    int mag = abs(gx) + abs(gy);\n"
"    out[y*W+x] = (uchar)clamp(mag, 0, 255);\n"
"}\n";

/* 10. Bilateral filter (noise reduction, common in camera pipelines) */
static const char *bilateral_src =
"__kernel void bilateral(\n"
"    __global const float* in, __global float* out,\n"
"    const int W, const int H,\n"
"    const float sigma_s, const float sigma_r) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if (x >= W || y >= H) return;\n"
"    int r = (int)(2.0f * sigma_s);\n"
"    float center = in[y*W+x];\n"
"    float sum_w = 0.0f, sum_v = 0.0f;\n"
"    float inv_ss = -0.5f / (sigma_s * sigma_s);\n"
"    float inv_sr = -0.5f / (sigma_r * sigma_r);\n"
"    for (int dy = -r; dy <= r; dy++) {\n"
"        for (int dx = -r; dx <= r; dx++) {\n"
"            int nx = clamp(x+dx, 0, W-1);\n"
"            int ny = clamp(y+dy, 0, H-1);\n"
"            float val = in[ny*W+nx];\n"
"            float ds = (float)(dx*dx + dy*dy);\n"
"            float dr = (val - center) * (val - center);\n"
"            float w = exp(ds * inv_ss + dr * inv_sr);\n"
"            sum_w += w;\n"
"            sum_v += w * val;\n"
"        }\n"
"    }\n"
"    out[y*W+x] = sum_v / sum_w;\n"
"}\n";


int main(int argc, char **argv) {
    /* Load OpenCL */
    void *lib = NULL;
    const char *paths[] = {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", NULL};
    for (int i = 0; paths[i]; i++) { lib = dlopen(paths[i], RTLD_NOW); if (lib) break; }
    if (!lib) { fprintf(stderr, "No OpenCL: %s\n", dlerror()); return 1; }

    #define LOAD(name) fn_##name = (pfn_##name)dlsym(lib, #name); \
        if (!fn_##name) { fprintf(stderr, "Missing: " #name "\n"); return 1; }
    LOAD(clGetPlatformIDs); LOAD(clGetDeviceIDs);
    LOAD(clCreateContext); LOAD(clCreateCommandQueue);
    LOAD(clCreateProgramWithSource); LOAD(clBuildProgram);
    LOAD(clCreateKernel); LOAD(clCreateBuffer);
    LOAD(clSetKernelArg); LOAD(clEnqueueNDRangeKernel);
    LOAD(clEnqueueReadBuffer); LOAD(clFinish);
    LOAD(clWaitForEvents); LOAD(clGetEventProfilingInfo);
    LOAD(clReleaseMemObject); LOAD(clReleaseKernel);
    LOAD(clReleaseProgram); LOAD(clReleaseCommandQueue);
    LOAD(clReleaseContext); LOAD(clReleaseEvent);
    LOAD(clGetProgramBuildInfo);
    #undef LOAD

    cl_int err;
    cl_platform_id plat; cl_device_id dev;
    fn_clGetPlatformIDs(1, &plat, NULL);
    fn_clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL);
    intptr_t ctx_props[] = { CL_CONTEXT_PLATFORM, (intptr_t)plat, 0 };
    cl_context ctx = fn_clCreateContext(ctx_props, 1, &dev, NULL, NULL, &err);
    cl_command_queue q = fn_clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);

    printf("=== Practical GPU Workload Benchmarks ===\n");
    printf("PowerVR BXM-8-256 @ MT6855\n\n");

    /* ============================================================
     * [1] 3x3 Gaussian Blur on 1080p grayscale
     * ============================================================ */
    {
        int W = 1920, H = 1080;
        size_t sz = W * H * sizeof(float);
        float *img = malloc(sz);
        for (int i = 0; i < W*H; i++) img[i] = (float)(rand() % 256) / 255.0f;
        float gauss3x3[] = {1/16.f, 2/16.f, 1/16.f, 2/16.f, 4/16.f, 2/16.f, 1/16.f, 2/16.f, 1/16.f};

        cl_mem dIn = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz, img, &err);
        cl_mem dOut = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, &err);
        cl_mem dFilt = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 9*sizeof(float), gauss3x3, &err);

        cl_kernel k = build_kernel(ctx, dev, conv3x3_src, "conv3x3", "-cl-fast-relaxed-math");
        if (k) {
            fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dIn);
            fn_clSetKernelArg(k, 1, sizeof(cl_mem), &dOut);
            fn_clSetKernelArg(k, 2, sizeof(int), &W);
            fn_clSetKernelArg(k, 3, sizeof(int), &H);
            fn_clSetKernelArg(k, 4, sizeof(cl_mem), &dFilt);
            size_t g[2] = {((W+15)/16)*16, ((H+15)/16)*16};
            size_t l[2] = {16, 16};
            double ms = bench_kernel(q, k, 2, g, l, 3, 10);
            double mpix = (double)(W*H) / 1e6;
            printf("[1] 3x3 Gaussian Blur (1080p, %.1f MP)\n", mpix);
            printf("    Time: %.2f ms  |  %.1f MP/s  |  %.0f FPS\n\n", ms, mpix/(ms/1000), 1000/ms);
            fn_clReleaseKernel(k);
        }
        fn_clReleaseMemObject(dIn); fn_clReleaseMemObject(dOut); fn_clReleaseMemObject(dFilt);
        free(img);
    }

    /* ============================================================
     * [2] 5x5 Gaussian Blur on 1080p
     * ============================================================ */
    {
        int W = 1920, H = 1080;
        size_t sz = W * H * sizeof(float);
        float *img = malloc(sz);
        for (int i = 0; i < W*H; i++) img[i] = (float)(rand() % 256) / 255.0f;
        float gauss5x5[] = {
            1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f,
            4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f,
            6/256.f, 24/256.f, 36/256.f, 24/256.f, 6/256.f,
            4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f,
            1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f
        };

        cl_mem dIn = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz, img, &err);
        cl_mem dOut = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, &err);
        cl_mem dFilt = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 25*sizeof(float), gauss5x5, &err);

        cl_kernel k = build_kernel(ctx, dev, conv5x5_src, "conv5x5", "-cl-fast-relaxed-math");
        if (k) {
            fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dIn);
            fn_clSetKernelArg(k, 1, sizeof(cl_mem), &dOut);
            fn_clSetKernelArg(k, 2, sizeof(int), &W);
            fn_clSetKernelArg(k, 3, sizeof(int), &H);
            fn_clSetKernelArg(k, 4, sizeof(cl_mem), &dFilt);
            size_t g[2] = {((W+15)/16)*16, ((H+15)/16)*16};
            size_t l[2] = {16, 16};
            double ms = bench_kernel(q, k, 2, g, l, 3, 10);
            double mpix = (double)(W*H) / 1e6;
            printf("[2] 5x5 Gaussian Blur (1080p)\n");
            printf("    Time: %.2f ms  |  %.1f MP/s  |  %.0f FPS\n\n", ms, mpix/(ms/1000), 1000/ms);
            fn_clReleaseKernel(k);
        }
        fn_clReleaseMemObject(dIn); fn_clReleaseMemObject(dOut); fn_clReleaseMemObject(dFilt);
        free(img);
    }

    /* ============================================================
     * [3] Sobel Edge Detection (1080p grayscale, uchar)
     * ============================================================ */
    {
        int W = 1920, H = 1080;
        size_t sz = W * H;
        uint8_t *img = malloc(sz);
        for (int i = 0; i < (int)sz; i++) img[i] = rand() % 256;

        cl_mem dIn = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz, img, &err);
        cl_mem dOut = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, &err);

        cl_kernel k = build_kernel(ctx, dev, sobel_src, "sobel", NULL);
        if (k) {
            fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dIn);
            fn_clSetKernelArg(k, 1, sizeof(cl_mem), &dOut);
            fn_clSetKernelArg(k, 2, sizeof(int), &W);
            fn_clSetKernelArg(k, 3, sizeof(int), &H);
            size_t g[2] = {((W+15)/16)*16, ((H+15)/16)*16};
            size_t l[2] = {16, 16};
            double ms = bench_kernel(q, k, 2, g, l, 3, 10);
            double mpix = (double)(W*H) / 1e6;
            printf("[3] Sobel Edge Detection (1080p, uchar)\n");
            printf("    Time: %.2f ms  |  %.1f MP/s  |  %.0f FPS\n\n", ms, mpix/(ms/1000), 1000/ms);
            fn_clReleaseKernel(k);
        }
        fn_clReleaseMemObject(dIn); fn_clReleaseMemObject(dOut);
        free(img);
    }

    /* ============================================================
     * [4] RGB→YUV Color Conversion (4K frame)
     * ============================================================ */
    {
        int W = 3840, H = 2160;
        int N = W * H;
        size_t sz = N * 4;  /* RGBA */
        uint8_t *rgb = malloc(sz);
        for (int i = 0; i < (int)sz; i++) rgb[i] = rand() % 256;

        cl_mem dRgb = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz, rgb, &err);
        cl_mem dYuv = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, &err);

        cl_kernel k = build_kernel(ctx, dev, rgb2yuv_src, "rgb2yuv", "-cl-fast-relaxed-math");
        if (k) {
            fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dRgb);
            fn_clSetKernelArg(k, 1, sizeof(cl_mem), &dYuv);
            fn_clSetKernelArg(k, 2, sizeof(int), &N);
            size_t g = ((N+255)/256)*256;
            size_t l = 256;
            double ms = bench_kernel(q, k, 1, &g, &l, 3, 10);
            double mpix = (double)N / 1e6;
            printf("[4] RGB->YUV (4K = %.1f MP, RGBA)\n", mpix);
            printf("    Time: %.2f ms  |  %.1f MP/s  |  %.0f FPS\n\n", ms, mpix/(ms/1000), 1000/ms);
            fn_clReleaseKernel(k);
        }
        fn_clReleaseMemObject(dRgb); fn_clReleaseMemObject(dYuv);
        free(rgb);
    }

    /* ============================================================
     * [5] Histogram (1080p grayscale)
     * ============================================================ */
    {
        int W = 1920, H = 1080;
        int N = W * H;
        uint8_t *img = malloc(N);
        uint32_t bins[256] = {0};
        for (int i = 0; i < N; i++) img[i] = rand() % 256;

        cl_mem dImg = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, N, img, &err);
        cl_mem dBins = fn_clCreateBuffer(ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, 256*sizeof(uint32_t), bins, &err);

        cl_kernel k = build_kernel(ctx, dev, histogram_src, "histogram", NULL);
        if (k) {
            fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dImg);
            fn_clSetKernelArg(k, 1, sizeof(cl_mem), &dBins);
            fn_clSetKernelArg(k, 2, sizeof(int), &N);
            size_t g = ((N+255)/256)*256;
            size_t l = 256;

            /* Reset bins before each run for correctness */
            fn_clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL);
            fn_clFinish(q);

            /* For histogram, we need to reset bins each run */
            int runs = 10;
            double total = 0;
            for (int r = 0; r < runs; r++) {
                memset(bins, 0, sizeof(bins));
                fn_clEnqueueReadBuffer(q, dBins, CL_TRUE, 0, 0, NULL, 0, NULL, NULL);  /* sync */
                /* Write zeros to bins buffer */
                cl_mem dBins2 = fn_clCreateBuffer(ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, 256*sizeof(uint32_t), bins, &err);
                fn_clSetKernelArg(k, 1, sizeof(cl_mem), &dBins2);
                cl_event ev;
                fn_clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, &ev);
                fn_clWaitForEvents(1, &ev);
                total += get_event_ms(ev);
                fn_clReleaseEvent(ev);
                fn_clReleaseMemObject(dBins2);
            }
            double ms = total / runs;
            printf("[5] Histogram 256-bin (1080p)\n");
            printf("    Time: %.2f ms  |  %.0f FPS\n\n", ms, 1000/ms);
            fn_clReleaseKernel(k);
        }
        fn_clReleaseMemObject(dImg); fn_clReleaseMemObject(dBins);
        free(img);
    }

    /* ============================================================
     * [6] SHA-256 Batch Hashing (256K messages × 64 bytes)
     * ============================================================ */
    {
        int N = 256 * 1024;  /* 256K messages */
        size_t msg_sz = N * 16 * sizeof(uint32_t);  /* 64 bytes each */
        size_t hash_sz = N * 8 * sizeof(uint32_t);   /* 32 bytes each */
        uint32_t *msgs = malloc(msg_sz);
        uint32_t *hashes = malloc(hash_sz);
        for (int i = 0; i < N * 16; i++) msgs[i] = rand();

        cl_mem dMsg = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, msg_sz, msgs, &err);
        cl_mem dHash = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, hash_sz, NULL, &err);

        cl_kernel k = build_kernel(ctx, dev, sha256_src, "sha256_batch", NULL);
        if (k) {
            fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dMsg);
            fn_clSetKernelArg(k, 1, sizeof(cl_mem), &dHash);
            fn_clSetKernelArg(k, 2, sizeof(int), &N);
            size_t g = ((N+255)/256)*256;
            size_t l = 256;
            double ms = bench_kernel(q, k, 1, &g, &l, 2, 5);
            double mhash_s = (double)N / (ms / 1000.0) / 1e6;
            double mb_s = (double)(N * 64) / (ms / 1000.0) / 1e6;
            printf("[6] SHA-256 Batch (%dK messages x 64B)\n", N/1024);
            printf("    Time: %.2f ms  |  %.1f MHash/s  |  %.1f MB/s throughput\n\n", ms, mhash_s, mb_s);
            fn_clReleaseKernel(k);
        }
        fn_clReleaseMemObject(dMsg); fn_clReleaseMemObject(dHash);
        free(msgs); free(hashes);
    }

    /* ============================================================
     * [7] FFT (1M-point complex, all stages)
     * ============================================================ */
    {
        int N = 1024 * 1024;  /* 1M points */
        int log2N = 20;
        size_t sz = N * 2 * sizeof(float);  /* float2 */
        float *data = malloc(sz);
        for (int i = 0; i < N*2; i++) data[i] = (float)(rand() % 1000) / 1000.0f;

        cl_mem dData = fn_clCreateBuffer(ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sz, data, &err);

        cl_kernel k = build_kernel(ctx, dev, fft_src, "fft_butterfly", "-cl-fast-relaxed-math");
        if (k) {
            fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dData);
            fn_clSetKernelArg(k, 2, sizeof(int), &N);
            size_t l = 256;

            /* Warmup: run all stages once */
            for (int s = 0; s < log2N; s++) {
                fn_clSetKernelArg(k, 1, sizeof(int), &s);
                size_t g = ((N/2+255)/256)*256;
                fn_clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL);
            }
            fn_clFinish(q);

            /* Timed: full FFT */
            int runs = 3;
            double total = 0;
            for (int r = 0; r < runs; r++) {
                /* Reload data */
                fn_clReleaseMemObject(dData);
                dData = fn_clCreateBuffer(ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sz, data, &err);
                fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dData);
                fn_clFinish(q);

                double run_ms = 0;
                for (int s = 0; s < log2N; s++) {
                    fn_clSetKernelArg(k, 1, sizeof(int), &s);
                    size_t g = ((N/2+255)/256)*256;
                    cl_event ev;
                    fn_clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, &ev);
                    fn_clWaitForEvents(1, &ev);
                    run_ms += get_event_ms(ev);
                    fn_clReleaseEvent(ev);
                }
                total += run_ms;
            }
            double ms = total / runs;
            printf("[7] FFT 1M-point complex (20 stages)\n");
            printf("    Time: %.2f ms  |  %.1f MFFT-samples/s\n\n", ms, (double)N/(ms/1000)/1e6);
            fn_clReleaseKernel(k);
        }
        fn_clReleaseMemObject(dData);
        free(data);
    }

    /* ============================================================
     * [8] FP16 Vectorized Dot Product (16M elements, half8)
     * ============================================================ */
    {
        int N = 16 * 1024 * 1024;
        int N8 = N / 8;
        size_t sz = N * sizeof(uint16_t);  /* half = 2 bytes */
        uint16_t *hA = malloc(sz), *hB = malloc(sz);
        /* Fill with half(1.0) = 0x3C00 */
        for (int i = 0; i < N; i++) { hA[i] = 0x3C00; hB[i] = 0x3C00; }

        cl_mem dA = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz, hA, &err);
        cl_mem dB = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz, hB, &err);
        int ngroups = 1024;
        cl_mem dPartial = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, ngroups*sizeof(float), NULL, &err);

        cl_kernel k = build_kernel(ctx, dev, fp16_dot_src, "fp16_dot", "-cl-fast-relaxed-math");
        if (k) {
            fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dA);
            fn_clSetKernelArg(k, 1, sizeof(cl_mem), &dB);
            fn_clSetKernelArg(k, 2, sizeof(cl_mem), &dPartial);
            fn_clSetKernelArg(k, 3, sizeof(int), &N8);
            size_t g = ngroups * 256;
            size_t l = 256;
            double ms = bench_kernel(q, k, 1, &g, &l, 3, 10);

            float *partial = malloc(ngroups * sizeof(float));
            fn_clEnqueueReadBuffer(q, dPartial, CL_TRUE, 0, ngroups*sizeof(float), partial, 0, NULL, NULL);
            double sum = 0;
            for (int i = 0; i < ngroups; i++) sum += partial[i];

            double gb_s = (2.0 * sz) / (ms / 1000.0) / 1e9;
            printf("[8] FP16 Dot Product (16M half, vectorized half8)\n");
            printf("    Time: %.2f ms  |  %.1f GB/s  |  sum=%.0f (expected %d)\n\n", ms, gb_s, sum, N);
            fn_clReleaseKernel(k);
            free(partial);
        }
        fn_clReleaseMemObject(dA); fn_clReleaseMemObject(dB); fn_clReleaseMemObject(dPartial);
        free(hA); free(hB);
    }

    /* ============================================================
     * [9] FP16 Tiled Matrix Multiply (512x512)
     * ============================================================ */
    {
        int M = 512, N = 512, K = 512;
        size_t sz = M * N * sizeof(uint16_t);
        uint16_t *hA = malloc(M*K*2), *hB = malloc(K*N*2), *hC = malloc(sz);
        /* half(0.5) = 0x3800, half(1.0) = 0x3C00 */
        for (int i = 0; i < M*K; i++) hA[i] = 0x3800;
        for (int i = 0; i < K*N; i++) hB[i] = 0x3800;

        cl_mem dA = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, M*K*2, hA, &err);
        cl_mem dB = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, K*N*2, hB, &err);
        cl_mem dC = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, &err);

        cl_kernel k = build_kernel(ctx, dev, fp16_matmul_src, "fp16_matmul", "-cl-fast-relaxed-math");
        if (k) {
            fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dA);
            fn_clSetKernelArg(k, 1, sizeof(cl_mem), &dB);
            fn_clSetKernelArg(k, 2, sizeof(cl_mem), &dC);
            fn_clSetKernelArg(k, 3, sizeof(int), &M);
            fn_clSetKernelArg(k, 4, sizeof(int), &N);
            fn_clSetKernelArg(k, 5, sizeof(int), &K);
            size_t g[2] = {M, N};
            size_t l[2] = {16, 16};
            double ms = bench_kernel(q, k, 2, g, l, 3, 5);
            double flops = 2.0 * M * N * K;
            double gflops = flops / (ms / 1000.0) / 1e9;
            printf("[9] FP16 Tiled GEMM 512x512x512\n");
            printf("    Time: %.2f ms  |  %.1f GFLOPS (FP16)\n\n", ms, gflops);
            fn_clReleaseKernel(k);
        }
        fn_clReleaseMemObject(dA); fn_clReleaseMemObject(dB); fn_clReleaseMemObject(dC);
        free(hA); free(hB); free(hC);
    }

    /* ============================================================
     * [10] Bilateral Filter (720p, sigma_s=3, sigma_r=0.1)
     *      Heavy per-pixel workload: ~49 samples per pixel
     * ============================================================ */
    {
        int W = 1280, H = 720;
        size_t sz = W * H * sizeof(float);
        float *img = malloc(sz);
        for (int i = 0; i < W*H; i++) img[i] = (float)(rand() % 256) / 255.0f;
        float sigma_s = 3.0f, sigma_r = 0.1f;

        cl_mem dIn = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sz, img, &err);
        cl_mem dOut = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, &err);

        cl_kernel k = build_kernel(ctx, dev, bilateral_src, "bilateral", "-cl-fast-relaxed-math");
        if (k) {
            fn_clSetKernelArg(k, 0, sizeof(cl_mem), &dIn);
            fn_clSetKernelArg(k, 1, sizeof(cl_mem), &dOut);
            fn_clSetKernelArg(k, 2, sizeof(int), &W);
            fn_clSetKernelArg(k, 3, sizeof(int), &H);
            fn_clSetKernelArg(k, 4, sizeof(float), &sigma_s);
            fn_clSetKernelArg(k, 5, sizeof(float), &sigma_r);
            size_t g[2] = {((W+15)/16)*16, ((H+15)/16)*16};
            size_t l[2] = {16, 16};
            double ms = bench_kernel(q, k, 2, g, l, 2, 5);
            double mpix = (double)(W*H) / 1e6;
            printf("[10] Bilateral Filter (720p, sigma=3/0.1, 7x7 window)\n");
            printf("     Time: %.2f ms  |  %.1f MP/s  |  %.0f FPS\n\n", ms, mpix/(ms/1000), 1000/ms);
            fn_clReleaseKernel(k);
        }
        fn_clReleaseMemObject(dIn); fn_clReleaseMemObject(dOut);
        free(img);
    }

    fn_clReleaseCommandQueue(q);
    fn_clReleaseContext(ctx);
    dlclose(lib);

    printf("=== All Workloads Complete ===\n");
    return 0;
}
