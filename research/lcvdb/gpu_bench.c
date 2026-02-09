/* L-Cache VDB — GPU OpenCL benchmark
 * Measures kernel launch overhead and dot product throughput
 * on PowerVR BXM-8-256 vs CPU NEON.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <dlfcn.h>

/* ---- Minimal OpenCL type defs (no headers needed) ---- */
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef void*    cl_platform_id;
typedef void*    cl_device_id;
typedef void*    cl_context;
typedef void*    cl_command_queue;
typedef void*    cl_program;
typedef void*    cl_kernel;
typedef void*    cl_mem;
typedef void*    cl_event;
typedef cl_uint  cl_platform_info;
typedef cl_uint  cl_device_info;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint  cl_program_build_info;
typedef cl_uint  cl_profiling_info;
typedef intptr_t cl_context_properties;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU 4
#define CL_MEM_READ_ONLY  (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
#define CL_PROFILING_COMMAND_QUEUED 0x1280
#define CL_PROFILING_COMMAND_SUBMIT 0x1281
#define CL_CONTEXT_PLATFORM 0x1084

/* Function pointer types */
typedef cl_int (*fn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*fn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*fn_clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*fn_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_program (*fn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*fn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_int (*fn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
typedef cl_kernel (*fn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (*fn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (*fn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*fn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*fn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*fn_clFinish)(cl_command_queue);
typedef cl_int (*fn_clGetEventProfilingInfo)(cl_event, cl_profiling_info, size_t, void*, size_t*);
typedef cl_int (*fn_clReleaseEvent)(cl_event);
typedef cl_int (*fn_clReleaseMemObject)(cl_mem);
typedef cl_int (*fn_clReleaseKernel)(cl_kernel);
typedef cl_int (*fn_clReleaseProgram)(cl_program);
typedef cl_int (*fn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*fn_clReleaseContext)(cl_context);
typedef cl_int (*fn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*fn_clWaitForEvents)(cl_uint, const cl_event*);

/* Globals for loaded functions */
static fn_clGetPlatformIDs p_clGetPlatformIDs;
static fn_clGetDeviceIDs p_clGetDeviceIDs;
static fn_clCreateContext p_clCreateContext;
static fn_clCreateCommandQueue p_clCreateCommandQueue;
static fn_clCreateProgramWithSource p_clCreateProgramWithSource;
static fn_clBuildProgram p_clBuildProgram;
static fn_clGetProgramBuildInfo p_clGetProgramBuildInfo;
static fn_clCreateKernel p_clCreateKernel;
static fn_clCreateBuffer p_clCreateBuffer;
static fn_clSetKernelArg p_clSetKernelArg;
static fn_clEnqueueNDRangeKernel p_clEnqueueNDRangeKernel;
static fn_clEnqueueReadBuffer p_clEnqueueReadBuffer;
static fn_clFinish p_clFinish;
static fn_clGetEventProfilingInfo p_clGetEventProfilingInfo;
static fn_clReleaseEvent p_clReleaseEvent;
static fn_clReleaseMemObject p_clReleaseMemObject;
static fn_clReleaseKernel p_clReleaseKernel;
static fn_clReleaseProgram p_clReleaseProgram;
static fn_clReleaseCommandQueue p_clReleaseCommandQueue;
static fn_clReleaseContext p_clReleaseContext;
static fn_clEnqueueWriteBuffer p_clEnqueueWriteBuffer;
static fn_clWaitForEvents p_clWaitForEvents;

#define LOAD_SYM(lib, name) do { \
    p_##name = (fn_##name)dlsym(lib, #name); \
    if (!p_##name) { printf("Missing: %s\n", #name); return 1; } \
} while(0)

/* RNG */
static uint32_t rng_state;
static void rng_seed(uint32_t s) { rng_state = s; }
static int8_t rand_i8(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (int8_t)(rng_state & 0xFF);
}

/* CPU reference dot product */
static int32_t cpu_dot(const int8_t *a, const int8_t *b) {
    int32_t sum = 0;
    for (int i = 0; i < 48; i++)
        sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* OpenCL kernel source — embedded */
static const char *kernel_src =
"__kernel void dot_batch(\n"
"    __global const char *query,\n"
"    __global const char *nodes,\n"
"    __global int *scores,\n"
"    const int node_count\n"
") {\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= node_count) return;\n"
"    char16 q0 = vload16(0, query);\n"
"    char16 q1 = vload16(1, query);\n"
"    char16 q2 = vload16(2, query);\n"
"    __global const char *nv = nodes + gid * 64;\n"
"    char16 n0 = vload16(0, nv);\n"
"    char16 n1 = vload16(1, nv);\n"
"    char16 n2 = vload16(2, nv);\n"
"    short16 p0 = convert_short16(q0) * convert_short16(n0);\n"
"    short16 p1 = convert_short16(q1) * convert_short16(n1);\n"
"    short16 p2 = convert_short16(q2) * convert_short16(n2);\n"
"    int16 s = convert_int16(p0) + convert_int16(p1) + convert_int16(p2);\n"
"    int8 h8 = s.lo + s.hi;\n"
"    int4 h4 = h8.lo + h8.hi;\n"
"    int2 h2 = h4.lo + h4.hi;\n"
"    scores[gid] = h2.x + h2.y;\n"
"}\n"
"\n"
"__kernel void dot_batch_local(\n"
"    __global const char *query,\n"
"    __global const char *nodes,\n"
"    __global int *scores,\n"
"    const int node_count\n"
") {\n"
"    __local char lq[48];\n"
"    int lid = get_local_id(0);\n"
"    if (lid < 48) lq[lid] = query[lid];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    int gid = get_global_id(0);\n"
"    if (gid >= node_count) return;\n"
"    char16 q0 = vload16(0, lq);\n"
"    char16 q1 = vload16(1, lq);\n"
"    char16 q2 = vload16(2, lq);\n"
"    __global const char *nv = nodes + gid * 64;\n"
"    char16 n0 = vload16(0, nv);\n"
"    char16 n1 = vload16(1, nv);\n"
"    char16 n2 = vload16(2, nv);\n"
"    short16 p0 = convert_short16(q0) * convert_short16(n0);\n"
"    short16 p1 = convert_short16(q1) * convert_short16(n1);\n"
"    short16 p2 = convert_short16(q2) * convert_short16(n2);\n"
"    int16 s = convert_int16(p0) + convert_int16(p1) + convert_int16(p2);\n"
"    int8 h8 = s.lo + s.hi;\n"
"    int4 h4 = h8.lo + h8.hi;\n"
"    int2 h2 = h4.lo + h4.hi;\n"
"    scores[gid] = h2.x + h2.y;\n"
"}\n";

int main(void) {
    printf("=== GPU OpenCL Dot Product Benchmark ===\n");
    printf("PowerVR BXM-8-256 vs CPU NEON\n\n");

    /* Load OpenCL */
    void *lib = dlopen("libOpenCL.so", RTLD_NOW);
    if (!lib) { printf("No libOpenCL.so: %s\n", dlerror()); return 1; }

    LOAD_SYM(lib, clGetPlatformIDs);
    LOAD_SYM(lib, clGetDeviceIDs);
    LOAD_SYM(lib, clCreateContext);
    LOAD_SYM(lib, clCreateCommandQueue);
    LOAD_SYM(lib, clCreateProgramWithSource);
    LOAD_SYM(lib, clBuildProgram);
    LOAD_SYM(lib, clGetProgramBuildInfo);
    LOAD_SYM(lib, clCreateKernel);
    LOAD_SYM(lib, clCreateBuffer);
    LOAD_SYM(lib, clSetKernelArg);
    LOAD_SYM(lib, clEnqueueNDRangeKernel);
    LOAD_SYM(lib, clEnqueueReadBuffer);
    LOAD_SYM(lib, clFinish);
    LOAD_SYM(lib, clGetEventProfilingInfo);
    LOAD_SYM(lib, clReleaseEvent);
    LOAD_SYM(lib, clReleaseMemObject);
    LOAD_SYM(lib, clReleaseKernel);
    LOAD_SYM(lib, clReleaseProgram);
    LOAD_SYM(lib, clReleaseCommandQueue);
    LOAD_SYM(lib, clReleaseContext);
    LOAD_SYM(lib, clEnqueueWriteBuffer);
    LOAD_SYM(lib, clWaitForEvents);

    /* Setup */
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    p_clGetPlatformIDs(1, &platform, NULL);
    p_clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context_properties ctx_props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context ctx = p_clCreateContext(ctx_props, 1, &device, NULL, NULL, &err);
    if (err) { printf("Context err: %d\n", err); return 1; }

    cl_command_queue queue = p_clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err) { printf("Queue err: %d\n", err); return 1; }

    /* Build program */
    size_t src_len = strlen(kernel_src);
    cl_program prog = p_clCreateProgramWithSource(ctx, 1, &kernel_src, &src_len, &err);
    if (err) { printf("Program err: %d\n", err); return 1; }

    err = p_clBuildProgram(prog, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err) {
        char log[4096];
        p_clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build error %d:\n%s\n", err, log);
        return 1;
    }
    printf("Kernel compiled OK\n");

    cl_kernel kern = p_clCreateKernel(prog, "dot_batch", &err);
    cl_kernel kern_local = p_clCreateKernel(prog, "dot_batch_local", &err);
    if (err) { printf("Kernel create err: %d\n", err); return 1; }
    printf("Kernels created OK\n\n");

    /* Test sizes */
    int test_sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
    int num_sizes = 7;

    for (int si = 0; si < num_sizes; si++) {
        int N = test_sizes[si];

        /* Generate data */
        int8_t query[48];
        int8_t *nodes = (int8_t *)calloc(N, 64);  /* 64 bytes per node */
        int32_t *gpu_scores = (int32_t *)calloc(N, sizeof(int32_t));
        int32_t *cpu_scores = (int32_t *)calloc(N, sizeof(int32_t));

        rng_seed(42);
        for (int i = 0; i < 48; i++) query[i] = rand_i8();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < 48; j++)
                nodes[i * 64 + j] = rand_i8();

        /* CPU reference */
        int64_t cpu_t0 = now_ns();
        for (int i = 0; i < N; i++)
            cpu_scores[i] = cpu_dot(query, (const int8_t *)(nodes + i * 64));
        int64_t cpu_t1 = now_ns();
        int64_t cpu_ns = cpu_t1 - cpu_t0;

        /* GPU buffers */
        cl_mem buf_query = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 48, query, &err);
        cl_mem buf_nodes = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * 64, nodes, &err);
        cl_mem buf_scores = p_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * sizeof(int32_t), NULL, &err);

        p_clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_query);
        p_clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_nodes);
        p_clSetKernelArg(kern, 2, sizeof(cl_mem), &buf_scores);
        p_clSetKernelArg(kern, 3, sizeof(int), &N);

        /* Warmup */
        size_t global = (N + 127) & ~127;  /* round up to 128 */
        size_t local = 128;
        if (global < local) local = global;

        for (int w = 0; w < 5; w++) {
            p_clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global, &local, 0, NULL, NULL);
        }
        p_clFinish(queue);

        /* Bench: measure total wall time for many dispatches */
        int gpu_iters = 1000;
        cl_event event;

        /* Single dispatch with profiling */
        p_clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global, &local, 0, NULL, &event);
        p_clFinish(queue);

        cl_ulong t_queued, t_submit, t_start, t_end;
        p_clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &t_queued, NULL);
        p_clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &t_submit, NULL);
        p_clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL);
        p_clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL);
        p_clReleaseEvent(event);

        /* Batch wall time */
        int64_t gpu_t0 = now_ns();
        for (int i = 0; i < gpu_iters; i++) {
            p_clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global, &local, 0, NULL, NULL);
        }
        p_clFinish(queue);
        int64_t gpu_t1 = now_ns();
        int64_t gpu_wall_per = (gpu_t1 - gpu_t0) / gpu_iters;

        /* Read back and verify */
        p_clEnqueueReadBuffer(queue, buf_scores, 1, 0, N * sizeof(int32_t), gpu_scores, 0, NULL, NULL);

        int mismatches = 0;
        for (int i = 0; i < N; i++) {
            if (gpu_scores[i] != cpu_scores[i]) {
                if (mismatches < 3)
                    printf("  MISMATCH node %d: gpu=%d cpu=%d\n", i, gpu_scores[i], cpu_scores[i]);
                mismatches++;
            }
        }

        printf("N=%4d | CPU: %5lld ns | GPU wall: %5lld ns | "
               "GPU prof: queue=%lld submit=%lld exec=%lld ns | "
               "match=%d/%d\n",
               N,
               (long long)cpu_ns,
               (long long)gpu_wall_per,
               (long long)(t_submit - t_queued),
               (long long)(t_start - t_submit),
               (long long)(t_end - t_start),
               N - mismatches, N);

        /* Also bench the local-memory variant */
        p_clSetKernelArg(kern_local, 0, sizeof(cl_mem), &buf_query);
        p_clSetKernelArg(kern_local, 1, sizeof(cl_mem), &buf_nodes);
        p_clSetKernelArg(kern_local, 2, sizeof(cl_mem), &buf_scores);
        p_clSetKernelArg(kern_local, 3, sizeof(int), &N);

        for (int w = 0; w < 5; w++)
            p_clEnqueueNDRangeKernel(queue, kern_local, 1, NULL, &global, &local, 0, NULL, NULL);
        p_clFinish(queue);

        int64_t gl_t0 = now_ns();
        for (int i = 0; i < gpu_iters; i++)
            p_clEnqueueNDRangeKernel(queue, kern_local, 1, NULL, &global, &local, 0, NULL, NULL);
        p_clFinish(queue);
        int64_t gl_t1 = now_ns();
        int64_t gpu_local_per = (gl_t1 - gl_t0) / gpu_iters;

        printf("       | GPU local: %5lld ns\n", (long long)gpu_local_per);

        p_clReleaseMemObject(buf_query);
        p_clReleaseMemObject(buf_nodes);
        p_clReleaseMemObject(buf_scores);
        free(nodes);
        free(gpu_scores);
        free(cpu_scores);
    }

    p_clReleaseKernel(kern);
    p_clReleaseKernel(kern_local);
    p_clReleaseProgram(prog);
    p_clReleaseCommandQueue(queue);
    p_clReleaseContext(ctx);
    dlclose(lib);

    printf("\nDone.\n");
    return 0;
}
