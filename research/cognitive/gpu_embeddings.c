/*
 * GPU Embeddings - OpenCL vector operations for memory retrieval
 * For Moto G Power 5G (Dimensity 7020 / IMG BXM-8-256)
 *
 * Operations:
 *   1. Batch dot products (query vs memory bank)
 *   2. Top-K similarity search
 *   3. FP16 embedding support
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>

// OpenCL types
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef size_t cl_size_t;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_mem;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_uint cl_device_type;
typedef cl_ulong cl_mem_flags;
typedef cl_uint cl_program_build_info;
typedef cl_uint cl_command_queue_properties;
typedef void (*cl_notify_fn)(const char*, const void*, size_t, void*);

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_PROGRAM_BUILD_LOG 0x1183

// OpenCL function pointers
typedef cl_int (*clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*clGetDeviceIDs_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*clCreateContext_t)(void*, cl_uint, const cl_device_id*, cl_notify_fn, void*, cl_int*);
typedef cl_command_queue (*clCreateCommandQueueWithProperties_t)(cl_context, cl_device_id, const cl_ulong*, cl_int*);
typedef cl_program (*clCreateProgramWithSource_t)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*clBuildProgram_t)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
typedef cl_int (*clGetProgramBuildInfo_t)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
typedef cl_kernel (*clCreateKernel_t)(cl_program, const char*, cl_int*);
typedef cl_mem (*clCreateBuffer_t)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (*clSetKernelArg_t)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*clEnqueueNDRangeKernel_t)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
typedef cl_int (*clEnqueueReadBuffer_t)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const void*, void*);
typedef cl_int (*clEnqueueWriteBuffer_t)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const void*, void*);
typedef cl_int (*clFinish_t)(cl_command_queue);
typedef cl_int (*clReleaseMemObject_t)(cl_mem);
typedef cl_int (*clReleaseKernel_t)(cl_kernel);
typedef cl_int (*clReleaseProgram_t)(cl_program);
typedef cl_int (*clReleaseCommandQueue_t)(cl_command_queue);
typedef cl_int (*clReleaseContext_t)(cl_context);

// Global CL function pointers
static clGetPlatformIDs_t clGetPlatformIDs;
static clGetDeviceIDs_t clGetDeviceIDs;
static clCreateContext_t clCreateContext;
static clCreateCommandQueueWithProperties_t clCreateCommandQueueWithProperties;
static clCreateProgramWithSource_t clCreateProgramWithSource;
static clBuildProgram_t clBuildProgram;
static clGetProgramBuildInfo_t clGetProgramBuildInfo;
static clCreateKernel_t clCreateKernel;
static clCreateBuffer_t clCreateBuffer;
static clSetKernelArg_t clSetKernelArg;
static clEnqueueNDRangeKernel_t clEnqueueNDRangeKernel;
static clEnqueueReadBuffer_t clEnqueueReadBuffer;
static clEnqueueWriteBuffer_t clEnqueueWriteBuffer;
static clFinish_t clFinish;
static clReleaseMemObject_t clReleaseMemObject;
static clReleaseKernel_t clReleaseKernel;
static clReleaseProgram_t clReleaseProgram;
static clReleaseCommandQueue_t clReleaseCommandQueue;
static clReleaseContext_t clReleaseContext;

// OpenCL kernel for batch dot products
const char* dot_product_kernel_src =
"__kernel void batch_dot_product(\n"
"    __global const float* query,\n"
"    __global const float* memory_bank,\n"
"    __global float* similarities,\n"
"    const int embedding_dim) {\n"
"    \n"
"    int mem_idx = get_global_id(0);\n"
"    float sum = 0.0f;\n"
"    \n"
"    // Compute dot product between query and memory[mem_idx]\n"
"    __global const float* mem_vec = memory_bank + mem_idx * embedding_dim;\n"
"    \n"
"    for (int i = 0; i < embedding_dim; i++) {\n"
"        sum += query[i] * mem_vec[i];\n"
"    }\n"
"    \n"
"    similarities[mem_idx] = sum;\n"
"}\n";

// More optimized version using local memory
const char* dot_product_optimized_src =
"__kernel void batch_dot_product_opt(\n"
"    __global const float* query,\n"
"    __global const float* memory_bank,\n"
"    __global float* similarities,\n"
"    const int embedding_dim,\n"
"    __local float* local_query) {\n"
"    \n"
"    int mem_idx = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    \n"
"    // Cooperatively load query into local memory\n"
"    for (int i = local_id; i < embedding_dim; i += local_size) {\n"
"        local_query[i] = query[i];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    \n"
"    // Compute dot product\n"
"    float sum = 0.0f;\n"
"    __global const float* mem_vec = memory_bank + mem_idx * embedding_dim;\n"
"    \n"
"    for (int i = 0; i < embedding_dim; i++) {\n"
"        sum += local_query[i] * mem_vec[i];\n"
"    }\n"
"    \n"
"    similarities[mem_idx] = sum;\n"
"}\n";

// CPU reference implementation
void cpu_batch_dot_product(const float* query, const float* memory_bank, 
                           float* similarities, int num_memories, int embedding_dim) {
    for (int m = 0; m < num_memories; m++) {
        float sum = 0.0f;
        const float* mem_vec = memory_bank + m * embedding_dim;
        for (int i = 0; i < embedding_dim; i++) {
            sum += query[i] * mem_vec[i];
        }
        similarities[m] = sum;
    }
}

// Find top-k indices (simple CPU implementation)
void find_top_k(const float* similarities, int n, int k, int* indices, float* scores) {
    // Simple selection sort for small k
    for (int i = 0; i < k; i++) {
        float max_val = -1e9f;
        int max_idx = -1;
        for (int j = 0; j < n; j++) {
            // Skip already selected
            int skip = 0;
            for (int p = 0; p < i; p++) {
                if (indices[p] == j) { skip = 1; break; }
            }
            if (skip) continue;
            
            if (similarities[j] > max_val) {
                max_val = similarities[j];
                max_idx = j;
            }
        }
        indices[i] = max_idx;
        scores[i] = max_val;
    }
}

static long get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

int main() {
    printf("\n");
    printf("========================================\n");
    printf("  GPU Embeddings - Vector Similarity\n");
    printf("========================================\n\n");

    // Configuration - larger memory bank to test GPU advantage
    const int embedding_dim = 512;  // Typical embedding size
    const int num_memories = 16384; // 16K memories - more realistic scale
    const int top_k = 5;            // Retrieve top-k memories

    printf("Configuration:\n");
    printf("  Embedding dim:  %d\n", embedding_dim);
    printf("  Memory bank:    %d vectors\n", num_memories);
    printf("  Top-K:          %d\n", top_k);
    printf("  Memory size:    %.2f MB\n", 
           (float)(num_memories * embedding_dim * sizeof(float)) / (1024*1024));
    printf("\n");

    // Load OpenCL
    void* libcl = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
    if (!libcl) {
        printf("ERROR: Cannot load OpenCL: %s\n", dlerror());
        return 1;
    }

    // Get function pointers
    clGetPlatformIDs = (clGetPlatformIDs_t)dlsym(libcl, "clGetPlatformIDs");
    clGetDeviceIDs = (clGetDeviceIDs_t)dlsym(libcl, "clGetDeviceIDs");
    clCreateContext = (clCreateContext_t)dlsym(libcl, "clCreateContext");
    clCreateCommandQueueWithProperties = (clCreateCommandQueueWithProperties_t)dlsym(libcl, "clCreateCommandQueueWithProperties");
    clCreateProgramWithSource = (clCreateProgramWithSource_t)dlsym(libcl, "clCreateProgramWithSource");
    clBuildProgram = (clBuildProgram_t)dlsym(libcl, "clBuildProgram");
    clGetProgramBuildInfo = (clGetProgramBuildInfo_t)dlsym(libcl, "clGetProgramBuildInfo");
    clCreateKernel = (clCreateKernel_t)dlsym(libcl, "clCreateKernel");
    clCreateBuffer = (clCreateBuffer_t)dlsym(libcl, "clCreateBuffer");
    clSetKernelArg = (clSetKernelArg_t)dlsym(libcl, "clSetKernelArg");
    clEnqueueNDRangeKernel = (clEnqueueNDRangeKernel_t)dlsym(libcl, "clEnqueueNDRangeKernel");
    clEnqueueReadBuffer = (clEnqueueReadBuffer_t)dlsym(libcl, "clEnqueueReadBuffer");
    clEnqueueWriteBuffer = (clEnqueueWriteBuffer_t)dlsym(libcl, "clEnqueueWriteBuffer");
    clFinish = (clFinish_t)dlsym(libcl, "clFinish");
    clReleaseMemObject = (clReleaseMemObject_t)dlsym(libcl, "clReleaseMemObject");
    clReleaseKernel = (clReleaseKernel_t)dlsym(libcl, "clReleaseKernel");
    clReleaseProgram = (clReleaseProgram_t)dlsym(libcl, "clReleaseProgram");
    clReleaseCommandQueue = (clReleaseCommandQueue_t)dlsym(libcl, "clReleaseCommandQueue");
    clReleaseContext = (clReleaseContext_t)dlsym(libcl, "clReleaseContext");

    printf("[OK] Loaded OpenCL functions\n");

    // Initialize OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: Failed to create context: %d\n", err);
        return 1;
    }
    
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: Failed to create command queue: %d\n", err);
        return 1;
    }
    
    printf("[OK] Created OpenCL context and queue\n");

    // Build kernel
    cl_program program = clCreateProgramWithSource(context, 1, &dot_product_kernel_src, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("ERROR: Build failed: %s\n", log);
        return 1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "batch_dot_product", &err);
    if (err != CL_SUCCESS) {
        printf("ERROR: Failed to create kernel: %d\n", err);
        return 1;
    }
    
    printf("[OK] Compiled OpenCL kernel\n\n");

    // Allocate and initialize data
    float* query = (float*)malloc(embedding_dim * sizeof(float));
    float* memory_bank = (float*)malloc(num_memories * embedding_dim * sizeof(float));
    float* similarities_gpu = (float*)malloc(num_memories * sizeof(float));
    float* similarities_cpu = (float*)malloc(num_memories * sizeof(float));
    int* top_indices = (int*)malloc(top_k * sizeof(int));
    float* top_scores = (float*)malloc(top_k * sizeof(float));

    // Initialize with random data (normalized vectors)
    srand(42);
    for (int i = 0; i < embedding_dim; i++) {
        query[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < num_memories * embedding_dim; i++) {
        memory_bank[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // Create GPU buffers
    cl_mem query_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      embedding_dim * sizeof(float), query, &err);
    cl_mem memory_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       num_memories * embedding_dim * sizeof(float), memory_bank, &err);
    cl_mem sim_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    num_memories * sizeof(float), NULL, &err);

    printf("========================================\n");
    printf("  Benchmark: CPU vs GPU\n");
    printf("========================================\n\n");

    // CPU benchmark
    long cpu_start = get_time_us();
    for (int iter = 0; iter < 100; iter++) {
        cpu_batch_dot_product(query, memory_bank, similarities_cpu, num_memories, embedding_dim);
    }
    long cpu_time = get_time_us() - cpu_start;
    printf("CPU (100 iterations):\n");
    printf("  Total time:   %ld us\n", cpu_time);
    printf("  Per query:    %.1f us\n", cpu_time / 100.0f);
    printf("\n");

    // GPU benchmark
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &query_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &memory_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sim_buf);
    clSetKernelArg(kernel, 3, sizeof(int), &embedding_dim);

    size_t global_size = num_memories;
    size_t local_size = 64;

    // Warmup
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    clFinish(queue);

    long gpu_start = get_time_us();
    for (int iter = 0; iter < 100; iter++) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    }
    clFinish(queue);
    long gpu_time = get_time_us() - gpu_start;
    
    // Read back results
    clEnqueueReadBuffer(queue, sim_buf, 1, 0, num_memories * sizeof(float), similarities_gpu, 0, NULL, NULL);

    printf("GPU (100 iterations):\n");
    printf("  Total time:   %ld us\n", gpu_time);
    printf("  Per query:    %.1f us\n", gpu_time / 100.0f);
    printf("  Speedup:      %.2fx\n", (float)cpu_time / gpu_time);
    printf("\n");

    // Verify correctness
    float max_diff = 0.0f;
    for (int i = 0; i < num_memories; i++) {
        float diff = fabsf(similarities_gpu[i] - similarities_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Correctness check:\n");
    printf("  Max difference: %.6f\n", max_diff);
    printf("  Status:         %s\n", max_diff < 0.001f ? "PASS" : "FAIL");
    printf("\n");

    // Find top-k memories
    find_top_k(similarities_gpu, num_memories, top_k, top_indices, top_scores);
    
    printf("========================================\n");
    printf("  Top-%d Retrieved Memories\n", top_k);
    printf("========================================\n\n");
    for (int i = 0; i < top_k; i++) {
        printf("  %d. Memory[%4d] score=%.4f\n", i+1, top_indices[i], top_scores[i]);
    }
    printf("\n");

    // Calculate throughput
    float ops_per_query = 2.0f * num_memories * embedding_dim;  // multiply-add
    float gflops_cpu = (ops_per_query * 100.0f) / (cpu_time * 1000.0f);
    float gflops_gpu = (ops_per_query * 100.0f) / (gpu_time * 1000.0f);
    
    printf("========================================\n");
    printf("  Throughput Analysis\n");
    printf("========================================\n\n");
    printf("  CPU: %.2f GFLOPS\n", gflops_cpu);
    printf("  GPU: %.2f GFLOPS\n", gflops_gpu);
    printf("\n");
    printf("  For cognitive architecture:\n");
    printf("  - GPU handles memory retrieval in parallel with CPU inference\n");
    printf("  - At %.1f us/query, can search %d memories while\n", 
           gpu_time / 100.0f, num_memories);
    printf("    CPU generates first few tokens\n");
    printf("\n");

    // Cleanup
    clReleaseMemObject(query_buf);
    clReleaseMemObject(memory_buf);
    clReleaseMemObject(sim_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    dlclose(libcl);
    
    free(query);
    free(memory_bank);
    free(similarities_gpu);
    free(similarities_cpu);
    free(top_indices);
    free(top_scores);

    return 0;
}
