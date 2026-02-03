/*
 * GPU Probe - Query PowerVR GPU capabilities via OpenCL
 * For Moto G Power 5G (Dimensity 7020 / IMG BXM-8-256)
 */

#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

// OpenCL types (minimal definitions to avoid headers)
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_uint cl_device_type;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_PLATFORM_NAME 0x0902
#define CL_PLATFORM_VENDOR 0x0903
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1023
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT 0x100A
#define CL_DEVICE_EXTENSIONS 0x1030

// Function pointer types
typedef cl_int (*clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*clGetPlatformInfo_t)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (*clGetDeviceIDs_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*clGetDeviceInfo_t)(cl_device_id, cl_device_info, size_t, void*, size_t*);

int main() {
    printf("\n");
    printf("========================================\n");
    printf("  PowerVR GPU Probe via OpenCL\n");
    printf("========================================\n\n");

    // Try to load OpenCL library
    void* libcl = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
    if (!libcl) {
        printf("ERROR: Cannot load libOpenCL.so: %s\n", dlerror());
        return 1;
    }
    printf("[OK] Loaded libOpenCL.so\n");

    // Get function pointers
    clGetPlatformIDs_t clGetPlatformIDs = (clGetPlatformIDs_t)dlsym(libcl, "clGetPlatformIDs");
    clGetPlatformInfo_t clGetPlatformInfo = (clGetPlatformInfo_t)dlsym(libcl, "clGetPlatformInfo");
    clGetDeviceIDs_t clGetDeviceIDs = (clGetDeviceIDs_t)dlsym(libcl, "clGetDeviceIDs");
    clGetDeviceInfo_t clGetDeviceInfo = (clGetDeviceInfo_t)dlsym(libcl, "clGetDeviceInfo");

    if (!clGetPlatformIDs || !clGetPlatformInfo || !clGetDeviceIDs || !clGetDeviceInfo) {
        printf("ERROR: Cannot find OpenCL functions\n");
        dlclose(libcl);
        return 1;
    }
    printf("[OK] Found OpenCL functions\n\n");

    // Get platforms
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        printf("ERROR: No OpenCL platforms found (err=%d)\n", err);
        dlclose(libcl);
        return 1;
    }
    printf("Found %u OpenCL platform(s)\n\n", num_platforms);

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    char buffer[1024];
    
    for (cl_uint p = 0; p < num_platforms; p++) {
        printf("--- Platform %u ---\n", p);
        
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
        printf("  Name:   %s\n", buffer);
        
        clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, sizeof(buffer), buffer, NULL);
        printf("  Vendor: %s\n", buffer);

        // Get devices
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            printf("  No devices found\n\n");
            continue;
        }

        cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (cl_uint d = 0; d < num_devices; d++) {
            printf("\n  --- Device %u ---\n", d);
            
            clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
            printf("    Name:             %s\n", buffer);
            
            clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
            printf("    Vendor:           %s\n", buffer);

            cl_uint compute_units = 0;
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
            printf("    Compute Units:    %u\n", compute_units);

            size_t workgroup_size = 0;
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
            printf("    Max Workgroup:    %zu\n", workgroup_size);

            cl_uint clock_freq = 0;
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_freq), &clock_freq, NULL);
            printf("    Clock (MHz):      %u\n", clock_freq);

            cl_ulong global_mem = 0;
            clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
            printf("    Global Memory:    %llu MB\n", global_mem / (1024*1024));

            cl_ulong local_mem = 0;
            clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
            printf("    Local Memory:     %llu KB\n", local_mem / 1024);

            cl_uint vec_width = 0;
            clGetDeviceInfo(devices[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(vec_width), &vec_width, NULL);
            printf("    Vector Width (f): %u\n", vec_width);

            clGetDeviceInfo(devices[d], CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, NULL);
            printf("    Extensions:       %.100s...\n", buffer);
        }

        free(devices);
        printf("\n");
    }

    free(platforms);
    dlclose(libcl);

    printf("========================================\n");
    printf("  Analysis for Cognitive Architecture\n");
    printf("========================================\n\n");
    printf("GPU can be used for:\n");
    printf("  1. Embedding generation (SGEMV)\n");
    printf("  2. Vector similarity (dot products)\n");
    printf("  3. Memory retrieval (parallel search)\n");
    printf("  4. Batch operations while CPU does inference\n");
    printf("\n");

    return 0;
}
