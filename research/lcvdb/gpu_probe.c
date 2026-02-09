/* Probe OpenCL capabilities on this device */
#define CL_TARGET_OPENCL_VERSION 200
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdint.h>

/* OpenCL type definitions (avoid needing headers) */
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_bitfield cl_device_type;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_PLATFORM_NAME 0x0902
#define CL_PLATFORM_VERSION 0x0901
#define CL_PLATFORM_VENDOR 0x0903
#define CL_PLATFORM_EXTENSIONS 0x0904
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DEVICE_VERSION 0x102F
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS 0x1003
#define CL_DEVICE_MAX_WORK_ITEM_SIZES 0x1005
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1023
#define CL_DEVICE_LOCAL_MEM_TYPE 0x1022
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_DEVICE_GLOBAL_MEM_CACHE_SIZE 0x101E
#define CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE 0x101D
#define CL_DEVICE_GLOBAL_MEM_CACHE_TYPE 0x101C
#define CL_DEVICE_EXTENSIONS 0x1030
#define CL_DEVICE_ADDRESS_BITS 0x100D
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT 0x1006
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT 0x100A
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR 0x1029
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT 0x1005
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_INT 0x1036
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT 0x103A
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR 0x1033
#define CL_DEVICE_IMAGE_SUPPORT 0x1016
#define CL_DEVICE_HALF_FP_CONFIG 0x1033
#define CL_DEVICE_SINGLE_FP_CONFIG 0x101B
#define CL_DEVICE_DOUBLE_FP_CONFIG 0x1032
#define CL_DEVICE_OPENCL_C_VERSION 0x103D
#define CL_DEVICE_SVM_CAPABILITIES 0x1053
#define CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE 0x1014
#define CL_DEVICE_MAX_CONSTANT_ARGS 0x1015

typedef cl_int (*clGetPlatformIDs_fn)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*clGetPlatformInfo_fn)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (*clGetDeviceIDs_fn)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*clGetDeviceInfo_fn)(cl_device_id, cl_device_info, size_t, void*, size_t*);

static void print_str(clGetDeviceInfo_fn fn, cl_device_id dev, cl_device_info param, const char *label) {
    char buf[4096] = {0};
    if (fn(dev, param, sizeof(buf), buf, NULL) == CL_SUCCESS)
        printf("  %-35s %s\n", label, buf);
}

static void print_uint(clGetDeviceInfo_fn fn, cl_device_id dev, cl_device_info param, const char *label) {
    cl_uint val = 0;
    if (fn(dev, param, sizeof(val), &val, NULL) == CL_SUCCESS)
        printf("  %-35s %u\n", label, val);
}

static void print_ulong(clGetDeviceInfo_fn fn, cl_device_id dev, cl_device_info param, const char *label) {
    cl_ulong val = 0;
    if (fn(dev, param, sizeof(val), &val, NULL) == CL_SUCCESS)
        printf("  %-35s %llu\n", label, (unsigned long long)val);
}

static void print_size(clGetDeviceInfo_fn fn, cl_device_id dev, cl_device_info param, const char *label) {
    size_t val = 0;
    if (fn(dev, param, sizeof(val), &val, NULL) == CL_SUCCESS)
        printf("  %-35s %zu\n", label, val);
}

int main(void) {
    printf("=== OpenCL GPU Probe ===\n\n");

    void *lib = dlopen("libOpenCL.so", RTLD_NOW);
    if (!lib) {
        printf("Failed to load libOpenCL.so: %s\n", dlerror());
        printf("Trying libPVROCL.so...\n");
        lib = dlopen("libPVROCL.so", RTLD_NOW);
        if (!lib) {
            printf("Failed: %s\n", dlerror());
            return 1;
        }
    }
    printf("Loaded OpenCL library OK\n\n");

    clGetPlatformIDs_fn clGetPlatformIDs = dlsym(lib, "clGetPlatformIDs");
    clGetPlatformInfo_fn clGetPlatformInfo = dlsym(lib, "clGetPlatformInfo");
    clGetDeviceIDs_fn clGetDeviceIDs = dlsym(lib, "clGetDeviceIDs");
    clGetDeviceInfo_fn clGetDeviceInfo = dlsym(lib, "clGetDeviceInfo");

    if (!clGetPlatformIDs || !clGetPlatformInfo || !clGetDeviceIDs || !clGetDeviceInfo) {
        printf("Missing OpenCL symbols\n");
        return 1;
    }

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    printf("Platforms: %u (err=%d)\n\n", num_platforms, err);

    cl_platform_id platforms[4];
    clGetPlatformIDs(num_platforms, platforms, NULL);

    for (cl_uint p = 0; p < num_platforms; p++) {
        char buf[1024];
        printf("--- Platform %u ---\n", p);
        if (clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(buf), buf, NULL) == 0)
            printf("  Name:    %s\n", buf);
        if (clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION, sizeof(buf), buf, NULL) == 0)
            printf("  Version: %s\n", buf);
        if (clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL) == 0)
            printf("  Vendor:  %s\n", buf);
        if (clGetPlatformInfo(platforms[p], CL_PLATFORM_EXTENSIONS, sizeof(buf), buf, NULL) == 0)
            printf("  Extensions: %s\n", buf);

        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        printf("  Devices: %u\n\n", num_devices);

        cl_device_id devices[4];
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (cl_uint d = 0; d < num_devices; d++) {
            printf("  --- Device %u ---\n", d);
            print_str(clGetDeviceInfo, devices[d], CL_DEVICE_NAME, "Name:");
            print_str(clGetDeviceInfo, devices[d], CL_DEVICE_VENDOR, "Vendor:");
            print_str(clGetDeviceInfo, devices[d], CL_DEVICE_VERSION, "Device Version:");
            print_str(clGetDeviceInfo, devices[d], CL_DRIVER_VERSION, "Driver Version:");
            print_str(clGetDeviceInfo, devices[d], CL_DEVICE_OPENCL_C_VERSION, "OpenCL C Version:");
            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, "Max Compute Units:");
            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, "Max Clock (MHz):");
            print_size(clGetDeviceInfo, devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, "Max Work Group Size:");
            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "Max Work Item Dims:");

            size_t wis[3] = {0};
            if (clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(wis), wis, NULL) == 0)
                printf("  %-35s %zu x %zu x %zu\n", "Max Work Item Sizes:", wis[0], wis[1], wis[2]);

            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_ADDRESS_BITS, "Address Bits:");
            print_ulong(clGetDeviceInfo, devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, "Global Mem Size:");
            print_ulong(clGetDeviceInfo, devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, "Max Mem Alloc:");
            print_ulong(clGetDeviceInfo, devices[d], CL_DEVICE_LOCAL_MEM_SIZE, "Local Mem Size:");
            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_LOCAL_MEM_TYPE, "Local Mem Type (1=local,2=global):");
            print_ulong(clGetDeviceInfo, devices[d], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "Global Cache Size:");
            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "Global Cacheline Size:");
            print_ulong(clGetDeviceInfo, devices[d], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "Max Constant Buffer:");
            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_MAX_CONSTANT_ARGS, "Max Constant Args:");
            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_IMAGE_SUPPORT, "Image Support:");
            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "Pref Vec Width char:");
            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "Pref Vec Width int:");
            print_uint(clGetDeviceInfo, devices[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "Pref Vec Width float:");
            print_ulong(clGetDeviceInfo, devices[d], CL_DEVICE_SINGLE_FP_CONFIG, "Single FP Config:");
            print_ulong(clGetDeviceInfo, devices[d], CL_DEVICE_DOUBLE_FP_CONFIG, "Double FP Config:");
            print_ulong(clGetDeviceInfo, devices[d], CL_DEVICE_SVM_CAPABILITIES, "SVM Capabilities:");
            print_str(clGetDeviceInfo, devices[d], CL_DEVICE_EXTENSIONS, "Extensions:");
            printf("\n");
        }
    }

    dlclose(lib);
    return 0;
}
