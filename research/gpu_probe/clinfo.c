/*
 * clinfo.c - OpenCL device capability enumeration via dlopen
 * Cross-compile for aarch64, run on Android device
 * Dynamically loads libOpenCL.so to avoid link-time dependencies
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>

/* Minimal OpenCL type definitions */
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef size_t   cl_size_t;
typedef intptr_t cl_platform_id;
typedef intptr_t cl_device_id;
typedef uint64_t cl_device_type;
typedef uint32_t cl_platform_info;
typedef uint32_t cl_device_info;
typedef uint64_t cl_device_fp_config;
typedef uint64_t cl_device_exec_capabilities;
typedef uint64_t cl_command_queue_properties;
typedef uint64_t cl_device_svm_capabilities;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF

/* Platform info */
#define CL_PLATFORM_PROFILE    0x0900
#define CL_PLATFORM_VERSION    0x0901
#define CL_PLATFORM_NAME       0x0902
#define CL_PLATFORM_VENDOR     0x0903
#define CL_PLATFORM_EXTENSIONS 0x0904

/* Device info */
#define CL_DEVICE_TYPE                          0x1000
#define CL_DEVICE_VENDOR_ID                     0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS             0x1002
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS      0x1003
#define CL_DEVICE_MAX_WORK_GROUP_SIZE            0x1004
#define CL_DEVICE_MAX_WORK_ITEM_SIZES            0x1005
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR   0x1006
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT  0x1007
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT    0x1008
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG   0x1009
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT  0x100A
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE 0x100B
#define CL_DEVICE_MAX_CLOCK_FREQUENCY           0x100C
#define CL_DEVICE_ADDRESS_BITS                  0x100D
#define CL_DEVICE_MAX_READ_IMAGE_ARGS           0x100E
#define CL_DEVICE_MAX_WRITE_IMAGE_ARGS          0x100F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE            0x1010
#define CL_DEVICE_IMAGE2D_MAX_WIDTH             0x1011
#define CL_DEVICE_IMAGE2D_MAX_HEIGHT            0x1012
#define CL_DEVICE_IMAGE3D_MAX_WIDTH             0x1013
#define CL_DEVICE_IMAGE3D_MAX_HEIGHT            0x1014
#define CL_DEVICE_IMAGE3D_MAX_DEPTH             0x1015
#define CL_DEVICE_IMAGE_SUPPORT                 0x1016
#define CL_DEVICE_MAX_PARAMETER_SIZE            0x1017
#define CL_DEVICE_MAX_SAMPLERS                  0x1018
#define CL_DEVICE_MEM_BASE_ADDR_ALIGN           0x1019
#define CL_DEVICE_SINGLE_FP_CONFIG              0x101B
#define CL_DEVICE_GLOBAL_MEM_CACHE_TYPE         0x101C
#define CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE     0x101D
#define CL_DEVICE_GLOBAL_MEM_CACHE_SIZE         0x101E
#define CL_DEVICE_GLOBAL_MEM_SIZE               0x101F
#define CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE      0x1020
#define CL_DEVICE_MAX_CONSTANT_ARGS             0x1021
#define CL_DEVICE_LOCAL_MEM_TYPE                 0x1022
#define CL_DEVICE_LOCAL_MEM_SIZE                 0x1023
#define CL_DEVICE_ERROR_CORRECTION_SUPPORT      0x1024
#define CL_DEVICE_PROFILING_TIMER_RESOLUTION    0x1025
#define CL_DEVICE_ENDIAN_LITTLE                 0x1026
#define CL_DEVICE_AVAILABLE                     0x1027
#define CL_DEVICE_COMPILER_AVAILABLE            0x1028
#define CL_DEVICE_EXECUTION_CAPABILITIES        0x1029
#define CL_DEVICE_QUEUE_PROPERTIES              0x102A
#define CL_DEVICE_NAME                          0x102B
#define CL_DEVICE_VENDOR                        0x102C
#define CL_DRIVER_VERSION                       0x102D
#define CL_DEVICE_PROFILE                       0x102E
#define CL_DEVICE_VERSION                       0x102F
#define CL_DEVICE_EXTENSIONS                    0x1030
#define CL_DEVICE_PLATFORM                      0x1031
#define CL_DEVICE_DOUBLE_FP_CONFIG              0x1032
#define CL_DEVICE_HALF_FP_CONFIG                0x1033
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF   0x1034
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR      0x1036
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT     0x1037
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_INT       0x1038
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG      0x1039
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT     0x103A
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE    0x103B
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF      0x103C
#define CL_DEVICE_OPENCL_C_VERSION              0x103D
#define CL_DEVICE_IMAGE_MAX_BUFFER_SIZE         0x1040
#define CL_DEVICE_IMAGE_MAX_ARRAY_SIZE          0x1041
#define CL_DEVICE_PRINTF_BUFFER_SIZE            0x1049
#define CL_DEVICE_SVM_CAPABILITIES              0x1053

/* FP config bits */
#define CL_FP_DENORM           (1 << 0)
#define CL_FP_INF_NAN          (1 << 1)
#define CL_FP_ROUND_TO_NEAREST (1 << 2)
#define CL_FP_ROUND_TO_ZERO    (1 << 3)
#define CL_FP_ROUND_TO_INF     (1 << 4)
#define CL_FP_FMA              (1 << 5)
#define CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT (1 << 6)
#define CL_FP_SOFT_FLOAT       (1 << 6)

/* SVM capability bits */
#define CL_DEVICE_SVM_COARSE_GRAIN_BUFFER (1 << 0)
#define CL_DEVICE_SVM_FINE_GRAIN_BUFFER   (1 << 1)
#define CL_DEVICE_SVM_FINE_GRAIN_SYSTEM   (1 << 2)
#define CL_DEVICE_SVM_ATOMICS             (1 << 3)

/* Function pointer types */
typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*pfn_clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*pfn_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);

static pfn_clGetPlatformIDs  fn_clGetPlatformIDs;
static pfn_clGetPlatformInfo fn_clGetPlatformInfo;
static pfn_clGetDeviceIDs    fn_clGetDeviceIDs;
static pfn_clGetDeviceInfo   fn_clGetDeviceInfo;

static char* get_string(cl_device_id dev, cl_device_info param) {
    size_t sz;
    if (fn_clGetDeviceInfo(dev, param, 0, NULL, &sz) != CL_SUCCESS) return strdup("(error)");
    char *buf = malloc(sz + 1);
    fn_clGetDeviceInfo(dev, param, sz, buf, NULL);
    buf[sz] = '\0';
    return buf;
}

static char* get_platform_string(cl_platform_id plat, cl_platform_info param) {
    size_t sz;
    if (fn_clGetPlatformInfo(plat, param, 0, NULL, &sz) != CL_SUCCESS) return strdup("(error)");
    char *buf = malloc(sz + 1);
    fn_clGetPlatformInfo(plat, param, sz, buf, NULL);
    buf[sz] = '\0';
    return buf;
}

static void print_fp_config(const char *label, cl_device_fp_config fp) {
    printf("  %-40s", label);
    if (fp == 0) { printf("(none)\n"); return; }
    if (fp & CL_FP_DENORM) printf("Denorm ");
    if (fp & CL_FP_INF_NAN) printf("InfNaN ");
    if (fp & CL_FP_ROUND_TO_NEAREST) printf("RoundNearest ");
    if (fp & CL_FP_ROUND_TO_ZERO) printf("RoundZero ");
    if (fp & CL_FP_ROUND_TO_INF) printf("RoundInf ");
    if (fp & CL_FP_FMA) printf("FMA ");
    printf("\n");
}

static void print_svm_caps(cl_device_svm_capabilities svm) {
    printf("  %-40s", "SVM capabilities:");
    if (svm == 0) { printf("(none)\n"); return; }
    if (svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) printf("CoarseGrain ");
    if (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) printf("FineGrainBuffer ");
    if (svm & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) printf("FineGrainSystem ");
    if (svm & CL_DEVICE_SVM_ATOMICS) printf("Atomics ");
    printf("\n");
}

int main(int argc, char **argv) {
    void *lib;
    const char *paths[] = {
        "libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        "/vendor/lib64/mt6855/libPVROCL.so",
        NULL
    };

    printf("=== OpenCL Device Info ===\n\n");

    for (int i = 0; paths[i]; i++) {
        lib = dlopen(paths[i], RTLD_NOW);
        if (lib) {
            printf("Loaded: %s\n", paths[i]);
            break;
        }
    }
    if (!lib) {
        fprintf(stderr, "Failed to load OpenCL library: %s\n", dlerror());
        return 1;
    }

    fn_clGetPlatformIDs  = (pfn_clGetPlatformIDs)  dlsym(lib, "clGetPlatformIDs");
    fn_clGetPlatformInfo = (pfn_clGetPlatformInfo)  dlsym(lib, "clGetPlatformInfo");
    fn_clGetDeviceIDs    = (pfn_clGetDeviceIDs)     dlsym(lib, "clGetDeviceIDs");
    fn_clGetDeviceInfo   = (pfn_clGetDeviceInfo)    dlsym(lib, "clGetDeviceInfo");

    if (!fn_clGetPlatformIDs || !fn_clGetPlatformInfo || !fn_clGetDeviceIDs || !fn_clGetDeviceInfo) {
        fprintf(stderr, "Failed to resolve OpenCL functions: %s\n", dlerror());
        return 1;
    }

    cl_uint num_plat = 0;
    cl_int err = fn_clGetPlatformIDs(0, NULL, &num_plat);
    if (err != CL_SUCCESS || num_plat == 0) {
        fprintf(stderr, "No OpenCL platforms found (err=%d)\n", err);
        return 1;
    }
    printf("Number of platforms: %u\n\n", num_plat);

    cl_platform_id *platforms = malloc(num_plat * sizeof(cl_platform_id));
    fn_clGetPlatformIDs(num_plat, platforms, NULL);

    for (cl_uint p = 0; p < num_plat; p++) {
        char *pname    = get_platform_string(platforms[p], CL_PLATFORM_NAME);
        char *pvendor  = get_platform_string(platforms[p], CL_PLATFORM_VENDOR);
        char *pversion = get_platform_string(platforms[p], CL_PLATFORM_VERSION);
        char *pprofile = get_platform_string(platforms[p], CL_PLATFORM_PROFILE);
        char *pext     = get_platform_string(platforms[p], CL_PLATFORM_EXTENSIONS);

        printf("--- Platform %u ---\n", p);
        printf("  Name:       %s\n", pname);
        printf("  Vendor:     %s\n", pvendor);
        printf("  Version:    %s\n", pversion);
        printf("  Profile:    %s\n", pprofile);
        printf("  Extensions: %s\n\n", pext);
        free(pname); free(pvendor); free(pversion); free(pprofile); free(pext);

        cl_uint num_dev = 0;
        fn_clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_dev);
        if (num_dev == 0) { printf("  (no devices)\n\n"); continue; }

        cl_device_id *devices = malloc(num_dev * sizeof(cl_device_id));
        fn_clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_dev, devices, NULL);

        for (cl_uint d = 0; d < num_dev; d++) {
            cl_device_id dev = devices[d];
            printf("  --- Device %u ---\n", d);

            /* Strings */
            char *s;
            s = get_string(dev, CL_DEVICE_NAME);    printf("  %-40s %s\n", "Name:", s); free(s);
            s = get_string(dev, CL_DEVICE_VENDOR);   printf("  %-40s %s\n", "Vendor:", s); free(s);
            s = get_string(dev, CL_DEVICE_VERSION);  printf("  %-40s %s\n", "Device version:", s); free(s);
            s = get_string(dev, CL_DRIVER_VERSION);  printf("  %-40s %s\n", "Driver version:", s); free(s);
            s = get_string(dev, CL_DEVICE_PROFILE);  printf("  %-40s %s\n", "Profile:", s); free(s);
            s = get_string(dev, CL_DEVICE_OPENCL_C_VERSION); printf("  %-40s %s\n", "OpenCL C version:", s); free(s);

            /* Integers */
            cl_uint val_u;
            cl_ulong val_ul;
            size_t val_sz;
            cl_uint val_b;

            fn_clGetDeviceInfo(dev, CL_DEVICE_VENDOR_ID, sizeof(val_u), &val_u, NULL);
            printf("  %-40s 0x%04X\n", "Vendor ID:", val_u);

            fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(val_u), &val_u, NULL);
            printf("  %-40s %u\n", "Max compute units:", val_u);

            fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(val_u), &val_u, NULL);
            printf("  %-40s %u MHz\n", "Max clock frequency:", val_u);

            fn_clGetDeviceInfo(dev, CL_DEVICE_ADDRESS_BITS, sizeof(val_u), &val_u, NULL);
            printf("  %-40s %u\n", "Address bits:", val_u);

            /* Work dimensions */
            cl_uint max_dims;
            fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_dims), &max_dims, NULL);
            printf("  %-40s %u\n", "Max work item dimensions:", max_dims);

            size_t *work_sizes = malloc(max_dims * sizeof(size_t));
            fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_dims * sizeof(size_t), work_sizes, NULL);
            printf("  %-40s", "Max work item sizes:");
            for (cl_uint i = 0; i < max_dims; i++) printf(" %zu", work_sizes[i]);
            printf("\n");
            free(work_sizes);

            fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(val_sz), &val_sz, NULL);
            printf("  %-40s %zu\n", "Max work group size:", val_sz);

            /* Vector widths */
            printf("\n  --- Vector Widths (preferred / native) ---\n");
            struct { const char *name; cl_device_info pref; cl_device_info nat; } vw[] = {
                {"char",   0x1006, 0x1036}, {"short",  0x1007, 0x1037},
                {"int",    0x1008, 0x1038}, {"long",   0x1009, 0x1039},
                {"float",  0x100A, 0x103A}, {"double", 0x100B, 0x103B},
                {"half",   0x1034, 0x103C},
            };
            for (int i = 0; i < 7; i++) {
                cl_uint pref = 0, nat = 0;
                fn_clGetDeviceInfo(dev, vw[i].pref, sizeof(pref), &pref, NULL);
                fn_clGetDeviceInfo(dev, vw[i].nat, sizeof(nat), &nat, NULL);
                printf("  %-40s %u / %u\n", vw[i].name, pref, nat);
            }

            /* FP configs */
            printf("\n  --- Floating Point ---\n");
            cl_device_fp_config fp;
            fn_clGetDeviceInfo(dev, CL_DEVICE_HALF_FP_CONFIG, sizeof(fp), &fp, NULL);
            print_fp_config("Half precision (cl_khr_fp16):", fp);
            fn_clGetDeviceInfo(dev, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fp), &fp, NULL);
            print_fp_config("Single precision:", fp);
            fn_clGetDeviceInfo(dev, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp), &fp, NULL);
            print_fp_config("Double precision (cl_khr_fp64):", fp);

            /* Memory */
            printf("\n  --- Memory ---\n");
            fn_clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(val_ul), &val_ul, NULL);
            printf("  %-40s %lu MB\n", "Global memory size:", (unsigned long)(val_ul / (1024*1024)));
            fn_clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(val_ul), &val_ul, NULL);
            printf("  %-40s %lu KB\n", "Global memory cache size:", (unsigned long)(val_ul / 1024));
            fn_clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(val_u), &val_u, NULL);
            printf("  %-40s %u bytes\n", "Global memory cacheline size:", val_u);
            fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(val_ul), &val_ul, NULL);
            printf("  %-40s %lu MB\n", "Max memory alloc size:", (unsigned long)(val_ul / (1024*1024)));
            fn_clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(val_ul), &val_ul, NULL);
            printf("  %-40s %lu KB\n", "Local memory size:", (unsigned long)(val_ul / 1024));
            fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(val_ul), &val_ul, NULL);
            printf("  %-40s %lu KB\n", "Max constant buffer size:", (unsigned long)(val_ul / 1024));
            fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(val_u), &val_u, NULL);
            printf("  %-40s %u\n", "Max constant args:", val_u);
            fn_clGetDeviceInfo(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(val_u), &val_u, NULL);
            printf("  %-40s %u bits\n", "Memory base addr align:", val_u);

            /* Images */
            printf("\n  --- Image Support ---\n");
            fn_clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(val_b), &val_b, NULL);
            printf("  %-40s %s\n", "Image support:", val_b ? "YES" : "NO");
            if (val_b) {
                fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(val_u), &val_u, NULL);
                printf("  %-40s %u\n", "Max read image args:", val_u);
                fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(val_u), &val_u, NULL);
                printf("  %-40s %u\n", "Max write image args:", val_u);
                fn_clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(val_sz), &val_sz, NULL);
                printf("  %-40s %zu\n", "Image2D max width:", val_sz);
                fn_clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(val_sz), &val_sz, NULL);
                printf("  %-40s %zu\n", "Image2D max height:", val_sz);
                fn_clGetDeviceInfo(dev, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(val_sz), &val_sz, NULL);
                printf("  %-40s %zu\n", "Image3D max width:", val_sz);
                fn_clGetDeviceInfo(dev, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(val_sz), &val_sz, NULL);
                printf("  %-40s %zu\n", "Image3D max height:", val_sz);
                fn_clGetDeviceInfo(dev, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(val_sz), &val_sz, NULL);
                printf("  %-40s %zu\n", "Image3D max depth:", val_sz);
                fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_SAMPLERS, sizeof(val_u), &val_u, NULL);
                printf("  %-40s %u\n", "Max samplers:", val_u);
            }

            /* SVM (OpenCL 2.0) */
            printf("\n  --- OpenCL 2.0 Features ---\n");
            cl_device_svm_capabilities svm = 0;
            cl_int svm_err = fn_clGetDeviceInfo(dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm), &svm, NULL);
            if (svm_err == CL_SUCCESS) {
                print_svm_caps(svm);
            } else {
                printf("  %-40s (not supported, err=%d)\n", "SVM capabilities:", svm_err);
            }

            fn_clGetDeviceInfo(dev, CL_DEVICE_PRINTF_BUFFER_SIZE, sizeof(val_sz), &val_sz, NULL);
            printf("  %-40s %zu KB\n", "Printf buffer size:", val_sz / 1024);

            /* Misc */
            printf("\n  --- Misc ---\n");
            fn_clGetDeviceInfo(dev, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(val_sz), &val_sz, NULL);
            printf("  %-40s %zu bytes\n", "Max kernel parameter size:", val_sz);
            fn_clGetDeviceInfo(dev, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(val_sz), &val_sz, NULL);
            printf("  %-40s %zu ns\n", "Profiling timer resolution:", val_sz);
            fn_clGetDeviceInfo(dev, CL_DEVICE_ENDIAN_LITTLE, sizeof(val_b), &val_b, NULL);
            printf("  %-40s %s\n", "Little endian:", val_b ? "YES" : "NO");
            fn_clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(val_b), &val_b, NULL);
            printf("  %-40s %s\n", "Device available:", val_b ? "YES" : "NO");
            fn_clGetDeviceInfo(dev, CL_DEVICE_COMPILER_AVAILABLE, sizeof(val_b), &val_b, NULL);
            printf("  %-40s %s\n", "Compiler available:", val_b ? "YES" : "NO");

            /* Extensions */
            s = get_string(dev, CL_DEVICE_EXTENSIONS);
            printf("\n  --- Extensions ---\n");
            /* Print each extension on its own line */
            char *tok = strtok(s, " ");
            while (tok) {
                printf("    %s\n", tok);
                tok = strtok(NULL, " ");
            }
            free(s);
            printf("\n");
        }
        free(devices);
    }
    free(platforms);
    dlclose(lib);
    return 0;
}
