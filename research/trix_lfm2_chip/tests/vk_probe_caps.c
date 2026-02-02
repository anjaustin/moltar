/*
 * vk_probe_caps — Query Vulkan device capabilities on the Motorola
 *
 * Reports: API version, device limits, extensions, subgroup properties,
 * texture format support, max workgroup size, shared memory size.
 *
 * Cross-compile:
 *   NDK=~/Library/Android/sdk/ndk/28.2.13676358
 *   $NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang \
 *       -O2 -o vk_probe_caps tests/vk_probe_caps.c -lvulkan
 *   adb push vk_probe_caps /data/local/tmp/
 *   adb shell /data/local/tmp/vk_probe_caps
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK_VK(call) do { \
    VkResult r = (call); \
    if (r != VK_SUCCESS) { \
        fprintf(stderr, "Vulkan error %d at %s:%d\n", r, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

int main(void) {
    printf("=== Vulkan Capability Probe ===\n\n");

    /* Instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vk_probe_caps",
        .applicationVersion = 1,
        .pEngineName = "trix",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    VkInstance instance;
    CHECK_VK(vkCreateInstance(&inst_info, NULL, &instance));

    /* Physical device */
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance, &dev_count, NULL);
    printf("Physical devices: %u\n", dev_count);
    if (dev_count == 0) { fprintf(stderr, "No Vulkan devices\n"); return 1; }

    VkPhysicalDevice *devs = malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &dev_count, devs);
    VkPhysicalDevice gpu = devs[0];

    /* Device properties */
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);
    printf("\n--- Device Properties ---\n");
    printf("  Device: %s\n", props.deviceName);
    printf("  API version: %u.%u.%u\n",
           VK_VERSION_MAJOR(props.apiVersion),
           VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));
    printf("  Driver version: %u\n", props.driverVersion);
    printf("  Vendor ID: 0x%04x\n", props.vendorID);
    printf("  Device ID: 0x%04x\n", props.deviceID);
    printf("  Device type: %d (0=other,1=integrated,2=discrete,3=virtual,4=cpu)\n",
           props.deviceType);

    /* Limits */
    VkPhysicalDeviceLimits *lim = &props.limits;
    printf("\n--- Key Limits ---\n");
    printf("  maxComputeWorkGroupCount: [%u, %u, %u]\n",
           lim->maxComputeWorkGroupCount[0],
           lim->maxComputeWorkGroupCount[1],
           lim->maxComputeWorkGroupCount[2]);
    printf("  maxComputeWorkGroupSize: [%u, %u, %u]\n",
           lim->maxComputeWorkGroupSize[0],
           lim->maxComputeWorkGroupSize[1],
           lim->maxComputeWorkGroupSize[2]);
    printf("  maxComputeWorkGroupInvocations: %u\n",
           lim->maxComputeWorkGroupInvocations);
    printf("  maxComputeSharedMemorySize: %u bytes\n",
           lim->maxComputeSharedMemorySize);
    printf("  maxStorageBufferRange: %u bytes\n",
           lim->maxStorageBufferRange);
    printf("  maxPushConstantsSize: %u bytes\n",
           lim->maxPushConstantsSize);
    printf("  maxBoundDescriptorSets: %u\n",
           lim->maxBoundDescriptorSets);
    printf("  maxImageDimension1D: %u\n", lim->maxImageDimension1D);
    printf("  maxImageDimension2D: %u\n", lim->maxImageDimension2D);
    printf("  maxSamplerAllocationCount: %u\n", lim->maxSamplerAllocationCount);
    printf("  maxTexelBufferElements: %u\n", lim->maxTexelBufferElements);
    printf("  timestampComputeAndGraphics: %u\n",
           lim->timestampComputeAndGraphics);
    printf("  timestampPeriod: %.2f ns\n", lim->timestampPeriod);

    /* Subgroup properties (Vulkan 1.1) */
    VkPhysicalDeviceSubgroupProperties subgroup_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup_props,
    };
    vkGetPhysicalDeviceProperties2(gpu, &props2);
    printf("\n--- Subgroup Properties ---\n");
    printf("  subgroupSize: %u\n", subgroup_props.subgroupSize);
    printf("  supportedStages: 0x%08x\n", subgroup_props.supportedStages);
    printf("    COMPUTE: %s\n",
           (subgroup_props.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ? "yes" : "no");
    printf("  supportedOperations: 0x%08x\n", subgroup_props.supportedOperations);
    printf("    BASIC: %s\n",
           (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) ? "yes" : "no");
    printf("    VOTE: %s\n",
           (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) ? "yes" : "no");
    printf("    ARITHMETIC: %s\n",
           (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ? "yes" : "no");
    printf("    BALLOT: %s\n",
           (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) ? "yes" : "no");
    printf("    SHUFFLE: %s\n",
           (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) ? "yes" : "no");
    printf("    SHUFFLE_RELATIVE: %s\n",
           (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) ? "yes" : "no");
    printf("    CLUSTERED: %s\n",
           (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) ? "yes" : "no");
    printf("    QUAD: %s\n",
           (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) ? "yes" : "no");

    /* Device features */
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(gpu, &features);
    printf("\n--- Key Features ---\n");
    printf("  shaderFloat64: %s\n", features.shaderFloat64 ? "yes" : "no");
    printf("  shaderInt64: %s\n", features.shaderInt64 ? "yes" : "no");
    printf("  shaderInt16: %s\n", features.shaderInt16 ? "yes" : "no");
    printf("  shaderStorageImageReadWithoutFormat: %s\n",
           features.shaderStorageImageReadWithoutFormat ? "yes" : "no");
    printf("  shaderStorageImageWriteWithoutFormat: %s\n",
           features.shaderStorageImageWriteWithoutFormat ? "yes" : "no");
    printf("  textureCompressionASTC_LDR: %s\n",
           features.textureCompressionASTC_LDR ? "yes" : "no");
    printf("  textureCompressionETC2: %s\n",
           features.textureCompressionETC2 ? "yes" : "no");

    /* Vulkan 1.1 features */
    VkPhysicalDevice16BitStorageFeatures f16_storage = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    };
    VkPhysicalDeviceFeatures2 features2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f16_storage,
    };
    vkGetPhysicalDeviceFeatures2(gpu, &features2);
    printf("\n--- 16-bit Storage ---\n");
    printf("  storageBuffer16BitAccess: %s\n",
           f16_storage.storageBuffer16BitAccess ? "yes" : "no");
    printf("  uniformAndStorageBuffer16BitAccess: %s\n",
           f16_storage.uniformAndStorageBuffer16BitAccess ? "yes" : "no");
    printf("  storagePushConstant16: %s\n",
           f16_storage.storagePushConstant16 ? "yes" : "no");
    printf("  storageInputOutput16: %s\n",
           f16_storage.storageInputOutput16 ? "yes" : "no");

    /* Extensions */
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &ext_count, NULL);
    VkExtensionProperties *exts = malloc(ext_count * sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &ext_count, exts);
    printf("\n--- Extensions (%u total) ---\n", ext_count);

    /* Print all, but highlight the interesting ones */
    const char *interesting[] = {
        "VK_KHR_shader_float16_int8",
        "VK_KHR_16bit_storage",
        "VK_KHR_8bit_storage",
        "VK_KHR_shader_integer_dot_product",
        "VK_KHR_storage_buffer_storage_class",
        "VK_EXT_subgroup_size_control",
        "VK_KHR_shader_subgroup_extended_types",
        "VK_KHR_variable_pointers",
        "VK_KHR_shader_non_semantic_info",
        "VK_KHR_external_memory",
        "VK_EXT_external_memory_host",
        "VK_KHR_push_descriptor",
        "VK_KHR_descriptor_update_template",
        "VK_EXT_scalar_block_layout",
        "VK_KHR_shader_float_controls",
        NULL
    };

    for (uint32_t i = 0; i < ext_count; i++) {
        int highlight = 0;
        for (int j = 0; interesting[j]; j++) {
            if (strcmp(exts[i].extensionName, interesting[j]) == 0) {
                highlight = 1;
                break;
            }
        }
        if (highlight) {
            printf("  >>> %s (v%u)\n", exts[i].extensionName, exts[i].specVersion);
        } else {
            printf("      %s (v%u)\n", exts[i].extensionName, exts[i].specVersion);
        }
    }

    /* Check R32F texture format support for compute sampling */
    printf("\n--- Texture Format Support (for LUT sampling) ---\n");
    VkFormatProperties fmt;

    vkGetPhysicalDeviceFormatProperties(gpu, VK_FORMAT_R32_SFLOAT, &fmt);
    printf("  R32_SFLOAT:\n");
    printf("    linearTiling:  sampled=%s storage=%s\n",
           (fmt.linearTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? "yes" : "no",
           (fmt.linearTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? "yes" : "no");
    printf("    optimalTiling: sampled=%s storage=%s filter=%s\n",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? "yes" : "no",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? "yes" : "no",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? "LINEAR" : "NEAREST_ONLY");

    vkGetPhysicalDeviceFormatProperties(gpu, VK_FORMAT_R16_SFLOAT, &fmt);
    printf("  R16_SFLOAT:\n");
    printf("    optimalTiling: sampled=%s storage=%s filter=%s\n",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? "yes" : "no",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? "yes" : "no",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? "LINEAR" : "NEAREST_ONLY");

    vkGetPhysicalDeviceFormatProperties(gpu, VK_FORMAT_R8_UNORM, &fmt);
    printf("  R8_UNORM:\n");
    printf("    optimalTiling: sampled=%s storage=%s filter=%s\n",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? "yes" : "no",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? "yes" : "no",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? "LINEAR" : "NEAREST_ONLY");

    vkGetPhysicalDeviceFormatProperties(gpu, VK_FORMAT_R32G32B32A32_SFLOAT, &fmt);
    printf("  RGBA32_SFLOAT:\n");
    printf("    optimalTiling: sampled=%s storage=%s filter=%s\n",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? "yes" : "no",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? "yes" : "no",
           (fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? "LINEAR" : "NEAREST_ONLY");

    /* Memory properties */
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(gpu, &mem_props);
    printf("\n--- Memory Types (%u) ---\n", mem_props.memoryTypeCount);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        VkMemoryType *mt = &mem_props.memoryTypes[i];
        printf("  [%u] heap=%u flags:", i, mt->heapIndex);
        if (mt->propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) printf(" DEVICE_LOCAL");
        if (mt->propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) printf(" HOST_VISIBLE");
        if (mt->propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) printf(" HOST_COHERENT");
        if (mt->propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) printf(" HOST_CACHED");
        if (mt->propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) printf(" LAZY");
        printf("\n");
    }
    printf("\n--- Memory Heaps (%u) ---\n", mem_props.memoryHeapCount);
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        printf("  [%u] size=%llu MB flags:%s\n", i,
               (unsigned long long)(mem_props.memoryHeaps[i].size / (1024*1024)),
               (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " DEVICE_LOCAL" : "");
    }

    /* Queue families */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &qf_count, NULL);
    VkQueueFamilyProperties *qf = malloc(qf_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &qf_count, qf);
    printf("\n--- Queue Families (%u) ---\n", qf_count);
    for (uint32_t i = 0; i < qf_count; i++) {
        printf("  [%u] count=%u flags:", i, qf[i].queueCount);
        if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) printf(" GRAPHICS");
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) printf(" COMPUTE");
        if (qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) printf(" TRANSFER");
        if (qf[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) printf(" SPARSE");
        printf("  timestampValid=%u minImageGranularity=[%u,%u,%u]\n",
               qf[i].timestampValidBits,
               qf[i].minImageTransferGranularity.width,
               qf[i].minImageTransferGranularity.height,
               qf[i].minImageTransferGranularity.depth);
    }

    free(qf);
    free(exts);
    free(devs);
    vkDestroyInstance(instance, NULL);
    printf("\n=== Done ===\n");
    return 0;
}
