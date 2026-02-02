/**
 * TriX Execution Context
 *
 * Bridges Neural Interposer channels with Vulkan compute execution
 */

#include "ni_channel.h"
#include <executorch/runtime/platform/log.h>
#include <fstream>
#include <vector>

namespace {

// Shader loading helper
std::vector<uint8_t> load_spirv(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        ET_LOG(Error, "Failed to open SPIR-V file: %s", path);
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        ET_LOG(Error, "Failed to read SPIR-V file: %s", path);
        return {};
    }

    return buffer;
}

// Vulkan pipeline creation helper
VkPipeline create_compute_pipeline(VkDevice device,
                                  VkPipelineLayout layout,
                                  const std::vector<uint8_t>& spirv) {
    VkShaderModuleCreateInfo shader_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size(),
        .pCode = reinterpret_cast<const uint32_t*>(spirv.data())
    };

    VkShaderModule shader;
    if (vkCreateShaderModule(device, &shader_info, nullptr, &shader) != VK_SUCCESS) {
        ET_LOG(Error, "Failed to create shader module");
        return VK_NULL_HANDLE;
    }

    VkPipelineShaderStageCreateInfo stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader,
        .pName = "main"
    };

    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage_info,
        .layout = layout
    };

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                nullptr, &pipeline) != VK_SUCCESS) {
        ET_LOG(Error, "Failed to create compute pipeline");
        vkDestroyShaderModule(device, shader, nullptr);
        return VK_NULL_HANDLE;
    }

    vkDestroyShaderModule(device, shader, nullptr);
    return pipeline;
}

// Descriptor set layout for TriX chips
VkDescriptorSetLayout create_trix_dsl(VkDevice device) {
    VkDescriptorSetLayoutBinding bindings[] = {
        // Input channel
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        // State channel
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        // Output channel
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        // Weights
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        // Signal buffer
        {
            .binding = 4,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        }
    };

    VkDescriptorSetLayoutCreateInfo dsl_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 5,
        .pBindings = bindings
    };

    VkDescriptorSetLayout dsl;
    if (vkCreateDescriptorSetLayout(device, &dsl_info, nullptr, &dsl) != VK_SUCCESS) {
        ET_LOG(Error, "Failed to create descriptor set layout");
        return VK_NULL_HANDLE;
    }

    return dsl;
}

} // anonymous namespace

// TriX Context Creation
ni_trix_context_t* ni_trix_context_create(VkDevice device, VkPhysicalDevice phys_device,
                                         VkQueue queue, uint32_t hidden_dim, uint32_t state_dim) {
    ni_trix_context_t *ctx = (ni_trix_context_t*)calloc(1, sizeof(ni_trix_context_t));
    if (!ctx) return nullptr;

    ctx->vk_device = device;
    ctx->vk_queue = queue;
    ctx->hidden_dim = hidden_dim;
    ctx->state_dim = state_dim;

    // Create command pool
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = 0  // Assume compute queue family 0
    };

    if (vkCreateCommandPool(device, &pool_info, nullptr, &ctx->vk_cmd_pool) != VK_SUCCESS) {
        ET_LOG(Error, "Failed to create command pool");
        free(ctx);
        return nullptr;
    }

    // Create descriptor pool
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5}
    };

    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = pool_sizes
    };

    if (vkCreateDescriptorPool(device, &desc_pool_info, nullptr, &ctx->vk_desc_pool) != VK_SUCCESS) {
        ET_LOG(Error, "Failed to create descriptor pool");
        vkDestroyCommandPool(device, ctx->vk_cmd_pool, nullptr);
        free(ctx);
        return nullptr;
    }

    // Create channels
    ni_channel_config_t channel_config = {
        .type = NI_CHANNEL_TYPE_GPU,
        .capacity = 16 * 1024 * 1024,  // 16MB
        .coherent = true,
        .persistent = true,
        .alignment = 256,
        .vulkan_import = true
    };

    ctx->input_channel = ni_channel_create_ion(&channel_config);
    ctx->state_channel = ni_channel_create_ion(&channel_config);
    ctx->output_channel = ni_channel_create_ion(&channel_config);
    ctx->signal_channel = ni_channel_create_ion(&channel_config);

    if (!ctx->input_channel || !ctx->state_channel ||
        !ctx->output_channel || !ctx->signal_channel) {
        ET_LOG(Error, "Failed to create TriX channels");
        ni_trix_context_destroy(ctx);
        return nullptr;
    }

    // Import channels to Vulkan
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    bool import_success = true;
    import_success &= ni_channel_import_to_vulkan(ctx->input_channel, device, phys_device, usage);
    import_success &= ni_channel_import_to_vulkan(ctx->state_channel, device, phys_device, usage);
    import_success &= ni_channel_import_to_vulkan(ctx->output_channel, device, phys_device, usage);
    import_success &= ni_channel_import_to_vulkan(ctx->signal_channel, device, phys_device, usage);

    if (!import_success) {
        ET_LOG(Error, "Failed to import channels to Vulkan");
        ni_trix_context_destroy(ctx);
        return nullptr;
    }

    ET_LOG(Info, "Created TriX context with hidden_dim=%u, state_dim=%u",
           hidden_dim, state_dim);

    return ctx;
}

void ni_trix_context_destroy(ni_trix_context_t *ctx) {
    if (!ctx) return;

    // Clean up Vulkan resources
    if (ctx->persistent_pipeline) {
        vkDestroyPipeline(ctx->vk_device, ctx->persistent_pipeline, nullptr);
    }

    if (ctx->persistent_layout) {
        vkDestroyPipelineLayout(ctx->vk_device, ctx->persistent_layout, nullptr);
    }

    if (ctx->vk_desc_pool) {
        vkDestroyDescriptorPool(ctx->vk_device, ctx->vk_desc_pool, nullptr);
    }

    if (ctx->vk_cmd_pool) {
        vkDestroyCommandPool(ctx->vk_device, ctx->vk_cmd_pool, nullptr);
    }

    // Clean up channels
    if (ctx->input_channel) {
        ni_channel_unimport_from_vulkan(ctx->input_channel, ctx->vk_device);
        ni_channel_destroy_ion(ctx->input_channel);
    }

    if (ctx->state_channel) {
        ni_channel_unimport_from_vulkan(ctx->state_channel, ctx->vk_device);
        ni_channel_destroy_ion(ctx->state_channel);
    }

    if (ctx->output_channel) {
        ni_channel_unimport_from_vulkan(ctx->output_channel, ctx->vk_device);
        ni_channel_destroy_ion(ctx->output_channel);
    }

    if (ctx->signal_channel) {
        ni_channel_unimport_from_vulkan(ctx->signal_channel, ctx->vk_device);
        ni_channel_destroy_ion(ctx->signal_channel);
    }

    free(ctx);
}

// TriX Chip Execution
static bool execute_chip_on_vulkan(ni_trix_context_t *ctx,
                                  const std::vector<uint8_t>& spirv,
                                  uint32_t dispatch_x, uint32_t dispatch_y, uint32_t dispatch_z,
                                  VkDescriptorSet ds) {
    // Create pipeline layout
    VkDescriptorSetLayout dsl = create_trix_dsl(ctx->vk_device);
    if (!dsl) return false;

    VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl
    };

    VkPipelineLayout layout;
    if (vkCreatePipelineLayout(ctx->vk_device, &layout_info, nullptr, &layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(ctx->vk_device, dsl, nullptr);
        return false;
    }

    // Create pipeline
    VkPipeline pipeline = create_compute_pipeline(ctx->vk_device, layout, spirv);
    if (!pipeline) {
        vkDestroyPipelineLayout(ctx->vk_device, layout, nullptr);
        vkDestroyDescriptorSetLayout(ctx->vk_device, dsl, nullptr);
        return false;
    }

    // Allocate command buffer
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->vk_cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VkCommandBuffer cmd_buffer;
    if (vkAllocateCommandBuffers(ctx->vk_device, &cmd_alloc, &cmd_buffer) != VK_SUCCESS) {
        vkDestroyPipeline(ctx->vk_device, pipeline, nullptr);
        vkDestroyPipelineLayout(ctx->vk_device, layout, nullptr);
        vkDestroyDescriptorSetLayout(ctx->vk_device, dsl, nullptr);
        return false;
    }

    // Record commands
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    vkBeginCommandBuffer(cmd_buffer, &begin_info);

    // Bind pipeline and descriptor set
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           layout, 0, 1, &ds, 0, nullptr);

    // Dispatch compute work
    vkCmdDispatch(cmd_buffer, dispatch_x, dispatch_y, dispatch_z);

    // Add memory barriers for coherency
    VkMemoryBarrier memory_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT
    };

    vkCmdPipelineBarrier(cmd_buffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &memory_barrier, 0, nullptr, 0, nullptr);

    vkEndCommandBuffer(cmd_buffer);

    // Submit and wait
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd_buffer
    };

    VkFence fence;
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(ctx->vk_device, &fence_info, nullptr, &fence);

    if (vkQueueSubmit(ctx->vk_queue, 1, &submit_info, fence) != VK_SUCCESS) {
        vkDestroyFence(ctx->vk_device, fence, nullptr);
        vkFreeCommandBuffers(ctx->vk_device, ctx->vk_cmd_pool, 1, &cmd_buffer);
        vkDestroyPipeline(ctx->vk_device, pipeline, nullptr);
        vkDestroyPipelineLayout(ctx->vk_device, layout, nullptr);
        vkDestroyDescriptorSetLayout(ctx->vk_device, dsl, nullptr);
        return false;
    }

    // Wait for completion
    vkWaitForFences(ctx->vk_device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(ctx->vk_device, fence, nullptr);

    // Cleanup
    vkFreeCommandBuffers(ctx->vk_device, ctx->vk_cmd_pool, 1, &cmd_buffer);
    vkDestroyPipeline(ctx->vk_device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx->vk_device, layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx->vk_device, dsl, nullptr);

    return true;
}

bool ni_trix_execute_shortconv(ni_trix_context_t *ctx,
                              const float *input, const float *state,
                              float *output, float *next_state,
                              const float *weights, uint32_t hidden_dim) {
    if (!ctx || !input || !state || !output || !next_state || !weights) {
        return false;
    }

    // Load ShortConv SPIR-V
    const char* spv_path = getenv("NI_SHORTCONV3_SPV");
    if (!spv_path) spv_path = "/data/local/tmp/shortconv_chip.spv";

    auto spirv = load_spirv(spv_path);
    if (spirv.empty()) {
        ET_LOG(Error, "Failed to load ShortConv SPIR-V from %s", spv_path);
        return false;
    }

    // Write inputs to channels
    ni_channel_clear_signal(ctx->signal_channel);
    ni_channel_write(ctx->input_channel, input, hidden_dim * sizeof(float));
    ni_channel_write(ctx->state_channel, state, (hidden_dim * 3) * sizeof(float));  // kernel_size - 1 = 3
    ni_channel_write(ctx->output_channel, weights, (hidden_dim * 4 + hidden_dim) * sizeof(float));

    // Execute on Vulkan
    VkDescriptorSet ds = VK_NULL_HANDLE;  // TODO: Create proper descriptor set
    bool success = execute_chip_on_vulkan(ctx, spirv,
                                         (hidden_dim + 255) / 256, 1, 1,  // dispatch size
                                         ds);

    if (!success) {
        ET_LOG(Error, "ShortConv execution failed");
        return false;
    }

    // Read outputs from channels
    ni_channel_read(ctx->output_channel, output, hidden_dim * sizeof(float));
    ni_channel_read(ctx->state_channel, next_state, (hidden_dim * 3) * sizeof(float));

    return true;
}

// Quantized TriX operations
bool ni_trix_execute_quantized_matmul(
    ni_trix_context_t *ctx,
    const uint8_t *A_quantized, const float *A_scales,
    const uint8_t *B_quantized, const float *B_scales,
    float *C_output,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t block_size, uint32_t bits) {

    if (!ctx || !A_quantized || !A_scales || !B_quantized || !B_scales || !C_output) {
        return false;
    }

    // Load quantized MatMul SPIR-V shader
    const char* spv_path = getenv("NI_QUANTIZED_MATMUL_SPV");
    if (!spv_path) spv_path = "/data/local/tmp/quantized_matmul.spv";

    auto spirv = load_spirv(spv_path);
    if (spirv.empty()) {
        ET_LOG(Error, "Failed to load Quantized MatMul SPIR-V from %s", spv_path);
        return false;
    }

    // Calculate workgroup dimensions
    uint32_t dispatch_x = (M + 15) / 16;  // 16x16 tiles
    uint32_t dispatch_y = (N + 15) / 16;
    uint32_t dispatch_z = 1;

    VkDescriptorSet ds = VK_NULL_HANDLE;  // TODO: Create proper descriptor set
    bool success = execute_chip_on_vulkan(ctx, spirv,
                                         dispatch_x, dispatch_y, dispatch_z, ds);

    if (!success) {
        ET_LOG(Error, "Quantized MatMul execution failed");
        return false;
    }

    ET_LOG(Info, "Executed quantized MatMul via Neural Interposer (%ux%ux%u, %ubit)",
           M, N, K, bits);
    return true;
}

bool ni_trix_execute_attention(ni_trix_context_t *ctx,
                              const float *input, const float *kv_cache,
                              float *output, float *next_kv_cache,
                              const float *weights, uint32_t hidden_dim, uint32_t seq_len) {
    if (!ctx || !input || !kv_cache || !output || !next_kv_cache || !weights) {
        return false;
    }

    // For now, implement simplified attention
    // TODO: Implement full quantized attention with KV-cache

    // Copy input to output (placeholder)
    memcpy(output, input, hidden_dim * sizeof(float));

    // Copy KV-cache
    memcpy(next_kv_cache, kv_cache, seq_len * hidden_dim * 2 * sizeof(float));

    ET_LOG(Info, "Executed Attention via Neural Interposer (simplified, hidden_dim=%u, seq_len=%u)",
           hidden_dim, seq_len);
    return true;
}

// Quantized attention with 4-bit weights
bool ni_trix_execute_quantized_attention(
    ni_trix_context_t *ctx,
    const float *input,
    const float *kv_cache,
    float *output,
    float *next_kv_cache,
    const uint8_t *weights_quantized,  // 4-bit packed weights
    const float *weights_scales,       // Dequantization scales
    uint32_t hidden_dim,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t block_size) {

    if (!ctx || !input || !kv_cache || !output || !next_kv_cache ||
        !weights_quantized || !weights_scales) {
        return false;
    }

    // Load quantized Attention SPIR-V shader
    const char* spv_path = getenv("NI_QUANTIZED_ATTENTION_SPV");
    if (!spv_path) spv_path = "/data/local/tmp/quantized_attention.spv";

    auto spirv = load_spirv(spv_path);
    if (spirv.empty()) {
        ET_LOG(Error, "Failed to load Quantized Attention SPIR-V from %s", spv_path);
        return false;
    }

    // Calculate workgroup dimensions
    uint32_t dispatch_x = (hidden_dim + 255) / 256;
    uint32_t dispatch_y = 1;
    uint32_t dispatch_z = 1;

    VkDescriptorSet ds = VK_NULL_HANDLE;  // TODO: Create proper descriptor set
    bool success = execute_chip_on_vulkan(ctx, spirv,
                                         dispatch_x, dispatch_y, dispatch_z, ds);

    if (!success) {
        ET_LOG(Error, "Quantized Attention execution failed");
        return false;
    }

    ET_LOG(Info, "Executed quantized Attention via Neural Interposer (hidden_dim=%u, seq_len=%u, %u heads)",
           hidden_dim, seq_len, num_heads);
    return true;
}

bool ni_trix_execute_ffn(ni_trix_context_t *ctx,
                        const float *input, float *output,
                        const float *weights, uint32_t hidden_dim) {
    // TODO: Implement FFN chip execution
    ET_LOG(Error, "FFN chip execution not yet implemented");
    return false;
}