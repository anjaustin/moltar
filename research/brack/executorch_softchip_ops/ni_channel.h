/**
 * Neural Interposer Channel Abstraction Layer for Android
 *
 * ION-based coherent memory implementation for ExecuTorch integration
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <linux/ion.h>
#include <linux/dma-buf.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

// Channel types for Neural Interposer
typedef enum {
    NI_CHANNEL_TYPE_CPU,      // Control logic, branching
    NI_CHANNEL_TYPE_GPU,      // Parallel compute (Vulkan)
    NI_CHANNEL_TYPE_CACHE,    // Coherent state storage
    NI_CHANNEL_TYPE_MEMORY    // Backing store
} ni_channel_type_t;

// ION-based channel descriptor
typedef struct {
    ni_channel_type_t type;
    size_t capacity;           // Buffer size in bytes
    void *coherent_mem;        // ION-mapped coherent memory
    uint32_t read_ptr;         // Circular buffer read position
    uint32_t write_ptr;        // Circular buffer write position
    atomic_uint signal;        // Signaling flag (futex-based)
    atomic_ullong version;     // Coherency version
    int ion_fd;                // ION device file descriptor
    int dma_buf_fd;            // DMA-BUF file descriptor for Vulkan import
    bool owns_memory;          // Whether to free on destroy
    VkDeviceMemory vk_memory;  // Vulkan memory handle (when imported)
    VkBuffer vk_buffer;        // Vulkan buffer handle (when created)
} ni_channel_t;

// Channel configuration
typedef struct {
    ni_channel_type_t type;
    size_t capacity;
    bool coherent;             // Require cache coherency (ION)
    bool persistent;           // Keep mapped across operations
    uint32_t alignment;        // Memory alignment requirement
    bool vulkan_import;        // Whether to prepare for Vulkan import
} ni_channel_config_t;

// ION Memory Management
ni_channel_t* ni_channel_create_ion(const ni_channel_config_t *config);
void ni_channel_destroy_ion(ni_channel_t *ch);

// Channel I/O (zero-copy)
void ni_channel_write(ni_channel_t *ch, const void *data, size_t len);
void ni_channel_read(ni_channel_t *ch, void *data, size_t len);
void* ni_channel_get_ptr(ni_channel_t *ch, size_t offset);
size_t ni_channel_available(ni_channel_t *ch);

// Channel signaling
void ni_channel_signal(ni_channel_t *ch);
void ni_channel_wait_signal(ni_channel_t *ch);
bool ni_channel_poll_signal(ni_channel_t *ch);
void ni_channel_clear_signal(ni_channel_t *ch);

// Coherency
void ni_channel_flush(ni_channel_t *ch);
void ni_channel_invalidate(ni_channel_t *ch);
uint64_t ni_channel_get_version(ni_channel_t *ch);

// Vulkan Integration
bool ni_channel_import_to_vulkan(ni_channel_t *ch, VkDevice device,
                                VkPhysicalDevice physical_device,
                                VkBufferUsageFlags usage_flags);
void ni_channel_unimport_from_vulkan(ni_channel_t *ch, VkDevice device);

// TriX Chip Execution Context
typedef struct {
    ni_channel_t *input_channel;
    ni_channel_t *state_channel;
    ni_channel_t *output_channel;
    ni_channel_t *signal_channel;

    // Vulkan resources (shared across chips)
    VkDevice vk_device;
    VkQueue vk_queue;
    VkCommandPool vk_cmd_pool;
    VkDescriptorPool vk_desc_pool;

    // Persistent kernel state
    VkPipeline persistent_pipeline;
    VkPipelineLayout persistent_layout;
    VkDescriptorSet persistent_ds;
    bool persistent_running;

    // Chip metadata
    uint32_t hidden_dim;
    uint32_t state_dim;
} ni_trix_context_t;

// TriX Context Management
ni_trix_context_t* ni_trix_context_create(VkDevice device, VkPhysicalDevice phys_device,
                                         VkQueue queue, uint32_t hidden_dim, uint32_t state_dim);
void ni_trix_context_destroy(ni_trix_context_t *ctx);

// TriX Chip Execution (bridged to ExecuTorch)
bool ni_trix_execute_shortconv(ni_trix_context_t *ctx,
                              const float *input, const float *state,
                              float *output, float *next_state,
                              const float *weights, uint32_t hidden_dim);

bool ni_trix_execute_attention(ni_trix_context_t *ctx,
                              const float *input, const float *kv_cache,
                              float *output, float *next_kv_cache,
                              const float *weights, uint32_t hidden_dim, uint32_t seq_len);

bool ni_trix_execute_ffn(ni_trix_context_t *ctx,
                        const float *input, float *output,
                        const float *weights, uint32_t hidden_dim);

#ifdef __cplusplus
}
#endif

#endif // NI_CHANNEL_H