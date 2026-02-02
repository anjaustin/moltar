/**
 * Neural Interposer Channel Abstraction Layer Implementation
 *
 * ION-based coherent memory for Android/Motorola devices
 */

#include "ni_channel.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <executorch/runtime/platform/log.h>

namespace {

// ION ioctl helpers
static int ion_open() {
    int fd = open("/dev/ion", O_RDONLY);
    if (fd < 0) {
        ET_LOG(Error, "Failed to open ION device: %s", strerror(errno));
    }
    return fd;
}

static int ion_alloc(int ion_fd, size_t size, unsigned int heap_id, unsigned int flags) {
    struct ion_allocation_data alloc_data = {
        .len = size,
        .heap_id_mask = 1 << heap_id,
        .flags = flags,
        .fd = 0,
    };

    if (ioctl(ion_fd, ION_IOC_ALLOC, &alloc_data) < 0) {
        ET_LOG(Error, "ION allocation failed: %s", strerror(errno));
        return -1;
    }

    return alloc_data.fd;
}

static int dma_buf_sync(int dma_fd, bool read, bool write) {
    struct dma_buf_sync sync = {
        .flags = 0,
    };

    if (read) sync.flags |= DMA_BUF_SYNC_READ;
    if (write) sync.flags |= DMA_BUF_SYNC_WRITE;

    if (ioctl(dma_fd, DMA_BUF_IOCTL_SYNC, &sync) < 0) {
        ET_LOG(Error, "DMA-BUF sync failed: %s", strerror(errno));
        return -1;
    }

    return 0;
}

// Futex-based signaling
static int futex_wait(atomic_uint *addr, uint32_t val) {
    return syscall(SYS_futex, addr, FUTEX_WAIT, val, NULL, NULL, 0);
}

static int futex_wake(atomic_uint *addr, int count) {
    return syscall(SYS_futex, addr, FUTEX_WAKE, count, NULL, NULL, 0);
}

// Vulkan memory type finding
static uint32_t find_vulkan_memory_type(VkPhysicalDevice phys_device,
                                       uint32_t type_bits,
                                       VkMemoryPropertyFlags req_flags) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & req_flags) == req_flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

} // anonymous namespace

// ION Channel Creation
ni_channel_t* ni_channel_create_ion(const ni_channel_config_t *config) {
    if (!config) return nullptr;

    ni_channel_t *ch = (ni_channel_t*)calloc(1, sizeof(ni_channel_t));
    if (!ch) return nullptr;

    ch->type = config->type;
    ch->capacity = config->capacity;
    ch->owns_memory = true;

    // Open ION device
    ch->ion_fd = ion_open();
    if (ch->ion_fd < 0) {
        free(ch);
        return nullptr;
    }

    // Allocate ION memory (system heap for coherency)
    unsigned int heap_id = ION_HEAP_SYSTEM;
    unsigned int flags = ION_FLAG_CACHED;  // Enable caching for coherency

    ch->dma_buf_fd = ion_alloc(ch->ion_fd, config->capacity, heap_id, flags);
    if (ch->dma_buf_fd < 0) {
        close(ch->ion_fd);
        free(ch);
        return nullptr;
    }

    // Map the DMA-BUF
    ch->coherent_mem = mmap(nullptr, config->capacity,
                           PROT_READ | PROT_WRITE, MAP_SHARED,
                           ch->dma_buf_fd, 0);
    if (ch->coherent_mem == MAP_FAILED) {
        ET_LOG(Error, "Failed to mmap ION buffer: %s", strerror(errno));
        close(ch->dma_buf_fd);
        close(ch->ion_fd);
        free(ch);
        return nullptr;
    }

    // Initialize atomic fields
    atomic_init(&ch->signal, 0);
    atomic_init(&ch->version, 0);

    // Initialize circular buffer pointers
    ch->read_ptr = 0;
    ch->write_ptr = 0;

    ET_LOG(Info, "Created ION channel: type=%d, capacity=%zu bytes",
           config->type, config->capacity);

    return ch;
}

void ni_channel_destroy_ion(ni_channel_t *ch) {
    if (!ch) return;

    if (ch->vk_memory && ch->vk_buffer) {
        // Note: Vulkan resources should be destroyed by caller
        ET_LOG(Warning, "Vulkan resources still allocated in channel");
    }

    if (ch->coherent_mem && ch->owns_memory) {
        munmap(ch->coherent_mem, ch->capacity);
    }

    if (ch->dma_buf_fd >= 0) {
        close(ch->dma_buf_fd);
    }

    if (ch->ion_fd >= 0) {
        close(ch->ion_fd);
    }

    free(ch);
}

// Channel I/O Operations
void ni_channel_write(ni_channel_t *ch, const void *data, size_t len) {
    if (!ch || !data || len > ch->capacity) return;

    // Simple circular buffer write (no wrapping for now)
    if (ch->write_ptr + len <= ch->capacity) {
        memcpy((char*)ch->coherent_mem + ch->write_ptr, data, len);
        ch->write_ptr += len;

        // Ensure coherency
        if (ch->type == NI_CHANNEL_TYPE_GPU) {
            dma_buf_sync(ch->dma_buf_fd, false, true);
        }
    }
}

void ni_channel_read(ni_channel_t *ch, void *data, size_t len) {
    if (!ch || !data || len > ch->capacity) return;

    // Simple circular buffer read (no wrapping for now)
    if (ch->read_ptr + len <= ch->capacity) {
        memcpy(data, (char*)ch->coherent_mem + ch->read_ptr, len);
        ch->read_ptr += len;

        // Ensure coherency
        if (ch->type == NI_CHANNEL_TYPE_GPU) {
            dma_buf_sync(ch->dma_buf_fd, true, false);
        }
    }
}

void* ni_channel_get_ptr(ni_channel_t *ch, size_t offset) {
    if (!ch || offset >= ch->capacity) return nullptr;
    return (char*)ch->coherent_mem + offset;
}

size_t ni_channel_available(ni_channel_t *ch) {
    if (!ch) return 0;
    return ch->write_ptr - ch->read_ptr;
}

// Signaling Operations
void ni_channel_signal(ni_channel_t *ch) {
    if (!ch) return;
    atomic_store(&ch->signal, 1);
    atomic_fetch_add(&ch->version, 1);
    futex_wake(&ch->signal, 1);
}

void ni_channel_wait_signal(ni_channel_t *ch) {
    if (!ch) return;

    while (atomic_load(&ch->signal) == 0) {
        futex_wait(&ch->signal, 0);
    }
}

bool ni_channel_poll_signal(ni_channel_t *ch) {
    if (!ch) return false;
    return atomic_load(&ch->signal) != 0;
}

void ni_channel_clear_signal(ni_channel_t *ch) {
    if (!ch) return;
    atomic_store(&ch->signal, 0);
}

// Coherency Operations
void ni_channel_flush(ni_channel_t *ch) {
    if (!ch) return;
    dma_buf_sync(ch->dma_buf_fd, false, true);
}

void ni_channel_invalidate(ni_channel_t *ch) {
    if (!ch) return;
    dma_buf_sync(ch->dma_buf_fd, true, false);
}

uint64_t ni_channel_get_version(ni_channel_t *ch) {
    if (!ch) return 0;
    return atomic_load(&ch->version);
}

// Vulkan Integration
bool ni_channel_import_to_vulkan(ni_channel_t *ch, VkDevice device,
                                VkPhysicalDevice physical_device,
                                VkBufferUsageFlags usage_flags) {
    if (!ch || !device) return false;

    // Import DMA-BUF to Vulkan memory
    VkImportMemoryFdInfoKHR import_info = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = dup(ch->dma_buf_fd)  // Duplicate FD for Vulkan
    };

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = ch->capacity,
        .memoryTypeIndex = find_vulkan_memory_type(physical_device,
                                                   UINT32_MAX,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    };

    if (vkAllocateMemory(device, &alloc_info, nullptr, &ch->vk_memory) != VK_SUCCESS) {
        ET_LOG(Error, "Failed to import DMA-BUF to Vulkan memory");
        return false;
    }

    // Create Vulkan buffer
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = ch->capacity,
        .usage = usage_flags,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    if (vkCreateBuffer(device, &buffer_info, nullptr, &ch->vk_buffer) != VK_SUCCESS) {
        vkFreeMemory(device, ch->vk_memory, nullptr);
        ch->vk_memory = VK_NULL_HANDLE;
        ET_LOG(Error, "Failed to create Vulkan buffer");
        return false;
    }

    // Bind memory to buffer
    if (vkBindBufferMemory(device, ch->vk_buffer, ch->vk_memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device, ch->vk_buffer, nullptr);
        vkFreeMemory(device, ch->vk_memory, nullptr);
        ch->vk_memory = VK_NULL_HANDLE;
        ch->vk_buffer = VK_NULL_HANDLE;
        ET_LOG(Error, "Failed to bind Vulkan memory to buffer");
        return false;
    }

    ET_LOG(Info, "Successfully imported ION buffer to Vulkan");
    return true;
}

void ni_channel_unimport_from_vulkan(ni_channel_t *ch, VkDevice device) {
    if (!ch || !device) return;

    if (ch->vk_buffer) {
        vkDestroyBuffer(device, ch->vk_buffer, nullptr);
        ch->vk_buffer = VK_NULL_HANDLE;
    }

    if (ch->vk_memory) {
        vkFreeMemory(device, ch->vk_memory, nullptr);
        ch->vk_memory = VK_NULL_HANDLE;
    }
}