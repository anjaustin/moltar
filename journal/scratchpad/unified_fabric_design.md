# Unified Memory Fabric: Eliminating Cache Coherency Overhead

## The Problem

Our Probe B tests showed:
- Cross-core prefetch: **0.86x** (slower due to cache coherency)
- Concurrent access: **0.75x** (worse due to contention + coherency)

The issue is ARM's cache coherency protocol (MESI/MOESI on CCI-550):

```
┌─────────────────────────────────────────────────────────────────┐
│            CURRENT: Cache Coherent CPU Cluster                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Core 0 reads addr X:                                          │
│   1. Check L1 (miss)                                            │
│   2. Check L2 (miss)                                            │
│   3. Snoop other cores via CCI (no data)                        │
│   4. Read from DRAM → cache in L1/L2 (state: Exclusive)         │
│                                                                  │
│   Core 7 wants addr X:                                          │
│   1. Check L1 (miss)                                            │
│   2. Check L2 (miss)                                            │
│   3. Snoop other cores via CCI → Core 0 has it!                 │
│   4. Cache-to-cache transfer (state: Shared on both)            │
│   5. Invalidation traffic if either writes                      │
│                                                                  │
│   OVERHEAD: Snoop latency + transfer latency > DRAM latency!    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## The Opportunity

GPU is **NOT cache-coherent** with CPU. It has a direct path to DRAM:

```
┌─────────────────────────────────────────────────────────────────┐
│                GPU PATH (Non-Coherent)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   GPU reads addr Y:                                             │
│   1. Check GPU cache (if any)                                   │
│   2. Read directly from DRAM via EMI                            │
│   3. NO SNOOP to CPU cores                                      │
│   4. NO coherency protocol overhead                             │
│                                                                  │
│   CPU reads addr Y (if GPU wrote it):                           │
│   - CPU must explicitly invalidate its cache first              │
│   - Or use uncached/write-through mapping                       │
│   - Then read from DRAM                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Hypothesis: Bypass Coherency via Shared Non-Coherent Region

What if we:
1. Allocate a memory region that BOTH CPU and GPU access as **non-coherent**
2. GPU writes model weights/data to this region
3. CPU reads from this region with cache bypassed (or explicit invalidation)
4. Eliminate coherency overhead entirely

```
┌─────────────────────────────────────────────────────────────────┐
│              PROPOSED: Non-Coherent Shared Region                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌─────────────────────┐                      │
│                    │   DRAM Region       │                      │
│                    │   (Non-Coherent)    │                      │
│                    │                     │                      │
│                    │   Model Weights     │                      │
│                    │   KV Cache          │                      │
│                    │   Activations       │                      │
│                    └─────────┬───────────┘                      │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              │                               │                   │
│              ▼                               ▼                   │
│   ┌─────────────────────┐       ┌─────────────────────┐        │
│   │   CPU Access        │       │   GPU Access        │        │
│   │   (uncached or      │       │   (normal)          │        │
│   │    explicit inv)    │       │                     │        │
│   │                     │       │                     │        │
│   │   - No snoop        │       │   - Direct DRAM     │        │
│   │   - Direct DRAM     │       │   - No coherency    │        │
│   │   - Consistent!     │       │                     │        │
│   └─────────────────────┘       └─────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Options

### Option A: ION/DMA-BUF with Cache Bypass

Linux provides ION (or DMA-BUF) heaps that can allocate non-coherent memory:

```c
// Allocate uncached ION buffer
int ion_fd = open("/dev/ion", O_RDONLY);
struct ion_allocation_data alloc = {
    .len = size,
    .heap_id_mask = ION_HEAP_TYPE_SYSTEM,  // or CARVEOUT
    .flags = ION_FLAG_CACHED_NEEDS_SYNC,   // or uncached
};
ioctl(ion_fd, ION_IOC_ALLOC, &alloc);

// Map without caching
void *ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, 
                 MAP_SHARED, alloc.fd, 0);

// For MediaTek: may need MTK-specific heap
// ION_HEAP_TYPE_MULTIMEDIA or similar
```

### Option B: /dev/mem with Uncached Mapping

Direct physical memory access with cache attributes:

```c
int fd = open("/dev/mem", O_RDWR | O_SYNC);
// O_SYNC often implies uncached on ARM

void *ptr = mmap(NULL, size, PROT_READ|PROT_WRITE,
                 MAP_SHARED, fd, phys_addr);
```

### Option C: GPU Memory Export (dmabuf)

Have GPU allocate memory and export to CPU:

```c
// GPU side (via Vulkan/OpenCL)
VkBuffer buffer = createBuffer(...);
VkMemoryGetFdInfoKHR info = {
    .memory = bufferMemory,
    .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT,
};
vkGetMemoryFdKHR(device, &info, &dma_buf_fd);

// CPU side
void *ptr = mmap(NULL, size, PROT_READ, MAP_SHARED, dma_buf_fd, 0);
// Access is non-coherent, goes directly to DRAM
```

### Option D: CMA (Contiguous Memory Allocator) with Cache Ops

```c
// Allocate from CMA region
void *buf = dma_alloc_coherent(dev, size, &phys, GFP_KERNEL);

// Or for non-coherent:
void *buf = dma_alloc_attrs(dev, size, &phys, GFP_KERNEL,
                            DMA_ATTR_NON_CONSISTENT);

// Manual cache management:
dma_sync_single_for_cpu(dev, phys, size, DMA_FROM_DEVICE);
// ... read data ...
dma_sync_single_for_device(dev, phys, size, DMA_TO_DEVICE);
```

## The Real Question: Is Uncached Access Faster?

Uncached access eliminates coherency overhead but also eliminates caching benefits.

Trade-off analysis:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACCESS PATTERNS                               │
├───────────────────┬─────────────────┬───────────────────────────┤
│ Pattern           │ Cached Better?  │ Uncached Better?          │
├───────────────────┼─────────────────┼───────────────────────────┤
│ Repeated access   │ YES (cache hit) │ NO (repeated DRAM)        │
│ Sequential stream │ MAYBE (prefetch)│ MAYBE (no pollution)      │
│ Random access     │ NO (miss+evict) │ YES (no cache thrash)     │
│ Cross-core share  │ NO (coherency)  │ YES (no snoop)            │
│ Large working set │ NO (cache miss) │ YES (predictable)         │
└───────────────────┴─────────────────┴───────────────────────────┘
```

For LLM inference:
- Model weights: Large, mostly sequential, read-only → **Uncached may win**
- KV cache: Random access pattern, cross-layer → **Uncached may win**
- Activations: Reused within layer → **Cached probably better**

## Proposed Experiment: Probe B v7

Test uncached access to eliminate coherency:

```c
// probe_b_v7.c - Uncached shared region test

#include <sys/mman.h>
#include <fcntl.h>

int main() {
    // Allocate via ION or /dev/mem with uncached attributes
    int fd = open("/dev/ion", O_RDWR);
    // ... allocate uncached buffer ...
    
    // Test 1: CPU uncached read latency
    // (no cache, no coherency, direct DRAM)
    
    // Test 2: GPU writes, CPU reads (uncached)
    // (should be fast - no coherency protocol)
    
    // Test 3: Compare to cached cross-core
    // (should show uncached wins for sharing)
}
```

## Potential Architecture: Hybrid Cached/Uncached

```
┌─────────────────────────────────────────────────────────────────┐
│              HYBRID MEMORY ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Model Weights                         │   │
│   │                    (UNCACHED)                            │   │
│   │   - 2GB+ of weights                                      │   │
│   │   - Sequential read access                               │   │
│   │   - No benefit from caching (too large)                  │   │
│   │   - GPU can prefetch → DRAM rows open                    │   │
│   │   - CPU reads uncached → direct DRAM, no coherency       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    KV Cache                              │   │
│   │                    (UNCACHED)                            │   │
│   │   - Random access across layers                          │   │
│   │   - Cross-layer sharing                                  │   │
│   │   - No cache benefit (random, large)                     │   │
│   │   - GPU manages eviction to swap                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Activations                           │   │
│   │                    (CACHED - normal)                     │   │
│   │   - Small, reused within layer                          │   │
│   │   - Benefits from L1/L2 cache                           │   │
│   │   - Keep coherent for CPU compute                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Might Work for GPU "Fabric"

If we use uncached regions:

1. **GPU prefetch now makes sense**: GPU reads ahead → DRAM rows open → CPU reads uncached hit open rows → **row buffer benefit transfers!**

2. **No coherency overhead**: CPU reads uncached, no snoop needed, direct DRAM access

3. **Predictable latency**: No cache miss variance, consistent DRAM latency

The earlier probes failed because:
- Probe B v5/v6 used cached memory
- GPU read → data in GPU "cache" or system cache
- CPU read → coherency protocol to get data
- Overhead exceeded benefit

With uncached:
- GPU read → data stays in DRAM (DRAM row opened)
- CPU read uncached → direct DRAM access (row already open)
- **Row buffer benefit transfers without coherency overhead!**

## Next Steps

1. **Check ION availability**: `ls /dev/ion` or `/dev/dma_heap/`
2. **Allocate uncached buffer**: Test with MTK ION heap
3. **Probe B v7**: Test GPU→CPU sharing with uncached region
4. **Measure**: Does row buffer benefit now transfer?

## Risk Assessment

- **Uncached access is slower for repeated reads**: Need to ensure access pattern is streaming/random, not repeated
- **Requires modified llama.cpp**: Memory allocation changes, not trivial
- **Platform-specific**: ION/dmabuf APIs vary by Android version and vendor

## Success Criteria

Probe B v7 should show:
- GPU prefetch + CPU uncached read: **>1.5x faster** than baseline
- Variance reduction: **CV < 5%**

If this works, we've created a "unified fabric" where GPU can effectively warm DRAM rows for CPU without cache coherency overhead.
