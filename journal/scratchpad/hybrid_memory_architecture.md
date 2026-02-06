# Hybrid Memory Architecture for Mobile LLM Inference

## Discovery: Cached vs Uncached Memory

Probe B v8 revealed a nuanced picture of memory performance:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CACHED vs UNCACHED PERFORMANCE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ACCESS PATTERN        │    WINNER     │  SPEEDUP    │  VARIANCE           │
│   ─────────────────────────────────────────────────────────────────────────│
│   Sequential, any size  │    CACHED     │  2-4x       │  Higher CV          │
│   Random, < 16MB        │    CACHED     │  1.5-3x     │  Higher CV          │
│   Random, > 16MB        │    UNCACHED   │  1.2-1.4x   │  Much lower CV      │
│   Repeated access       │    CACHED     │  2-3x       │  Varies             │
│                                                                              │
│   KEY INSIGHT: Hardware prefetch makes cached sequential access very fast.  │
│   But for large random access, cache thrashing makes uncached better.       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implications for LLM Inference

### Model Weights: CACHED ✓
- Access pattern: Sequential (layer by layer, weight by weight)
- Size: 1-4 GB (huge, but accessed sequentially)
- Hardware prefetch works excellently
- Cached is 2-4x faster for sequential access
- **Keep in normal cached memory**

### KV Cache: UNCACHED ✓
- Access pattern: Random (attention to arbitrary past tokens)
- Size: 100MB - 1GB (depends on context length)
- No reuse pattern (each query hits different keys)
- Cached causes thrashing, evicts useful data
- **Allocate via dma_heap uncached**

### Activations: CACHED ✓
- Access pattern: Repeated within layer
- Size: 1-10 MB per layer
- High reuse (same activations used multiple times)
- Fits in L2 cache
- **Keep in normal cached memory**

### Embedding Table: UNCACHED ✓
- Access pattern: Sparse random lookup
- Size: 50-200 MB (vocab_size × hidden_dim)
- No locality (tokens are random)
- **Allocate via dma_heap uncached**

## Proposed Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYBRID MEMORY ALLOCATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    CACHED REGION (malloc/mmap)                       │   │
│   │                                                                      │   │
│   │   ┌──────────────────┐  ┌──────────────────┐                        │   │
│   │   │  Model Weights   │  │   Activations    │                        │   │
│   │   │  Sequential R/O  │  │   Repeated R/W   │                        │   │
│   │   │  ~2GB            │  │   ~10MB          │                        │   │
│   │   └──────────────────┘  └──────────────────┘                        │   │
│   │                                                                      │   │
│   │   Benefits from: HW prefetch, L1/L2 cache, low latency              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    UNCACHED REGION (dma_heap)                        │   │
│   │                                                                      │   │
│   │   ┌──────────────────┐  ┌──────────────────┐                        │   │
│   │   │    KV Cache      │  │   Embeddings     │                        │   │
│   │   │   Random R/W     │  │   Random R/O     │                        │   │
│   │   │   ~500MB         │  │   ~100MB         │                        │   │
│   │   └──────────────────┘  └──────────────────┘                        │   │
│   │                                                                      │   │
│   │   Benefits from: No cache thrashing, consistent latency, low CV     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation in llama.cpp

### Option 1: Custom Allocator

```cpp
// llama_allocator.h

class HybridAllocator {
public:
    enum MemoryType {
        CACHED,     // Normal malloc, for weights/activations
        UNCACHED    // dma_heap, for KV cache/embeddings
    };
    
    void* allocate(size_t size, MemoryType type);
    void deallocate(void* ptr, size_t size, MemoryType type);
    
private:
    int dma_heap_fd = -1;
    std::map<void*, int> uncached_fds;  // Track buffer FDs for cleanup
};

// Usage in ggml:
void* ggml_backend_buffer_alloc(size_t size, bool random_access) {
    HybridAllocator alloc;
    return alloc.allocate(size, 
        random_access ? HybridAllocator::UNCACHED : HybridAllocator::CACHED);
}
```

### Option 2: Buffer Type Hints

Add memory type hints to ggml tensor creation:

```cpp
struct ggml_tensor_params {
    // ... existing params ...
    enum ggml_mem_type {
        GGML_MEM_AUTO,      // Let system decide
        GGML_MEM_CACHED,    // Force cached
        GGML_MEM_UNCACHED   // Force uncached (dma_heap)
    } mem_type;
};

// In model loading:
// Weights → GGML_MEM_CACHED (sequential access)
// KV cache → GGML_MEM_UNCACHED (random access)
```

## Expected Impact

Based on probe measurements:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPECTED PERFORMANCE IMPACT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Component       │ Current │ Hybrid  │ Improvement │ Variance              │
│   ────────────────────────────────────────────────────────────────────────  │
│   Weight access   │ Cached  │ Cached  │ No change   │ No change             │
│   KV cache access │ Cached  │ Uncached│ ~1.3x faster│ CV: 10% → 2%          │
│   Embeddings      │ Cached  │ Uncached│ ~1.3x faster│ CV: 13% → 5%          │
│   Activations     │ Cached  │ Cached  │ No change   │ No change             │
│                                                                              │
│   OVERALL: KV cache is ~20% of attention time.                              │
│   With 1.3x speedup on KV access: ~5-6% inference speedup                   │
│   Plus: Much more consistent latency (lower jitter)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## What About GPU?

The GPU fabric idea is now **obsolete** for this use case because:

1. **Sequential access** (weights): HW prefetch already optimal, GPU can't help
2. **Random access** (KV cache): GPU doesn't know what's next, can't prefetch

However, GPU could still help with:
- **Background KV cache eviction** to swap (async, when CPU idle)
- **Parallel attention scoring** (actual compute, not memory)
- **Model weight compression/decompression** (if using on-the-fly decompression)

## Next Steps

1. **Probe C**: Test uncached KV cache in actual llama.cpp
   - Modify ggml to use dma_heap for KV cache allocation
   - Measure tok/s and latency variance
   
2. **Quantify KV cache access pattern**
   - Profile actual KV cache access in llama.cpp
   - Confirm random access hypothesis
   
3. **Implement HybridAllocator**
   - Create Android-specific allocator
   - Integrate with llama.cpp build

## Risk Assessment

- **Platform-specific**: dma_heap API is Android/MediaTek specific
- **Memory management complexity**: Need to track uncached allocations separately
- **Potential fragmentation**: Uncached region managed differently from cached

## Conclusion

The "GPU memory fabric" concept evolved into something better: **hybrid cached/uncached memory allocation** based on access patterns.

- Cache helps sequential/repeated access (weights, activations)
- Uncached helps large random access (KV cache, embeddings)
- GPU prefetch concept is abandoned (HW prefetch already optimal)

This is a simpler, more robust solution that doesn't require GPU involvement.
