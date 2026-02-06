# REFLECT v2: Memory Fabric - The Pattern Emerges

## What We've Learned

### The Core Discovery

**Uncached memory eliminates coherency overhead, enabling cross-core cooperation that FAILED with cached memory.**

| Test | Cached | Uncached |
|------|--------|----------|
| Cross-core prefetch (random) | 0.86x | 1.09x |
| Concurrent access (random) | 0.75x | 1.09x |
| Cross-core prefetch (sequential) | ~0.9x | 0.91-0.99x |

The pattern is clear:
- **Random access benefits from uncached prefetch**
- **Sequential access is already optimized by HW, don't mess with it**

### The Variance Story

Uncached access has dramatically lower variance:
- Cached random: CV 10-65%
- Uncached random: CV 1-7%
- Uncached with prefetch: CV 1.6-2.7%

**Predictability is a feature, not just speed.**

### The Bandwidth Story

- CPU alone: ~10 GB/s
- GPU alone: ~8 GB/s  
- CPU + GPU concurrent (cached): bandwidth competition
- CPU + GPU concurrent (uncached): ~18 GB/s combined (additive!)

**Uncached access allows parallel bandwidth usage without contention.**

---

## Why Does This Work?

### The Cache Coherency Tax

With cached memory:
1. Core A reads address X → X is cached in A's L1/L2 (Exclusive state)
2. Core B reads address X → snoop protocol, cache-to-cache transfer
3. Overhead: 50-100ns per transfer

With uncached memory:
1. Core A reads address X → X fetched from DRAM (not cached)
2. Core B reads address X → X fetched from DRAM (not cached)
3. Both reads go directly to DRAM, no coordination needed

For large working sets where cache misses dominate anyway, the coherency protocol is pure overhead.

### The DRAM Row Buffer Effect

DRAM is organized in rows (~8KB). Opening a row takes ~15-20ns. Reading from an open row takes ~5ns.

When one core reads address X:
- Row containing X is opened
- Row buffer holds the entire row
- Other addresses in that row are faster to read

With uncached access, this row buffer effect transfers between cores:
- Core A reads X, opens row
- Core B reads X+64 (same row), benefits from open row
- No cache coherency in the way

### The SMI/EMI Architecture Helps

MediaTek's SMI (Smart Multimedia Interface) connects everything to the EMI (memory controller). Different agents (CPU, GPU, display, etc.) have separate paths.

When both CPU and GPU use uncached access:
- Both paths are active simultaneously
- EMI handles arbitration efficiently
- Combined bandwidth exceeds single-agent bandwidth

---

## The Pattern: Access Pattern Determines Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ACCESS PATTERN DECISION TREE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Is the working set < 16MB?                                                │
│   ├─ YES → Use CACHED memory (cache hits dominate)                          │
│   └─ NO → Continue...                                                       │
│                                                                              │
│   Is access sequential?                                                      │
│   ├─ YES → Use CACHED memory (HW prefetch is excellent)                     │
│   │        DO NOT add prefetch threads (causes contention)                  │
│   └─ NO → Continue...                                                       │
│                                                                              │
│   Is data accessed repeatedly?                                              │
│   ├─ YES → Use CACHED memory (reuse benefits)                               │
│   └─ NO → Use UNCACHED memory                                               │
│           CAN add prefetch threads (benefits from row warming)              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The LLM Inference Mapping

### Model Weights
- Size: 1-4 GB
- Access: Sequential (layer by layer)
- Pattern: Read once per token
- **Strategy: CACHED (HW prefetch handles it)**

### KV Cache
- Size: 100MB - 1GB+
- Access: Random (attention to arbitrary past tokens)
- Pattern: Read once per attention head
- **Strategy: UNCACHED + prefetch thread**

### Embeddings
- Size: 50-200 MB
- Access: Random (sparse token lookups)
- Pattern: Read once per token
- **Strategy: UNCACHED (no prefetch - can't predict tokens)**

### Activations
- Size: 1-10 MB
- Access: Repeated within layer
- Pattern: Multiple reads
- **Strategy: CACHED (reuse benefits)**

---

## The Funky Part: What's the FABRIC?

We keep talking about "fabric" but haven't defined it. Let me get wild:

### Fabric Concept 1: Memory Type Router

A runtime system that routes allocations to cached/uncached based on access pattern:

```c
void* fabric_alloc(size_t size, access_pattern_t pattern) {
    if (pattern == SEQUENTIAL || pattern == REPEATED || size < 16MB) {
        return malloc(size);  // Cached
    } else {
        return dma_heap_alloc(size);  // Uncached
    }
}
```

### Fabric Concept 2: Prefetch Coordinator

A background service that manages prefetch threads:

```c
// Main inference thread (BIG core)
void inference_thread() {
    for (int layer = 0; layer < num_layers; layer++) {
        // Tell fabric to prefetch KV cache for next attention
        fabric_prefetch_kv(layer + 1);
        
        // Process current layer
        process_layer(layer);
        
        // Fabric handles KV prefetch in background
    }
}

// Fabric prefetch thread (LITTLE core)
void fabric_prefetch_thread() {
    while (running) {
        int layer = get_next_prefetch_request();
        // Read KV cache entries for this layer (uncached)
        prefetch_kv_entries(layer);
    }
}
```

### Fabric Concept 3: Bandwidth Allocator

A system that balances memory bandwidth between components:

```
Total bandwidth: ~18 GB/s (with uncached parallel access)

Allocation:
- Weights (sequential, cached): ~8 GB/s (BIG cores via HW prefetch)
- KV cache (random, uncached): ~6 GB/s (prefetch threads + compute)
- Activations (cached): ~2 GB/s (fits in cache mostly)
- Overhead/slack: ~2 GB/s
```

### Fabric Concept 4: The LITTLE Core Fabric

Use the 6 LITTLE cores as a dedicated memory management subsystem:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LITTLE CORE FABRIC                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   BIG Cores (2x A78)          LITTLE Cores (6x A55)             │
│   ┌─────────────────┐         ┌─────────────────┐               │
│   │                 │         │  KV Prefetch    │ (2 cores)     │
│   │   Inference     │←───────│  Read ahead     │               │
│   │   Compute       │         ├─────────────────┤               │
│   │                 │         │  Swap Manager   │ (1 core)      │
│   │   (weights +    │←───────│  UFS I/O        │               │
│   │    activations) │         ├─────────────────┤               │
│   │                 │         │  Memory Monitor │ (1 core)      │
│   │                 │←───────│  Track patterns │               │
│   └─────────────────┘         ├─────────────────┤               │
│                               │  Spare          │ (2 cores)     │
│                               │  Overflow tasks │               │
│                               └─────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

This is a full "memory fabric" - the LITTLE cores become a dedicated memory management subsystem, freeing the BIG cores for pure compute.

---

## The Synthesis Question

What do we build first?

Option A: Just use uncached for KV cache, measure tok/s improvement
- Simplest test
- Validates the core hypothesis
- Requires modifying llama.cpp allocator

Option B: Build the LITTLE core prefetch system
- More complex
- Potentially bigger gains for random access
- Requires thread coordination

Option C: Build the full fabric with role assignment
- Most complex
- Most potential
- Significant engineering effort

**Recommendation: Start with Option A.** It's the minimal test of the core hypothesis. If uncached KV cache improves tok/s, then pursue B and C.

---

## Remaining Questions

1. What's the actual KV cache access pattern in llama.cpp? Is it really random?
2. How much of inference time is spent in KV cache operations?
3. Is there a way to tell the model uses uncached memory without code changes? (mmap flags?)
4. What's the power consumption of uncached access vs cached?

---

## The Core Insight

**The "fabric" isn't hardware - it's a SOFTWARE ABSTRACTION that routes memory access to the right physical path based on access pattern.**

- Sequential → Cached (HW prefetch)
- Large random → Uncached (no coherency overhead)
- KV cache → Uncached + LITTLE core prefetch (row warming)

The hardware already has multiple paths (cached/uncached, CPU/GPU). We just need to USE THEM CORRECTLY.

This is not about building new hardware. It's about using existing hardware in a coordinated way.

**The wood cuts itself when you understand the grain.**
