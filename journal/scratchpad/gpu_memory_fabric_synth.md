# Synthesis: GPU as Memory Fabric Layer (FINAL)

## Executive Summary

**CONCEPT FALSIFIED.** GPU cannot serve as a memory fabric layer for CPU inference on this hardware.

The row buffer speedup exists (2.66x for sequential vs random access) but **cannot be exploited** because:
1. Cross-core prefetch has cache coherency overhead (0.86x - makes things worse)
2. Concurrent GPU traffic causes bandwidth contention (0.75x - even worse)

The memory controller is already well-optimized. GPU traffic doesn't help; it hurts.

---

## Probe Results Summary

### Probe A: Bandwidth Interference - PASSED ✅
**Question**: Does GPU memory traffic steal CPU bandwidth?

```
Results (with CPUs in performance governor, DRAM at 4266 MHz):
- CPU alone:       10.01 GB/s
- CPU with GPU:    10.01 GB/s (0% loss!)
- GPU bandwidth:    8.03 GB/s
- Combined:        18.04 GB/s
```

**Finding**: Concurrent access achieves ~18 GB/s combined, exceeding the previous "11 GB/s ceiling" which was an artifact of governor instability. GPU traffic causes **zero CPU bandwidth loss**.

### Probe B: Latency Reduction - FAILED ❌

Multiple tests performed:

| Test | Question | Result |
|------|----------|--------|
| v4 Test 1 | Cache prefetch benefit | **1.0x** - HW already optimal |
| v4 Test 2 | Row locality (seq vs random) | **2.66x** - huge benefit exists! |
| v4 Test 3 | Prefetch reduces variance | **15% → 3% CV** - yes |
| v5 | Cross-core cache prefetch | **0.86x** - coherency overhead hurts |
| v6 | Concurrent row warming | **0.75x** - contention kills benefit |

**Key Insight**: The 2.66x row buffer benefit **exists** but **cannot be transferred** from one core to another due to cache coherency overhead.

---

## Why It Failed

### The Row Buffer Paradox

```
                      DRAM ROW BUFFER MECHANICS
┌────────────────────────────────────────────────────────────────┐
│                                                                 │
│   DRAM Row Buffer:                                              │
│   • Opening a row takes ~15-20ns                                │
│   • Reading from open row takes ~5ns                            │
│   • Sequential access keeps row open → 2.66x faster!            │
│                                                                 │
│   THE PROBLEM:                                                  │
│   • GPU reads → data goes to GPU cache (or system cache)        │
│   • CPU needs same data → cache coherency transfer              │
│   • Transfer latency EXCEEDS row buffer savings                 │
│                                                                 │
│   GPU prefetch path:                                            │
│   DRAM → [row open: 15ns] → GPU/System Cache → [coherency: 50ns+] → CPU
│                                                                 │
│   Direct CPU path:                                              │
│   DRAM → [row open: 15ns] → CPU                                 │
│                                                                 │
│   GPU "prefetch" is SLOWER, not faster!                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Why Probe A Passed But Probe B Failed

Probe A measured **bandwidth** - both CPU and GPU can read independently at high speed without stealing from each other. The memory controller handles this well.

Probe B tested **cooperation** - having one core read to help another. This fails because:
1. Cache coherency protocol adds latency
2. Data read by one core goes to that core's cache
3. Getting it to another core requires cache-to-cache transfer
4. This transfer takes longer than just reading DRAM directly

---

## Empirical Data

### Probe B v4 - Row Locality Test
```
Sequential access: 41 µs for 10K reads  = 4.1 ns/read
Random access:    109 µs for 10K reads = 10.9 ns/read
Row buffer speedup: 2.66x
```

### Probe B v5 - Cross-Core Prefetch
```
NO PREFETCH:   p50 = 141 µs (14.1 ns/read)
WITH PREFETCH: p50 = 164 µs (16.4 ns/read)
Speedup: 0.86x (SLOWER!)
```

### Probe B v6 - Concurrent Row Warming
```
CPU ALONE:  p50 = 100 µs
CPU + GPU:  p50 = 133 µs
Speedup: 0.75x (EVEN SLOWER!)

Variance:
CPU ALONE: CV = 6.5%
CPU + GPU: CV = 16.1% (WORSE!)
```

---

## Conclusions

### What We Learned

1. **Row buffer locality is real and significant** - 2.66x speedup for sequential vs random access
2. **Cross-core cache sharing has overhead** - cache coherency negates any prefetch benefit
3. **Concurrent access causes contention** - even reading "different" data hurts performance
4. **The memory controller is already optimized** - hardware prefetch works well

### What Doesn't Work

- GPU reading ahead to "warm" DRAM rows for CPU
- Using GPU to bring data into shared cache for CPU
- Concurrent GPU traffic to keep DRAM rows open

### What Might Work (Pivot Options)

The GPU can still be useful, but **not as a memory fabric**. Alternative uses:

1. **Background Model Loading**
   - GPU loads model weights while CPU does inference
   - Works because they're not competing for the same data
   - GPU → UFS/storage transfer, CPU → DRAM transfer
   
2. **KV Cache Eviction**
   - GPU manages swapping KV cache to UFS
   - Happens during generation pauses (between prompts)
   - No contention because CPU is idle
   
3. **Async Memory Management**
   - GPU defragments memory, compacts buffers
   - Runs when CPU has spare cycles
   - Improves memory efficiency without runtime cost

4. **Actual GPU Compute**
   - Use GPU for what it's designed for: parallel compute
   - Small matrix ops, attention scoring
   - Requires Vulkan/OpenCL implementation

---

## Recommended Next Steps

### Immediate: Abandon Memory Fabric Concept
The probes definitively show it doesn't work on this hardware. Don't spend more time trying to make it work.

### Short-term: Optimize What We Have
- Keep DRAM at 4266 MHz (lower variance)
- Keep CPUs in performance governor for inference
- Ensure sequential memory access patterns in llama.cpp

### Medium-term: Explore GPU Compute
If we want to use the PowerVR BXM-8-256, use it for actual computation:
- Look into Vulkan backend for llama.cpp
- Test if small ops can offload profitably
- Consider attention computation offload

### Long-term: Memory Efficiency
Since bandwidth isn't the bottleneck, focus on memory efficiency:
- Smaller quantization (Q4_0 → Q3_K)
- Better KV cache management
- Model architecture optimization

---

## Files Created

### On Phone (`/data/local/tmp/`)
```
probe_a         # Original bandwidth test
probe_a_v2      # Stabilized version
probe_b         # Initial latency test (timer too coarse)
probe_b_v2      # Cycle counter version
probe_b_v3      # Batch latency test
probe_b_v4      # Large-scale latency + row buffer test
probe_b_v5      # Cross-core prefetch test
probe_b_v6      # Concurrent row warming test
```

### On Mac (`/tmp/`)
```
probe_b_v4.c    # Source
probe_b_v5.c    # Source
probe_b_v6.c    # Source
```

---

## Key Insight for Future Work

> **Cache coherency is the enemy of cross-core prefetch.**
> 
> On shared-memory systems with coherent caches, having one core read
> data to "help" another core is counterproductive. The coherency
> protocol overhead exceeds the DRAM latency savings.
> 
> Use multiple cores for **parallel independent work**, not for
> **cooperative prefetching**.

---

*Investigation completed: 2026-02-04*
*Result: Concept falsified, pivot recommended*
