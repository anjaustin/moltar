# Reflections: GPU as Memory Fabric Layer

## Core Insight

The GPU prefetch idea rests on a stack of unverified assumptions. Each assumption is a potential failure point:

```
Assumption Stack (bottom to top):
─────────────────────────────────
A1: Row buffer misses are significant (Node 1)
    ↑ IF FALSE: No problem to solve
A2: GPU and CPU can share row buffers (Node 3)
    ↑ IF FALSE: GPU access doesn't warm CPU path
A3: GPU access doesn't steal bandwidth from CPU (Node 2)
    ↑ IF FALSE: Net negative, we make things worse
A4: Hardware prefetcher isn't already doing this (Node 4)
    ↑ IF FALSE: We're duplicating work, adding overhead
A5: Timing window is sufficient (Node 5 vs Node 8)
    ↑ IF FALSE: Rows close before CPU arrives
```

**The key realization**: We should falsify from the bottom up. If A1 is false, none of the rest matters. If A3 is false, even true A1 and A2 don't help us.

## The Falsification Order

1. **Probe A: Bandwidth Interference** (Tests A3)
   - Run CPU matvec alone, measure bandwidth
   - Run CPU matvec with GPU doing sustained reads, measure bandwidth
   - If CPU bandwidth drops >5%, STOP. The idea is dead.
   - Time: 1 hour

2. **Probe B: Row Buffer Warming** (Tests A2, indirectly A1)
   - Run CPU reads to address X after GPU touched X vs untouched
   - Measure latency difference
   - If no difference, GPU doesn't warm CPU's path. STOP.
   - Time: 2 hours

3. **Probe C: Sequential Access Pattern** (Tests A4)
   - Profile CPU matvec without prefetch
   - Look for hardware prefetcher activity (perf counters if available)
   - If HW prefetch is already saturating bandwidth, we can't help. STOP.
   - Time: 2 hours

4. **Probe D: End-to-End Test** (Tests A5 and the full system)
   - Only if A, B, C pass
   - Build actual prefetch shader
   - Measure tok/s with and without
   - Time: 4 hours

**Total time if all probes fail early: 1 hour**
**Total time if all probes pass: 9 hours**

## Resolved Tensions

### Node 2 vs Node 5 (Bandwidth vs Timing)
Resolution: Probe A answers this definitively. We don't need to theorize about whether GPU bandwidth is "free" — we measure it.

### Node 4 vs Everything (Hardware Prefetcher)
Resolution: If the hardware prefetcher is already effective, our matvec bandwidth should be close to theoretical max. We measured 51%. That 49% gap suggests the HW prefetcher isn't solving everything. But the gap could be other things (Node 1 options). Probe C investigates.

### Node 8 vs Node 5 (Row Buffer Size vs Prefetch Amount)
Resolution: This tells us the optimal prefetch distance. If rows stay open for ~100us, prefetching 5 layers ahead is too far — the first layers' rows will close. We should prefetch just 1-2 layers ahead. The exact number depends on row buffer policy, which Probe B might reveal indirectly through latency patterns.

### Node 9 (Cache Hierarchy)
Resolution: This is knowable but not critical. If GPU warms shared L3, great. If it only warms DRAM row buffers, that might still help. Probe B measures the end effect regardless of mechanism.

## What I Now Understand

**The core question is not "can we build a prefetch shader?" but "is the memory system amenable to prefetch assistance?"**

The answer depends on hardware we don't control and can't fully inspect. But we CAN measure the effects. The probes are designed to produce actionable yes/no answers with minimal investment before hitting a "no."

**The pivot (Node 12) is valuable regardless.** If GPU prefetch fails, GPU as "memory janitor" (handling cold paths, background loads, KV cache eviction) might work. These are async operations where GPU's 270us dispatch overhead doesn't matter because they're not on the critical path.

## The Laundry Method Applied

**Coarse buckets:**
- Bucket A: Memory controller behavior (Probes A, B)
- Bucket B: CPU prefetcher behavior (Probe C)
- Bucket C: End-to-end integration (Probe D)

**The delta (boundary items):**
- What if GPU access SOMETIMES helps and SOMETIMES hurts depending on access pattern? We might see inconsistent results in Probe A.
- What if the improvement is real but smaller than measurement noise? We need statistical rigor in Probe D.

**Partition first, search within:**
- Don't build Probe D until A, B, C pass.
- Don't theorize about row buffer policies; measure latencies.
- Don't optimize the prefetch shader until we know prefetch helps.

## Remaining Questions

1. Can we access memory controller performance counters on MT6855? (Would make Probe B much more informative)
2. Does MediaTek expose DRAM timing info anywhere in /sys or /proc?
3. What is the actual shared L3 size, if any, on Dimensity 930?
4. Is there prior art on GPU-assisted prefetch on any mobile platform?

## What Would This Look Like If It Were Easy?

If the memory system were perfect for this use case:
- GPU reads would be free (zero interference with CPU)
- Rows would stay open indefinitely
- GPU could prefetch entire model at startup and it would stay "warm"
- tok/s would improve 20-30%

Reality is messier. But the probes tell us how messy.

## The Simple Summary

**Don't build the thing. Build the test for whether to build the thing.**

Probe A is 50 lines of code and 1 hour. It either kills the idea or justifies the next probe. This is the Lincoln Manifold in action: the first chop reveals the dullness of the blade.
