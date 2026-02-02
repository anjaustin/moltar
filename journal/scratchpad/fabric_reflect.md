# Reflections: The TriX Fabric Layer

## The Core Tension

Nodes 1, 2, 4, 5, and 10 all orbit the same question: **How does a hardware-level fabric know what to do without being told by the application?**

This is the delta. This is the sock that could be underwear.

Let me sit with this.

## Why the Tension Is False

I've been thinking about this as if the fabric needs to *understand* the workload. It doesn't. Consider:

A CPU cache doesn't understand what a program is doing. It just notices access patterns and keeps hot lines resident. A hardware prefetcher doesn't know about matrices — it detects sequential and strided access patterns and fetches ahead. A branch predictor doesn't understand conditionals — it recognizes patterns.

The fabric doesn't need to understand inference. It needs to **optimize the hardware response to observable patterns.** The patterns of LFM inference are:

1. **Large sequential reads** (weight matrices, scanned linearly)
2. **Small hot buffers** (activation vectors, reused across layers)  
3. **Predictable compute sequence** (same ops, same sizes, same order, every token)
4. **Memory-bandwidth bound** (more time waiting for data than computing)

These patterns are detectable and optimizable at the system level without any application knowledge.

## Resolving Node 2 vs Node 1: The Persistent Vulkan Process

I was confused about what the Vulkan process computes if it doesn't know about the application. But I was thinking too literally.

The persistent Vulkan process doesn't compute *for* llama.cpp. It **keeps the GPU warm and the memory coherent.** Its job is:

1. **Keep Vulkan context alive** — no teardown/rebuild cost
2. **Keep pipelines compiled** — shader compilation is a one-time cost that's already paid
3. **Keep buffers mapped** — HOST_COHERENT memory stays mapped, no map/unmap cycles
4. **Keep TMU caches warm** — periodically touch LUT textures so they stay in TMU cache
5. **Serve as a pre-warmed compute surface** — when GPU work IS needed (large batched ops, multi-token scenarios), the overhead is near zero because everything is already initialized

The 386us dispatch we measured includes pipeline creation, descriptor allocation, command buffer recording, queue submission. If all of that is pre-done and we're just signaling via timeline semaphore, the actual work-submission cost is the semaphore signal — which is a single atomic write to mapped memory. Nanoseconds.

But this raises the question: **who submits work to this pre-warmed GPU?** The answer might be: nobody, yet. The persistent process is Phase 1 of the fabric. It just keeps things warm. Phase 2, later, could add an IPC mechanism where a TriX-aware application *could* submit work — but that's optional. The value of Phase 1 alone is that when llama.cpp uses the Vulkan backend itself, the driver and GPU are already warm.

Wait. Actually, that's a deeper insight.

## The Real Insight: The Fabric Optimizes the Environment, Not the Computation

The fabric doesn't do any computation for the application. It **prepares the environment** so the application's own computation runs faster. Like pre-heating an oven. The food cooks itself — but it cooks faster in a hot oven.

What "pre-heating" means on this hardware:

**CPU preparation:**
- Pin inference-heavy threads to A78 big cores (sched_setaffinity / cpusets)
- Set CPU governor to "performance" to prevent DVFS downclocking during inference
- Pre-fault and lock memory pages (mlock) so no page faults during inference
- Set up huge pages if available for the weight mmap region

**Memory preparation:**
- Ensure the weight file is mmap'd with MADV_SEQUENTIAL for prefetch
- Use MADV_HUGEPAGE on the mmap region
- Pre-fault all weight pages so they're resident in physical memory
- Set up the ION/dmabuf allocator so runtime allocations land in Vulkan-visible memory

**GPU preparation:**
- Keep the Vulkan instance and device alive (prevents driver teardown/rebuild)
- Keep compute pipelines compiled and resident
- Keep LUT textures (sigmoid, tanh) loaded and cache-hot in TMU
- Keep shared buffers mapped HOST_COHERENT

**Thermal preparation:**
- Monitor thermal sensors
- If approaching thermal throttle, proactively reduce GPU clock to leave headroom for CPU
- Log thermal events for profiling

**The zero-copy piece** comes from the memory preparation: if all allocations land in unified HOST_COHERENT memory, and the mmap'd weights are in the same physical address space, then there's literally nothing to copy. CPU and GPU see the same bytes at the same address. This is already true on the Dimensity 930 — the fabric just makes sure nothing accidentally breaks this property (like a driver that shadow-copies buffers, or an allocator that puts data in a non-coherent region).

## Resolving Node 5: Zero-Multiplying Without Model Knowledge

The zero-multiplying insight from Yinsen: skip multiplications by zero-valued weights.

At the fabric level, this translates to: **if we know a memory region contains mostly zeros, we can optimize access patterns for sparsity.** This doesn't require knowing it's a "weight matrix." It requires knowing the statistical properties of the data.

But honestly — this might not belong in the fabric at all. Zero-multiplying is an algorithmic optimization that happens inside the compute kernel (the NEON dot product, the Vulkan shader). The fabric can't skip multiplications because the fabric doesn't multiply. The application does.

Where the fabric CAN help with sparsity:
- **Memory layout optimization**: Reorder data so zero regions are contiguous, enabling larger prefetch strides and fewer cache lines touched
- **Page-level optimization**: If entire pages are zero, don't even fault them in — let the CPU read zeros from the zero page
- **Prefetch optimization**: If the access pattern skips zero blocks, the prefetcher should be tuned for that stride

But the actual "skip the multiply" has to happen in the code that does the multiply — which is inside llama.cpp or Yinsen's NEON kernels. Unless... the fabric provides optimized NEON routines via a shared library that the system's linker prefers over the default ones? Like providing an optimized BLAS that the application links against without knowing it?

That's actually how it works in the real world. Apple's Accelerate framework, Intel's MKL, ARM's ACL — they're system-level compute libraries that applications link against, and the system provides hardware-optimized implementations. The application calls `cblas_sgemv`, and the system library decides whether to use NEON, SVE, or AMX.

## Resolving Node 10: What Crosses the Boundary

The boundary between fabric and application is:

1. **Memory addresses** — the fabric controls where memory lives (which physical pages, which cache coloring, which NUMA node). Applications get pointers. The fabric decides what those pointers point to physically.

2. **CPU scheduling** — the fabric controls which core runs which thread. The application doesn't set its own affinity; the fabric's scheduler does.

3. **System libraries** — the fabric provides optimized implementations of standard functions (memcpy, math functions, BLAS). Applications call standard APIs and get fabric-optimized code.

4. **GPU state** — the fabric keeps the GPU warm. If the application uses Vulkan, it benefits from a pre-warmed driver. If it doesn't, the fabric's GPU process stays idle but ready.

Nothing "crosses" the boundary in the sense of an API call. The fabric is the ground the application stands on. The application doesn't interact with the ground — it just benefits from solid footing.

## Remaining Questions

1. **How do we actually pin llama.cpp threads on Android?** We'd need root or a shell context. Can we use `taskset` from adb shell? Or do we need a wrapper script?

2. **Can LD_PRELOAD work on Android?** Android's linker (linker64) supports LD_PRELOAD but with restrictions. For adb shell execution it should work. For installed apps, SELinux blocks it.

3. **Timeline semaphore latency** — we need to probe this. What's the actual cost of CPU→GPU signaling via timeline semaphore vs full queue submission? This is the next critical measurement.

4. **Does keeping the Vulkan context alive actually help?** The PowerVR driver might already keep the GPU initialized. We should measure: does a second dispatch after a warm-up period have lower latency than the first?

5. **Thermal management** — what sensors are exposed via sysfs on this device? Can we read GPU/CPU temperature and clock from /sys/?

## What I Now Understand

The fabric is three things:

1. **An environment optimizer** — CPU governors, core pinning, memory management, thermal monitoring. Makes the hardware run at its best.

2. **A persistent GPU context** — keeps Vulkan warm, pipelines compiled, buffers mapped. Eliminates cold-start costs for any GPU work.

3. **A zero-copy memory fabric** — ensures all allocations live in unified coherent memory, weights are mmap'd optimally, pages are pre-faulted and locked.

The applications are unaware. They just run faster because the hardware beneath them is managed intelligently.

The zero-multiplying piece lives in optimized compute kernels (NEON, Vulkan shaders) that the fabric could provide as system libraries — but that's a later phase. Phase 1 is the environment.
