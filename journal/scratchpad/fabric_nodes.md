# Nodes of Interest: The TriX Fabric Layer

## Node 1: The Fabric Is Infrastructure, Not Application Code

The fabric doesn't do inference. It doesn't know what a transformer is. It manages hardware resources — memory mapping, compute scheduling, cache coherency, thermal state — so that whatever runs on top runs better. This is the distinction between a device driver and an application.

Why it matters: This constrains the design. We can't hook into ggml ops. We can't parse GGUF. We can't know what layer we're on. We can only know about memory, cores, caches, and compute units.

## Node 2: The Persistent Vulkan Process Eliminates Per-Dispatch Overhead

We measured 386us per dispatch. But we were creating/submitting/waiting per call. A persistent process that keeps the Vulkan context, pipelines, and command buffers alive — and uses timeline semaphores for signaling — could reduce the CPU→GPU submission cost to a semaphore signal, which is nanoseconds, not microseconds.

Why it matters: This is the difference between "GPU is useless for small ops" (our TMU probe conclusion) and "GPU is free for small ops." If signaling cost drops from 386us to <10us, the entire calculation about GPU viability reverses.

Tension with Node 1: A persistent Vulkan process needs to know *what* to compute. If it's truly infrastructure, how does it know which buffers contain activations that need sigmoid applied?

## Node 3: Unified Memory Means Data Never Moves

The Dimensity 930 has a single LPDDR4X bus. CPU and GPU share physical memory. HOST_COHERENT memory type means CPU writes are immediately visible to GPU without explicit flush. This is the hardware foundation for zero-copy.

Why it matters: On discrete GPU systems (desktop), the copy cost dominates. Here it's literally zero. The data is already there. The only cost is coherency traffic on the bus, which happens automatically.

## Node 4: The LD_PRELOAD / Custom Allocator Gateway

If the fabric provides a custom memory allocator (via LD_PRELOAD interception of malloc, or via Android's ION/dmabuf), then every allocation an application makes can be placed in Vulkan-visible, HOST_COHERENT memory. The application doesn't know this. It just got a pointer. But now the fabric's GPU process can see everything.

Why it matters: This is the mechanism that makes the fabric invisible. No API changes. No recompilation. Just a smarter allocator underneath.

Tension with Node 2: Knowing that memory is Vulkan-visible doesn't tell the GPU process what to *do* with it. We still need a signaling mechanism.

## Node 5: Zero-Multiplying Requires Weight Knowledge

"The FPU doesn't care what it multiplies by" — but skipping multiplications entirely requires knowing which weights are zero. In Yinsen's ternary sparse model, 81% are zero, so you get 2.73x speedup. But this requires pre-analysis of the weight data.

Why it matters: This is where the fabric touches model data, which seems to violate Node 1's constraint of being model-agnostic.

Tension with Node 1: Can the fabric be model-agnostic AND know about weight sparsity? Resolution might be: the fabric doesn't know about "models" — it knows about "memory regions with statistical properties." It can detect sparsity in any buffer without knowing it's a weight matrix.

## Node 6: The Cache Coherency Streaming Model

Tripp mentioned the Vulkan process "living in the cache coherency perpetually." This means: the GPU's working set stays cache-hot. The compute shaders, the activation buffers, the LUT textures — they're always resident. No cold-start. No cache miss penalty on first access.

Why it matters: Cache residency is a second-order effect that can be as large as compute optimization. A warm L1/L2 cache line is 1ns access. A cold DRAM fetch is 50-100ns. If the fabric keeps the right data hot, everything goes faster.

## Node 7: Timeline Semaphores as the Zero-Overhead Signal

VK_KHR_timeline_semaphore is present on this device. Timeline semaphores allow CPU and GPU to synchronize via a monotonically increasing counter without full queue submission. CPU increments the counter → GPU wakes up and processes. No command buffer recording, no queue submission, no fence wait. Just an atomic counter update.

Why it matters: This is potentially the mechanism that makes the persistent Vulkan process work with near-zero latency signaling. The 386us dispatch includes command buffer recording + queue submit + driver kick + fence signal. Timeline semaphores bypass almost all of that.

## Node 8: The "Streaming" Metaphor Is Literal

The GPU process isn't dispatched on demand. It's literally streaming — a compute shader that runs in an infinite loop (or is rapidly re-signaled), processing whatever data appears in its mapped buffers. Like a DMA controller or a hardware accelerator, not a software function call.

Why it matters: This changes the mental model completely. We're not "calling the GPU." We're "feeding a hardware pipeline that's already running."

Tension with Node 2: Vulkan doesn't natively support infinite-loop compute shaders (no persistent threads like CUDA). The streaming model needs to be implemented via rapid re-dispatch using timeline semaphores, or via a long-running shader that spins on a buffer flag.

## Node 9: Core Pinning and DVFS Are Free Performance

Linux sched_setaffinity, cpusets, and devfreq governors can be set from userspace on Android (with shell/root). Pinning llama.cpp threads to the A78 big cores, setting the CPU governor to "performance" during inference, and managing thermal headroom — these are pure system-level optimizations that need no application changes.

Why it matters: This is the lowest-hanging fruit in the fabric. No Vulkan, no memory interception, no complexity. Just: "make the CPU go faster during inference." Could easily be worth 10-20% on its own.

## Node 10: The Delta — What Exactly Crosses the Boundary?

The fabric is invisible to applications. But *something* must cross the boundary between "application writes data" and "fabric processes data." What is that something?

Possibilities:
- Memory address ranges (fabric monitors specific regions)
- Cache line invalidation signals (hardware-level)
- Heuristic detection (fabric sees a pattern of memory access that looks like inference)
- Explicit shared memory regions (application allocates from a known pool, fabric watches the pool)

This is the critical boundary — the "delta" in LMM terms. The sock that could be underwear. Getting this wrong means the fabric either can't see the data (useless) or has to intercept too aggressively (brittle).

## Node 11: What Worked on M4?

I don't have details about what was done on the M4. But the M4 is Apple Silicon — unified memory, GPU and CPU share physical RAM, hardware cache coherency. The same principles apply: zero-copy, keep compute units warm, manage scheduling. The Dimensity 930 has the same unified memory architecture (less bandwidth, smaller caches, but same principle).

Why it matters: The M4 success proves the concept. The Motorola implementation needs to adapt it to ARM big.LITTLE + PowerVR + MediaTek driver quirks.
