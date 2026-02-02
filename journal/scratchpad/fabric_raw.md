# Raw Thoughts: The TriX Fabric Layer

## Stream of Consciousness

The fabric is not software that runs inference. The fabric is a system-level process that makes the hardware behave better for any workload — llama.cpp, ExecuTorch, whatever. It sits beneath everything. Applications don't know it exists. They just see faster hardware.

My first instinct was to build a ggml backend. That's wrong. That's application-level. Tripp is talking about something deeper — a process that owns the hardware resources and manages them intelligently. The applications are guests. The fabric is the host.

The key thing Tripp keeps coming back to is: minimal copying, zero-multiplying. These are hardware-level concerns. Data shouldn't move between CPU and GPU — it's the same physical memory on this SoC. The fabric knows this. It keeps buffers mapped, keeps the Vulkan context alive, keeps command buffers pre-recorded. When llama.cpp does a malloc and writes weights there, the fabric's GPU process can already see those bytes through the unified memory. No copy needed. No staging buffer. No transfer command.

The persistent Vulkan process is the interesting piece. Right now we measured 386us dispatch latency. But how much of that is per-submission driver overhead vs actual hardware scheduling? If you keep a Vulkan compute pipeline running as a stream — a daemon that perpetually waits for work, processes it, signals completion — you eliminate the per-dispatch overhead entirely. It's like the difference between opening a new TCP connection per request vs keeping a persistent connection. The connection setup is the expensive part.

But what does this streaming process actually compute? If llama.cpp doesn't talk to it, how does it know what to do? This is the part I'm fuzzy on. There are a few possibilities:

1. It watches memory. When llama.cpp writes to certain addresses, the fabric detects it and proactively starts computing (activations, reductions) on the GPU. Like a hardware prefetch but for compute.

2. It manages the memory allocator. When llama.cpp calls malloc, the fabric's custom allocator returns addresses from pre-mapped Vulkan-visible memory. Now everything llama.cpp allocates is automatically GPU-accessible.

3. It manages CPU affinity and scheduling. It pins llama.cpp's heavy threads to the A78 big cores, keeps the little cores for housekeeping, manages DVFS to prevent thermal throttling.

4. It manages cache coherency. It ensures the L2 cache lines that hold the hot activation vectors stay resident, that weight prefetch streams are flowing, that the CPU isn't fighting the GPU for cache.

The zero-multiplying piece connects to Yinsen's ternary sparsity work. If 81% of weights are zero, you can skip those MACs. But llama.cpp's Q4_0 format doesn't know about sparsity — it stores all weights densely. The fabric could intercept at a lower level: when the CPU starts a matvec, the fabric knows (from pre-analysis of the weight distribution) which blocks are effectively zero and could bias the memory prefetch to skip them. Or more radically: the fabric could rewrite the weight layout on disk/in memory to a sparse format that the hardware processes more efficiently, while presenting the same logical tensor to the application.

Wait — that last one is interesting. The fabric doesn't modify the application. But it could modify the data's physical layout in memory to match what the hardware processes fastest. Like a filesystem that reorganizes blocks for sequential access. The application sees the same file, but I/O is faster because the blocks are physically ordered.

What scares me: the boundary between "invisible hardware optimization" and "intercepting application behavior." If we have to hook into llama.cpp's memory allocations to make this work, that's not truly invisible. If we have to run a modified kernel driver, that's not portable. Where's the line?

What would this look like if it were easy? A daemon that starts at boot, maps all physical memory as Vulkan-visible, keeps a persistent compute pipeline warm, and uses hardware performance counters to detect when inference workloads are running and automatically optimize scheduling. Everything else is a detail.

## Questions Arising

- How does the fabric detect that llama.cpp is running inference vs doing something else?
- How does the persistent Vulkan process receive work without llama.cpp sending it explicitly?
- Can we use Android's ION/dmabuf allocator to ensure all allocations are Vulkan-visible?
- Can we use Linux cgroups/cpusets to manage core affinity without modifying the application?
- What's the mechanism for zero-copy between CPU and GPU on MediaTek? Is it truly automatic with HOST_COHERENT, or does the driver do invisible copies?
- The M4 fabric — what specifically was done there? What's the analog on ARM/Android?
- Does the Vulkan spec allow a persistent compute pipeline that runs indefinitely, waiting on timeline semaphores for work submission?
- Can we use VK_KHR_timeline_semaphore (which this device has!) for CPU→GPU signaling without full dispatch overhead?
- What does the LD_PRELOAD approach look like — intercept malloc to return Vulkan-mapped memory?

## First Instincts

- The persistent Vulkan process is the centerpiece. Everything else is supporting infrastructure.
- LD_PRELOAD to intercept memory allocation is probably the mechanism for making it invisible.
- Timeline semaphores are the key to low-latency CPU→GPU signaling.
- The "streaming" model means the GPU pipeline is always running, polling or waiting on a semaphore, not being started and stopped per operation.
- The zero-multiply piece might need to happen at weight-load time — reformat weights into a layout that naturally skips zeros.
- This is more like writing a device driver than writing application code.
