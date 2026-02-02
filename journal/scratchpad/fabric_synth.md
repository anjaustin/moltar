# Synthesis: The TriX Fabric Layer

## Architecture

The TriX Fabric is a **system-level daemon** that optimizes the Motorola Dimensity 930 hardware for inference workloads. Applications (llama.cpp, ExecuTorch, etc.) run unmodified. They see faster hardware, not a new API.

```
┌─────────────────────────────────────────────────────┐
│  llama.cpp / ExecuTorch / any application           │
│  (unmodified, unaware of fabric)                    │
├─────────────────────────────────────────────────────┤
│  Android userspace (libc, linker, Vulkan loader)    │
├─────────────────────────────────────────────────────┤
│  ╔═══════════════════════════════════════════════╗   │
│  ║           T R I X   F A B R I C              ║   │
│  ║                                               ║   │
│  ║  ┌───────────┐ ┌──────────┐ ┌─────────────┐  ║   │
│  ║  │ CPU Sched │ │ Mem Mgr  │ │ GPU Keeper  │  ║   │
│  ║  │           │ │          │ │             │  ║   │
│  ║  │ • affinity│ │ • mmap   │ │ • Vulkan ctx│  ║   │
│  ║  │ • governor│ │ • madvise│ │ • pipelines │  ║   │
│  ║  │ • thermal │ │ • mlock  │ │ • buffers   │  ║   │
│  ║  │ • DVFS    │ │ • hugepg │ │ • TMU LUTs  │  ║   │
│  ║  └───────────┘ └──────────┘ └─────────────┘  ║   │
│  ║                                               ║   │
│  ╚═══════════════════════════════════════════════╝   │
├─────────────────────────────────────────────────────┤
│  Linux kernel (scheduler, VMM, devfreq, thermal)    │
├─────────────────────────────────────────────────────┤
│  Hardware: 2×A78 + 6×A55 │ PowerVR BXM-8-256       │
│  LPDDR4X unified memory (~13 GB/s)                  │
└─────────────────────────────────────────────────────┘
```

The fabric has three subsystems:

### 1. CPU Scheduler (cpusched)

Manages core affinity, frequency governors, and thermal headroom.

**What it does:**
- Detects inference workload (high CPU utilization on sustained sequential memory reads)
- Pins compute-heavy threads to A78 big cores via sched_setaffinity
- Sets CPU governor to "performance" during inference bursts
- Monitors thermal sensors; proactively throttles GPU to save thermal budget for CPU
- Restores default scheduling when inference ends

**Implementation:** Shell script or C daemon using sysfs interfaces:
- `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- `/sys/devices/system/cpu/cpu*/online`
- `taskset` / `sched_setaffinity()` for thread pinning
- `/sys/class/thermal/thermal_zone*/temp` for thermal monitoring

**Expected impact:** 10-20% improvement from avoiding DVFS downclocking and keeping work on big cores.

### 2. Memory Manager (memmgr)

Optimizes memory layout, residency, and access patterns for the unified memory architecture.

**What it does:**
- Pre-faults and mlocks weight file pages (no page faults during inference)
- Sets madvise hints: MADV_SEQUENTIAL for weight reads, MADV_WILLNEED for prefetch
- Ensures all allocations land in HOST_COHERENT memory regions
- Configures transparent huge pages for the weight mmap region
- Monitors memory pressure and defends inference working set from eviction

**Implementation:** Wrapper script that runs before the application:
```bash
# Pre-fault the model file into page cache
cat /data/local/tmp/LFM2-350M-Q4_0.gguf > /dev/null
# Launch with optimized memory hints
trix_fabric_launch llama-cli --model /data/local/tmp/LFM2-350M-Q4_0.gguf ...
```

The launcher uses mmap + mlock + madvise on the model file before exec'ing the application. The application inherits the pre-faulted, locked file mapping.

**Expected impact:** Eliminates page fault stalls (could be significant for first-token latency), improves prefetch hit rate.

### 3. GPU Keeper (gpukeeper)

Persistent Vulkan process that keeps the GPU context warm and coherent buffers mapped.

**What it does:**
- Initializes Vulkan instance, device, compute queue at daemon start
- Pre-compiles common compute pipelines (sigmoid, RMSNorm, softmax, matvec)
- Allocates HOST_COHERENT buffers for activation vectors (D_MODEL=1024)
- Loads TMU LUT textures (sigmoid, tanh, exp) in R16_SFLOAT with LINEAR sampler
- Keeps everything alive — never tears down, never unmaps
- Uses VK_KHR_timeline_semaphore for low-latency CPU→GPU signaling
- Exposes a simple IPC interface (Unix domain socket or shared memory flag) for future TriX-aware applications to submit GPU work without Vulkan setup cost

**What it does NOT do:**
- Does not intercept application Vulkan calls
- Does not compute for the application unless explicitly asked via IPC
- Does not modify application behavior

**Expected impact:** 
- Eliminates GPU cold-start cost if/when GPU is used
- Keeps TMU caches warm for LUT activations
- Future: enables sub-10us GPU work submission via timeline semaphores (vs 386us today)

---

## Key Decisions

1. **Fabric is a daemon, not a library.** It runs as a separate process. Applications are not linked against it, don't call it, don't know about it. (From Node 1: infrastructure, not application code.)

2. **CPU optimization is Phase 1.** Core pinning and DVFS are the cheapest wins with the most impact. No Vulkan complexity needed. (From Reflection: pre-heating the oven.)

3. **Memory optimization is Phase 2.** Pre-faulting, mlock, madvise, huge pages. Still no GPU involvement. (From Node 3: unified memory is already zero-copy; just don't break it.)

4. **GPU keeper is Phase 3.** The persistent Vulkan process. Most complex, but builds on the foundation of Phases 1-2. (From Node 2: eliminates dispatch overhead, but only matters when GPU work is submitted.)

5. **Zero-multiplying is NOT in the fabric.** It belongs in optimized compute kernels (NEON, Vulkan shaders) that could be provided as system libraries in a future Phase 4. The fabric manages hardware; it doesn't implement algorithms. (From Reflection: resolved Node 5 tension.)

6. **Timeline semaphore probe is the next critical measurement.** Before building the full GPU keeper, we need to verify that timeline semaphore signaling is actually fast (<10us) on this driver. (From Node 7: this is hypothesized, not proven.)

---

## Implementation Plan

### Phase 1: CPU Scheduler (trix_cpusched)

A shell script + small C helper that:
1. Reads /proc/cpuinfo to identify big/little cores
2. Sets CPU governor to "performance" on big cores
3. Provides a `trix_pin` wrapper that sets affinity before exec
4. Monitors thermal sensors and logs to stderr

**Deliverable:** `trix_cpusched.sh` + `trix_pin` binary
**Test:** Run llama.cpp with and without. Measure tok/s difference.

### Phase 2: Memory Manager (trix_memmgr)

A launcher program that:
1. mmap's the model file with MAP_POPULATE | MAP_LOCKED
2. Sets MADV_SEQUENTIAL on the mmap region
3. Optionally enables transparent huge pages
4. exec's the target application

**Deliverable:** `trix_launch` binary
**Test:** Measure first-token latency and sustained tok/s with/without.

### Phase 3: GPU Keeper (trix_gpukeeper)

A C daemon that:
1. Initializes Vulkan 1.1, creates device with FP16/INT8/16-bit/8-bit storage
2. Creates compute pipelines for: sigmoid, tanh, silu, rmsnorm, softmax
3. Loads 256-entry R16_SFLOAT LUT textures for each activation
4. Allocates HOST_COHERENT buffers for D_MODEL=1024 activation vectors
5. Creates timeline semaphore for CPU→GPU signaling
6. Enters main loop: wait on semaphore → execute pre-recorded command buffer → signal completion
7. Exposes shared-memory IPC: client writes {pipeline_id, input_offset, output_offset} → signals semaphore → waits for completion semaphore

**Deliverable:** `trix_gpukeeper` daemon + `trix_gpu_client.h` (simple IPC header)
**Test:** Measure round-trip latency: client signal → GPU compute → client reads result. Compare to our 386us baseline.

### Phase 4 (Future): Optimized Compute Libraries

Provide hardware-optimized NEON/Vulkan implementations of key primitives:
- Q4_0 matvec with ghost-stream prefetch (from Yinsen)
- Ternary sparse dot product (zero-multiply, from Yinsen)
- Activation LUT+lerp via NEON (from Yinsen activation_chip)
- Fused RMSNorm+RoPE

These could be packaged as a shared library loadable via LD_PRELOAD or linked directly by TriX-aware applications.

---

## Success Criteria

- [ ] Phase 1: llama.cpp tok/s improves measurably (>5%) with CPU scheduler vs without
- [ ] Phase 2: First-token latency improves measurably with memory manager
- [ ] Phase 3: GPU keeper achieves <50us round-trip work submission via timeline semaphore
- [ ] Phase 3: GPU keeper stays alive for >1 hour without memory leak or crash
- [ ] All measurements are reproducible on the Moto G Power via adb
- [ ] llama.cpp binary is completely unmodified — same binary, same arguments
- [ ] Fabric processes can be started/stopped independently

---

## What Surprised Me

The synthesis is simpler than I expected. The fabric is not a complex interposition layer. It's three independent daemons that each manage one aspect of the hardware. They don't need to coordinate with each other (at least not in Phase 1-3). They don't need to understand applications. They just make the hardware behave better.

The persistent Vulkan process is the most novel piece, but even it is straightforward once you see it as a "keep things warm" service rather than a "compute for the application" service.

The zero-multiplying piece — which I thought was central — actually belongs in a different layer (compute kernels, not fabric). The fabric creates the conditions for fast computation. The computation itself lives elsewhere.
