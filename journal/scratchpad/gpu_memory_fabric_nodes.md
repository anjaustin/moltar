# Nodes of Interest: GPU as Memory Fabric Layer

## Node 1: The 49% Bandwidth Gap
CPU matvec achieves 51% of theoretical DRAM bandwidth (measured: 6.6 GB/s of 13 GB/s theoretical at old clock, now potentially 25.6 GB/s at 3200 MHz). The gap could be:
- Row buffer misses (GPU prefetch could help)
- Cache misses (GPU prefetch might help via shared L3)
- Bank conflicts (GPU prefetch cannot help)
- Compute/memory interleaving stalls (GPU prefetch cannot help)
- Memory controller arbitration overhead (GPU prefetch would make worse)

Why it matters: If row buffer misses aren't the dominant factor, the entire premise fails.

## Node 2: The Shared Bus Paradox
GPU and CPU share the same memory bus. Any GPU bandwidth is bandwidth the CPU can't use simultaneously. The math:
- GPU prefetch during matvec steals bandwidth from that matvec
- The matvec we're "helping" (layer N+2) isn't running yet
- Net effect: we slow down layer N to maybe speed up layer N+2

Tension with Node 1: Even if row buffer misses are significant, we might not have "free" bandwidth to exploit.

Why it matters: This could be a zero-sum or negative-sum game.

## Node 3: Memory Controller Opacity
We don't know the MT6855 memory controller's row buffer policy. Options:
- Open-page: rows stay open after access, subsequent accesses to same row are fast
- Closed-page: rows close after each access, every access pays the open penalty
- Adaptive: controller guesses based on access patterns

Modern mobile SoCs often use adaptive policies optimized for mixed traffic (CPU + GPU + display). The policy might already be well-tuned.

Why it matters: If the controller is already smart, our "help" might be interference.

## Node 4: ARM Hardware Prefetcher
The Cortex-A78 has sophisticated hardware prefetchers:
- L1 stride prefetcher
- L2 stream prefetcher
- TLB prefetcher

These observe access patterns and speculatively fetch data. If GPU traffic creates "noise" in the memory access stream, it could confuse the prefetchers, causing them to fetch wrong data or stop prefetching entirely.

Tension with Node 2: We might be "helping" by creating interference with a system that's already helping.

Why it matters: Hardware prefetchers are optimized over years of silicon design. Our shader is a blunt instrument.

## Node 5: The Timing Window
GPU dispatch is 270us. CPU matvec is 758us. The GPU has one dispatch window to get ahead. If it takes 270us to dispatch and then N us to prefetch:
- Layer N matvec: 0-758us
- GPU dispatch for layer N+2: starts at ~0us, shader runs at ~270us
- GPU prefetch completes: ~270us + prefetch_time
- Layer N+2 matvec starts: ~1516us (2 layers later)

The prefetch has ~1246us of headroom. At 12.8 GB/s (half of max bandwidth), we could touch 16 MB. A layer is 2.65 MB.

Why it matters: Timing seems feasible IF bandwidth sharing works.

## Node 6: The Measurement Problem
How do we know if this works? Candidates:
- Matvec throughput (tok/s) — but variance is high
- Matvec timing variance — row misses cause jitter
- Memory controller performance counters — if accessible
- Power consumption — fewer row activations = less power
- Cache miss counters — perf events might be accessible

Tension with Node 3: If we can't observe memory controller behavior, we're tuning blind.

Why it matters: Without measurement, we're cargo-culting.

## Node 7: The External Memory Import
Vulkan supports importing external memory:
- VK_EXT_external_memory_dma_buf
- VK_ANDROID_external_memory_android_hardware_buffer

The GGUF file can be mmap'd by CPU (llama.cpp already does this) and the same pages can be imported into Vulkan as a buffer. No copying needed.

Why it matters: The mechanism exists and is proven. This is not speculative.

## Node 8: The Row Buffer Size Constraint
DRAM row buffers are typically 8KB per bank. LPDDR4X has 8 banks per channel, 2 channels. That's:
- 8 KB × 8 banks × 2 channels = 128 KB total row buffer capacity

A weight tensor is 2.65 MB = 20.7× the row buffer capacity. Even with perfect prefetching, we can only keep a small fraction of the weights "warm" at any time.

Tension with Node 5: We can prefetch a lot of data, but the row buffers can only hold a little. Prefetching too far ahead might mean the rows close before CPU gets there.

Why it matters: The optimal prefetch distance is bounded by row buffer policy, not by bandwidth.

## Node 9: The Cache Hierarchy Question
Where does GPU-touched data land?
- DRAM row buffer (row open, but no cache)
- Shared L3 cache (if MT6855 has one)
- GPU's internal cache (useless for CPU)
- Nowhere useful (just opens the row)

The Dimensity 930 (MT6855) cache hierarchy:
- A78 cores: 64KB L1I, 64KB L1D, 512KB L2 per core
- A55 cores: 32KB L1I, 32KB L1D, 128KB L2 per core
- Shared L3: Unknown size, if any

If there's no shared L3, GPU reads don't help CPU caches at all.

Why it matters: GPU touching data might only help at the DRAM level, not the cache level.

## Node 10: The Interference Probe First
Before building anything complex, we need to measure:
1. CPU matvec bandwidth with GPU idle
2. CPU matvec bandwidth with GPU doing sustained reads
3. The delta tells us if GPU access is "free" or costs CPU performance

This is a much simpler experiment than a full prefetch system. If the delta is large negative, we stop. If it's small or positive, we continue.

Why it matters: This is the falsification test. Run it first.

## Node 11: The "What If We're Wrong" Exit
If GPU memory access hurts CPU performance, we should know this in 30 minutes of probing. The exit criteria:
- If CPU matvec slows >5% with GPU active: abandon
- If CPU matvec unchanged or faster: continue
- If measurement noise is too high: need better methodology

Tension with Node 6: We need a measurement methodology robust to noise.

Why it matters: Clear exit criteria prevent sunk cost fallacy.

## Node 12: Alternative Framing — GPU as Memory Pressure Relief
Instead of prefetch, what if GPU does the opposite: it handles cold/background memory operations so CPU can focus on hot data?
- GPU loads the NEXT model while CPU runs the CURRENT model
- GPU evicts old KV cache entries while CPU generates new ones
- GPU manages swap-out to the UFS while CPU does inference

This reframes GPU as "memory janitor" rather than "memory accelerator."

Tension with all above: This is a different problem entirely. But it might be more tractable.

Why it matters: If the prefetch idea fails, this is the pivot.
