# Nodes of Interest: GPU Verdict on Dimensity 930

## Node 1: The Dispatch Wall
GPU dispatch latency (160-330us) is 6-13x more expensive than the CPU operations it would replace (SwiGLU=25us, RMSNorm=1us). This isn't a software problem — it's the fundamental cost of crossing the CPU-GPU boundary on a low-power SoC. Timeline semaphores reduced it from 386us but can't eliminate it.
Why it matters: Even perfect overlap can't hide the dispatch if the operation is cheaper than the dispatch itself.

## Node 2: The Dependency Chain Trap
SwiGLU sits between gate/up matvecs (input) and down matvec (output). It's on the critical path. You can't overlap it with the matvecs it depends on or that depend on it. Cross-layer overlap is blocked by the residual connection: layer N+1's input needs layer N's full output. The only "free" overlap is across independent operations that don't exist in this architecture.
Why it matters: Probe B's "93.9% hidden" was an illusion — it tested overlap with a SYNTHETIC CPU workload, not the actual dependency graph.

## Node 3: The 97/3 Split
Matvecs = 97.2% of FFN layer time. Activations = 1.7%. Quantization = 1.1%. RMSNorm = 0.05%. Optimizing the 3% can never yield more than 3% improvement. We're deep in Amdahl's law territory.
Why it matters: The only way to improve throughput is to speed up matvecs or reduce their count.

## Node 4: Memory Bandwidth is the Real Bottleneck
CPU matvec at 758us for 2.65MB tensor = 3.5 GB/s per core. RAID 0 with 2 cores = 6.6 GB/s. LPDDR4X theoretical max = 13 GB/s. We're at 51% bandwidth utilization. The gap could be cache misses, bank conflicts, or the fact that matvec interleaves compute with loads.
Why it matters: If we could improve bandwidth utilization from 51% to 75%, that's a ~47% speedup on matvecs, which would be massive.
Tension with Node 3: bandwidth optimization would attack the 97%, unlike GPU which attacked the 3%.

## Node 5: The Fabric Already Works
v0.4 daemon: 44.0 tok/s. Bare taskset: 44.1 tok/s. The fabric captures 99.8% of theoretical maximum. Core pinning + memory prefaulting + adaptive CfC tick rate = mission accomplished for the daemon's original purpose.
Why it matters: Diminishing returns on daemon optimization. The wins are elsewhere.

## Node 6: Probe B Was Honest but Misleading
Probe B asked: "if CPU and GPU run simultaneously, does the GPU finish within the CPU's matvec time?" Answer: yes. But the right question was: "in the actual FFN pipeline, is there any operation we can move to GPU that isn't on the critical path?" Answer: no. The probe measured capability, not opportunity.
Why it matters: This is a lesson about probe design — always test in the context of the real computation graph, not in isolation.

## Node 7: The ShortConv Opportunity
LFM2 has 10 ShortConv layers. We've only probed FFN (attention+FFN). ShortConv is a 1D convolution with small kernels — different compute pattern. If ShortConv has more activation overhead relative to its matvecs, the GPU calculus might differ.
Why it matters: Untested territory. We assumed FFN represents all layers, but 10/16 layers are ShortConv.

## Node 8: Weight Reordering / Cache-Aware Tiling
The L2 probes showed a bandwidth cliff at 256KB. Q4_0 weight rows for [4608×1024] are 576 bytes each. The full tensor is 2.65MB = 10.3x L2 capacity. We're streaming from DRAM. Could reordering weights for cache-line alignment or tiling the computation to keep hot rows in L2 help?
Why it matters: This attacks the 97% (Node 3) via bandwidth (Node 4), potentially large win.
Tension: KleidiAI already does this internally. Would custom weight layout conflict with llama.cpp's expectations?

## Node 9: The CfC Controller's Unrealized Potential
The CfC neural controller runs at 10-100Hz, observing system state and making scheduling decisions. Currently it does initial core selection and adaptive rate. But it could potentially learn: thermal throttling patterns, memory pressure prediction, dynamic thread count adjustment (1 vs 2 threads based on system load), or even model-adaptive scheduling.
Why it matters: The daemon's unique value proposition is adaptive hardware optimization — we've only scratched the surface.

## Node 10: Sustained Performance vs Peak
44 tok/s is the peak under ideal conditions. What happens under sustained load? Thermal throttling? Background processes? Memory pressure from other apps? The daemon's CfC controller was designed for exactly this — maintaining performance under varying conditions.
Why it matters: Real-world performance may be significantly lower than lab benchmarks. The daemon's value might show up more in the delta between sustained and peak, not in peak itself.

## Node 11: The Larger Model Gap
LFM2-700M: 20.36 tok/s. LFM2-1.2B: 14.85 tok/s. These models are more memory-bandwidth-bound (larger weight tensors, same DRAM bandwidth). The fabric's memory management (pre-fault, mlock, madvise) might matter more for these. Untested: does the CfC controller help more with larger models?
Why it matters: The market for 350M models is narrow. 700M/1.2B are more practical. The fabric's value might be higher there.
