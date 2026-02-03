# Reflections: GPU Verdict on Dimensity 930

## Core Insight

**The GPU investigation was a success that produced a "no."** That's not failure — it's the probe-first methodology working exactly as designed. We spent ~4 hours of probe work to conclusively close a path that could have consumed weeks of misguided integration effort. The BXM-8-256 is architecturally wrong for this workload: high dispatch latency, shared memory bus, no independent bandwidth path. This isn't fixable with better shaders or cleverer pipelines.

## The Structural Truth

The LFM2 inference pipeline on Dimensity 930 is **entirely memory-bandwidth-limited**. Every operation reduces to "how fast can we stream bytes from DRAM to ALU?"

- Matvec: stream 2.65MB of weights → DRAM-limited
- SwiGLU: 4608 floats already in L1D from the matvec output → instant
- RMSNorm: 1024 floats in L1D → instant
- Quantization: 1024-4608 floats → instant

The GPU can't help because:
1. It shares the same DRAM bus (unified memory)
2. Its compute advantage only materializes with thousands of parallel operations AND independent memory paths
3. Its dispatch overhead exceeds the total cost of the operations we'd offload

**The GPU is a solution looking for a problem that doesn't exist on this device.**

## Resolved Tensions

### Node 1 (Dispatch Wall) vs Node 2 (Dependency Chain)
These reinforce each other. Even if dispatch were free (Node 1 resolved), the dependency chain (Node 2) means there's nowhere to put GPU work that isn't on the critical path. Both conditions would need to be resolved, and neither can be.

### Node 3 (97/3 Split) vs Node 4 (Bandwidth Bottleneck)
This is the key directional insight: **the only remaining optimization must attack the 97% (matvecs), and the only way to speed up matvecs is bandwidth.** Node 4 tells us there's a 49% gap between measured (6.6 GB/s RAID 0) and theoretical (13 GB/s). That gap is the optimization target.

### Node 5 (Fabric Works) vs Node 9 (CfC Potential)
The fabric works for *static* optimization. The CfC controller's value is in *dynamic* adaptation. These are different value propositions. Static: one-time setup, always active. Dynamic: responds to changing conditions, valuable over time.

### Node 7 (ShortConv) vs Node 3 (97/3 Split)
ShortConv layers are likely still dominated by matvecs (the linear projections around the conv). The 1D conv kernel itself might be relatively more expensive than SwiGLU, but it's still going to be dwarfed by the surrounding matvecs. Worth a quick probe but unlikely to change the conclusion.

## What I Now Understand

### The bandwidth gap is real and measurable
- Single A78: 3.5 GB/s (27% of theoretical)
- RAID 0 2×A78: 6.6 GB/s (51% of theoretical)
- Theoretical LPDDR4X: ~13 GB/s

The 49% gap between measured and theoretical could come from:
1. **Cache line waste**: Q4_0 blocks are 18 bytes, not power-of-2. Cache lines are 64 bytes. Partial line utilization.
2. **Bank conflicts**: LPDDR4X has multiple banks. Sequential access patterns may not spread across banks optimally.
3. **Refresh stalls**: DRAM refresh cycles steal bandwidth periodically.
4. **CPU-side stalls**: The matvec loop has data dependencies (scale multiply, accumulate) that may introduce bubbles in the load pipeline.
5. **Interleaved compute/load**: SDOT instructions and loads share issue slots. The CPU can't keep the memory bus 100% busy while also computing.

Reason 5 is probably dominant — and it means the theoretical 13 GB/s is never achievable for matvec. But going from 51% to 65-70% could be possible with better prefetching, loop restructuring, or weight layout optimization.

### Three viable directions remain

**Direction A: Bandwidth optimization (high effort, high potential)**
Attack the 49% bandwidth gap. Possible approaches:
- Software prefetch instructions (`prfm pldl1keep`) ahead of the load stream
- Weight layout reorganization (pad Q4_0 blocks to 32 bytes for cache-line alignment)
- Double-buffering: one register file decoding block N while loads fetch block N+2
- But: KleidiAI already does much of this. We'd need to understand exactly where they leave bandwidth on the table.

**Direction B: Sustained performance / thermal management (medium effort, medium potential)**
The daemon's real differentiator is maintaining performance under adverse conditions. Probing:
- How much does thermal throttling reduce tok/s after 60s, 120s, 300s of sustained generation?
- Can the CfC controller predict throttling and pre-emptively reduce thread count or adjust batch scheduling?
- Power/thermal characterization of the 3 models under sustained load

**Direction C: Multi-model intelligence (medium effort, unique value)**
The daemon knows system state. It could:
- Auto-select model size based on available RAM and thermal headroom
- Switch from 1.2B → 700M → 350M as device heats up
- Provide a "quality dial" that maps to model selection
- This is a PRODUCT feature, not just a benchmark improvement

## Remaining Questions

- What is the actual sustained tok/s after 5 minutes of continuous generation? (Never measured)
- Does KleidiAI use software prefetch? If not, could we add it via a ggml patch?
- What's the ShortConv kernel cost relative to its surrounding matvecs? (Quick probe needed)
- For the 1.2B model, does the daemon's memory management matter more? (Larger working set, more pressure)
