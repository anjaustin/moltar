# Reflections: W4A8 Integer Path Optimization

## Core Insight

**We are at 92% of theoretical DRAM bandwidth utilization. The axe is already sharp.**

The profiling revealed something profound: at ~17ms/tok with 204MB of weights, we're achieving 12 GB/s effective throughput against a 13 GB/s theoretical peak. KleidiAI + our 2-thread big-core configuration has extracted nearly all available performance from this hardware.

The question "can we use integer math to skip multiplies" was the wrong question. The SDOT instruction makes integer multiplies essentially free — 4 MACs per cycle. The bottleneck is **memory bandwidth**, not compute.

## Resolved Tensions

### Node 1 vs Node 7: Multiplies vs Sparsity
The Yinsen ternary kernel's 2.2x speedup isn't about avoiding multiplies — it's about **skipping loads for zero weights**. With 12% zeros in Q4_0, we'd save 12% of bandwidth. But the ternary kernel needs structured sparsity (entire rows/columns of zeros), not scattered zeros. The model wasn't trained for this.

**Resolution:** Yinsen's advantage awaits natively sparse models. For LFM2, we're bandwidth-bound with no sparsity to exploit.

### Node 2 vs Node 3: Quantization Inside KleidiAI
The 0.02% F32→Q8 time in the standard path is misleading — KleidiAI's `pack_func_ex` does the same quantization, just fused with the kernel. The 76.75% includes both packing AND compute, but since we're at 92% bandwidth utilization, most of that time is memory stalls anyway.

**Resolution:** Activation quantization isn't the bottleneck. It's baked into the kernel and amortized with compute.

### Node 5 vs Node 2: RMS_NORM Float Requirement
We can't keep a pure int8 pipeline because RMS_NORM needs float precision. But this doesn't matter — the F32→int8 conversion is so cheap (0.02%) that keeping activations in float between layers is effectively free.

**Resolution:** The "W4A8" framing assumed activation quantization was expensive. It isn't.

### Node 10 vs Node 4: Bandwidth Math
At 17ms/tok and 204MB weights: 204MB / 0.017s = 12 GB/s. 
At 13 GB/s theoretical: 92% utilization.
This is remarkably close to optimal for a memory-streaming workload.

**Resolution:** We're at the wall. Further optimization requires either smaller weights (lower quant) or smarter access patterns (speculative decoding).

## The Laundry Method Applied

**The pile:** All possible optimizations for LFM2 inference
**Coarse buckets:**
1. Compute optimizations (ALU, SIMD, kernel fusion)
2. Memory optimizations (bandwidth, caching, prefetch)  
3. Algorithmic optimizations (speculative decoding, pruning)
4. Model optimizations (quantization, sparsity)

**Narrowing within:**
- Bucket 1 (Compute): CLOSED — we're memory-bound, not compute-bound
- Bucket 2 (Memory): CLOSED — at 92% bandwidth utilization
- Bucket 3 (Algorithmic): OPEN — speculative decoding not explored
- Bucket 4 (Model): PARTIALLY OPEN — lower quant possible but requires retraining/reconversion

**The delta (boundary items):**
- Speculative decoding sits between Algorithm and Memory — it changes access pattern
- IQ2/IQ3 sits between Model and Memory — smaller weights = less bandwidth

## What I Now Understand

1. **The integer multiply question was a red herring.** SDOT makes int8 multiplies free. The bottleneck is fetching operands from DRAM, not computing on them.

2. **KleidiAI is near-optimal.** Arm's engineers have already solved the Q4_0 × F32 kernel. We can't beat them at their own game on their own hardware.

3. **W4A8 vs W4A32 is a non-distinction.** Activation quantization happens inside KleidiAI's pack function, fused with the kernel. The "A32" activations are converted to int8 on-the-fly with negligible overhead.

4. **The only levers left are:**
   - **Smaller quantization (IQ2/IQ3):** Reduces bytes loaded per weight. Requires model reconversion and quality testing.
   - **Speculative decoding:** Amortizes weight loads across multiple tokens. Requires a draft model or self-speculative approach.
   - **Native sparse models:** If Liquid AI releases ternary models, Yinsen wins. Until then, we wait.

5. **The fabric daemon solved the right problem.** Core pinning and memory prefaulting were system-level wins. The kernel-level work is done by KleidiAI.

## Remaining Questions

1. **Can IQ2/IQ3 run on KleidiAI?** Need to check if llama.cpp's KleidiAI integration supports sub-4-bit quants.

2. **Speculative decoding with self-draft:** Can LFM2-350M draft for itself with early exit? Some layers predict well before others.

3. **LFM2-350M drafting for 1.2B:** What's the acceptance rate? If >70%, we could 2-3x the effective throughput.

4. **Liquid AI's roadmap:** Any plans for sparse/ternary models?

## The Honest Answer

To Tripp's original question about integer math avoiding multiplies:

**The multiplies are already "avoided" in the sense that matters — SDOT makes them free. The actual bottleneck is DRAM bandwidth at 92% utilization. We can't make bits flow faster through the memory bus by changing arithmetic.**

The Yinsen insight about ternary {-1, 0, +1} weights remains valid but UNSATISFIED — it needs natively trained sparse models. Post-hoc ternarization destroys quality.

The path forward is:
1. **Speculative decoding** — software solution to bandwidth bottleneck
2. **Smaller quantization** — less data to transfer
3. **Wait for sparse models** — then Yinsen dominates

We are not leaving performance on the table. We are at the table's edge.
