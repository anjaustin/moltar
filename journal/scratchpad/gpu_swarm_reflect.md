# Reflections: GPU as CfC Swarm Coprocessor + Speech/Diffusion Fabric Layer

## Core Insight

There are two completely separate questions hiding in one conversation, and conflating them has been obscuring the real structure:

**Question A**: Can the GPU contribute to speech/diffusion inference compute?
**Question B**: Can the GPU serve as the fabric's intelligence substrate — a brain separate from the inference pipeline?

These have different answers and different implications.

## Resolving Question A: GPU for Speech/Diffusion Inference

Nodes 1, 4, 5, 8, 10 are all about this question. Let me trace the logic:

The GPU's dispatch overhead (270us) is fixed. For the GPU to contribute to inference, operations must be:
- Large enough that GPU compute < CPU compute + 270us dispatch savings
- Or: running while CPU is idle (pipeline overlap)

In the LLM case, operations are tiny (25-758us) and sequential. GPU can't help.

In speech/diffusion, operations are potentially larger:
- Mel spectrogram of 1s audio: ~80K multiplies per frame × 50 frames = 4M ops. On A78 NEON at ~8 GFLOPS: ~500us. GPU dispatch alone is 270us. The operation isn't big enough.
- Whisper encoder conv layer: 80×3000 input, 512 filters, kernel 3. That's 80×3000×512×3 = 369M MACs. On A78: ~46ms. On GPU with 256 ALUs at ~1GHz: ~1.4ms compute but memory-bound. The data (80×3000×4B = 960KB) exceeds shared memory (16KB). GPU must stream from DRAM. Bus contention returns.
- Diffusion U-Net matmul: varies by model but typically 512×512 or larger. Our probe showed GPU is 19x slower at matmul. No reason to think that changes at these sizes — the PowerVR's memory access pattern is fundamentally worse than CPU NEON for matrix operations.

**Verdict on Question A: No.** The same fundamental limitations that killed GPU for LLM inference — dispatch overhead for small ops, memory bandwidth for large ops — apply to speech/diffusion. The PowerVR BXM-8-256 is simply too weak and too DRAM-limited for any inference compute on this SoC.

The one exception might be if speech/diffusion processing happens when the CPU is completely idle (not running LLM). In that case there's no bus contention. But even then, the GPU is 19x slower at matmul and the dispatch overhead is still 270us. CPU wins unconditionally for any operation we can define.

**We should kill Question A permanently.** No more GPU-for-inference investigations on this hardware class.

## Resolving Question B: GPU as Fabric Brain

Nodes 2, 6, 7, 9, 11, 12 are about this. The synthesis is more nuanced:

**The "Persistent Observer" pattern (Node 6) resolves the dispatch tax problem.** At 10Hz, the 270us dispatch is 0.27% of 100ms. Trivial. The GPU runs a shader once per 100ms that:
1. Reads system state from unified memory (CPU temperatures, memory pressure, inference rate, model state)
2. Processes this through a swarm of CfC cells
3. Writes decisions back to unified memory (scheduling hints, thermal predictions, anomaly flags)

The CPU daemon reads these decisions on its next tick. Zero-copy, unified memory, no bus contention (GPU only touches a few KB of state data, not model weights).

**But Node 2's tension is real**: Does 128 CfC cells on GPU at 10Hz provide value that 128 CfC cells on CPU at 10Hz doesn't? The CPU cost is 128 × 35ns = 4.5us — literally nothing. The GPU dispatch cost is 270us — 60x more. The GPU version is strictly worse in every measurable dimension UNLESS the CPU cores are genuinely saturated.

**When are the CPU cores saturated?** During inference generation at -t 2, both A78 cores are at 100% doing matvec. The daemon currently runs at adaptive 10-100Hz in the gaps between inference steps. A gap exists: the CPU does ~24 tokens/second at 44 tok/s total, so each token takes ~23ms. Between tokens, there are pipeline stalls, memory fetches, scheduler operations. The daemon finds microsecond-scale gaps. 4.5us of CfC easily fits.

But in a multi-model scenario (Node 3), things change. LLM generating + Whisper encoding + TTS decoding, all on 2 A78 cores. The cores might be genuinely saturated with no gaps. Then the GPU as CfC coprocessor becomes valuable — it's the only silicon not doing inference work.

**This is the key insight: the GPU's value scales with CPU saturation, which scales with multi-model load.**

## Resolving Node 7: Ternary on GPU

The sparse-index format (sentinel-terminated lists) is terrible for GPU. Thread divergence is catastrophic on a 128-wide subgroup.

The dense trit-packed format (2 bits per weight, 4 per byte) is GPU-friendly: every thread processes the same number of bytes, same control flow. The cost: you process zeros too (80% of weights are zero). But the "processing" of a zero is just a branch-free zero-multiply, and on GPU with 128 threads doing it in lockstep, it's 128 trits evaluated simultaneously.

For a CfC cell with hidden_dim=8, concat_dim=10: the gate matvec is 8×10 = 80 trits = 20 bytes packed. One thread could process the entire matrix. 128 threads = 128 independent cells in one dispatch. The matvec becomes: for each of 8 rows, unpack 10 trits (3 bytes), multiply-by-sign-or-skip, accumulate. ~80 ops per cell.

**This is workable.** Dense trit format, 128 cells per dispatch, uniform control flow. Not faster than CPU per-cell, but free during CPU inference.

## Resolving Node 10: We Need Baselines

Node 10 is the honest node. We're designing around models we've never run on this device. Before any GPU swarm design, we should:
1. Run whisper.cpp on the device — get real STT numbers
2. Run a tiny TTS — or at least measure what the TTS pipeline looks like
3. Understand the real multi-model memory picture

This doesn't invalidate the GPU brain concept, but it grounds it.

## Resolved Tensions

- **Node 1 vs Node 6**: Dispatch tax is only a problem at high dispatch rates. At 10Hz, it's noise. RESOLVED: Use low-rate persistent observer pattern.
- **Node 2 vs Node 9**: CfC on GPU isn't needed for single-model fabric. It IS needed for multi-model saturated fabric. RESOLVED: GPU brain activates under multi-model load, CPU handles it otherwise.
- **Node 4 vs Node 10**: Speech operations might fit GPU sweet spot in theory. We have no numbers. RESOLVED: Probe first, design second. But Question A analysis suggests GPU loses unconditionally.
- **Node 7 vs Node 11**: Sparse index format bad for GPU, but dense trit format maps well to subgroup. RESOLVED: Use dense trit packing for GPU CfC cells.

## What I Now Understand

The GPU on this device has exactly one viable role: **background intelligence under CPU saturation.** Not inference compute. Not activation acceleration. Not matmul assist. A slow, cheap, always-available substrate for running the fabric's neural controller swarm when the CPU cores are fully occupied with multi-model inference.

The architecture is:
1. CPU runs 1-3 inference engines (LLM + STT + TTS) on both A78 cores
2. GPU runs a CfC swarm at 10Hz via timeline semaphore dispatch
3. CfC swarm observes system state (temps, rates, memory) through unified memory
4. CfC swarm writes control signals (scheduling, thermal, quality) back through unified memory
5. CPU daemon reads GPU control signals when it can, uses them for scheduling decisions

This is a narrow but genuine role. It's also perfectly aligned with the fabric philosophy: the GPU isn't an inference engine, it's part of the hardware optimization layer. Applications never know it exists.

## Remaining Questions

1. Is multi-model CPU saturation actually a real scenario, or do the models take turns?
2. What's the right CfC swarm size for useful fabric intelligence? 16? 64? 128?
3. Should we probe the multi-model scenario first (get whisper.cpp running) or build the GPU CfC swarm shader first?
4. Is there a TMU angle worth probing separately — not for inference, but for the CfC swarm's activation functions?

## What Would This Look Like If It Were Easy?

The simplest version: forget the GPU entirely. Run everything on CPU. Add whisper.cpp support to the fabric. The daemon manages memory, scheduling, and model switching. The CfC cells run on CPU in the gaps. Ship it.

The GPU CfC swarm is an optimization for the case where "gaps" disappear under multi-model load. Build the CPU version first. Add GPU as a fallback intelligence substrate later if needed.

**This is probably the right answer.**
