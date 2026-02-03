# Nodes of Interest: GPU as CfC Swarm Coprocessor + Speech/Diffusion Fabric Layer

## Node 1: The Dispatch Tax is Fixed, Not Proportional

The 270us mean GPU dispatch overhead is a fixed cost regardless of what the shader does. NOP = 270us. SiLU on 1024 elements = 297us. The compute is ~27us, the envelope is ~270us. This means the GPU becomes viable only when: (a) the operation takes long enough to amortize 270us, or (b) you batch enough work into one dispatch to amortize it. At 1000 CfC steps batched into one dispatch: 270us / 1000 = 0.27us per step. That beats CPU's 35ns? No — 0.27us = 270ns, still 8x slower than CPU. But the point isn't speed, it's FREE — the CPU is busy with inference.

Why it matters: This is the fundamental economics of every GPU use case. Any proposal must show either "big enough operation" or "enough batched work" or "concurrent with CPU so cost is zero."

## Node 2: The CfC Swarm Doesn't Need to Be Fast, It Needs to Be Free

The current fabric CfC cell takes 9.4us at 100Hz = 0.094% of CPU time. Irrelevant. Even 128 cells = 4.5us = still irrelevant. The GPU can't compete on latency for CfC. But if the CPU is saturated doing inference (both A78 cores at 100% on matvec), CfC cells on CPU steal cycles. On GPU, they're free — coprocessor v2 proved GPU compute during CPU matvec has zero DRAM contention.

Tension with Node 1: The dispatch tax (80us at p50 on CPU matvec time) is NOT free. It costs the CPU ~11% of a matvec to submit GPU work. Is 128 CfC cells worth 80us of CPU tax?

## Node 3: Multi-Model is the Real Product Expansion

LFM2-350M (209MB) + Whisper-tiny (75MB) + Kokoro TTS (80MB) = 364MB. Fits in 3.6GB with room. This is a voice-in, text-reasoning, voice-out pipeline on a $200 phone. The fabric managing all three models — scheduling, memory mapping, model hot-swap — is a more compelling product than any GPU compute trick.

Why it matters: This shifts the question from "what can the GPU compute?" to "what does the fabric need to orchestrate?" The GPU might have a role not in compute but in the orchestration intelligence.

## Node 4: Speech Has Operations That Fit the GPU's Sweet Spot

Mel spectrogram: 80 bins × N frames, each bin's FFT is independent. Vocoder: 1D conv with large receptive field, each output sample independent. These are wide, parallel, compute-bound operations that could saturate 256 ALUs. Key: they happen BEFORE/AFTER the LLM runs (in a pipeline), not concurrently. So bus contention doesn't matter — the CPU isn't doing matvec during speech processing.

Tension with Node 1: But dispatch overhead still applies. If mel spectrogram on CPU takes 500us, GPU dispatch alone is 270us + compute. Only wins if GPU compute is faster than CPU compute by more than the dispatch overhead.

## Node 5: TMU for Hardware-Accelerated Activation LUTs is Unexploited

The PowerVR has a Texture Mapping Unit that does hardware bilinear interpolation on R16_SFLOAT textures. Yinsen's software LUT+lerp for sigmoid/tanh is exactly what TMU does in hardware. For large buffers (4608 elements in FFN, or 16K+ in diffusion latents), TMU could process activations faster than CPU — the interpolation is free in silicon.

Why it matters: This is a genuinely unique hardware capability. No amount of CPU optimization can replicate hardware interpolation throughput. We measured TMU in early probes but never tested it for activation functions on real-sized buffers.

Tension with Node 1: Still subject to dispatch overhead. And CPU SiLU on 4608 elements = 25.6us. TMU would need to beat that by enough to offset 270us dispatch. Only viable for very large buffers or batched operations.

## Node 6: The "Persistent Observer" Pattern Avoids Dispatch Tax

Instead of dispatching GPU work per inference step (16× per token, 270us × 16 = 4320us = impossible), what if the GPU runs at a low fixed rate? Like the CfC daemon's 10-100Hz. One dispatch every 10-100ms. The GPU shader reads system state from shared memory (written by CPU), runs a swarm of CfC cells, writes decisions back. The 270us dispatch is 2.7% at 10Hz, 27% at 100Hz.

Why it matters: This decouples GPU dispatch rate from inference step rate. The GPU becomes a slow-but-free background intelligence engine.

## Node 7: The Ternary Matvec on GPU is Architecturally Mismatched

Yinsen's sparse ternary matvec (sentinel-terminated index lists, adds/subtracts only) is inherently sequential per row — you walk an unknown-length index list. GPU threads want uniform control flow. Thread divergence (different cells have different sparsity patterns, different sentinel positions) kills GPU utilization. The SME kernel uses predicate switchboard (all 16 lanes uniform) which maps to SIMD, not SIMT.

Tension with Node 2: If we want CfC on GPU, we might need to use the dense trit-packed format instead of sparse-index, even though it does more work, because it has uniform control flow.

## Node 8: Diffusion Latent Space Fits in Shared Memory

Typical latent: 64×64×4 = 16,384 floats = 64KB. Doesn't fit in 16KB shared memory. But at the per-channel level: 64×64 = 4096 floats = 16KB. One channel fits exactly. Process 4 channels sequentially = 4 dispatches. Each dispatch does all the activation/normalization for one channel in shared memory — no DRAM traffic.

But: diffusion models do matmul between channels (cross-attention, conv1x1). Those can't stay on-chip.

Why it matters: Partial on-chip processing of diffusion latents is possible but requires careful operation partitioning.

## Node 9: The Fabric's Real GPU Role May Be "Background Intelligence"

Combining Nodes 2, 3, 6: The GPU doesn't do inference work at all. It runs a persistent low-rate background process that observes and adapts. Examples: thermal prediction model (predict throttling before it happens), memory pressure predictor (when to evict a model), attention quality monitor (detect when LLM is generating nonsense), audio VAD (voice activity detection while LLM generates). These are all sub-1ms workloads that only need to run at 10Hz.

This is philosophically aligned with the fabric vision: "the fabric IS the hardware layer, applications just see faster hardware." The GPU becomes part of the fabric's brain, not part of the inference pipeline.

## Node 10: We Have Zero Real Benchmarks for Speech/Diffusion on This Device

Everything above is theoretical. We don't know:
- How fast whisper.cpp runs on this device
- Whether mel spectrogram is a bottleneck at all
- What the actual operation profile of a tiny diffusion model looks like
- Whether 3.6GB RAM is enough for concurrent models + OS overhead

Before designing a GPU role in speech/diffusion, we need baseline numbers from actual speech/diffusion workloads on actual hardware.

Tension with all other nodes: We might be designing solutions for problems that don't exist on this hardware.

## Node 11: The Yinsen 16x16 Tile Maps to GPU Subgroup Perfectly (Almost)

GPU subgroup = 128 threads. A 16×16 tile = 256 elements. Two tiles per subgroup. Or: each thread processes 2 elements of one tile. The trit-packed format (4 trits per byte, 16 trits per uint32) could map to a GPU kernel: each thread unpacks and processes 2 weight rows, accumulating into shared memory. No thread divergence if using dense trit format.

BUT: subgroup=128 is already the ENTIRE workgroup. There's no parallelism beyond one tile unless we dispatch multiple workgroups. And each workgroup has 16KB shared memory — exactly one 16×16 tile of floats (256 × 4 = 1KB) plus activations.

## Node 12: The CPU-GPU Communication Channel Already Exists

Unified memory means the CPU and GPU share the same address space. The fabric daemon already writes to shared buffers. A "GPU observer" shader could read from the same system state buffer the CfC daemon reads from (CPU temps, frequencies, memory pressure) and write decisions to a decision buffer the daemon reads. Zero-copy, no transfer. The 16KB shared memory is just a fast scratchpad — input/output live in DRAM.

Why it matters: No architectural work needed for CPU-GPU communication. It's already unified.
