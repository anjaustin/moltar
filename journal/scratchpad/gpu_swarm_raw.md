# Raw Thoughts: GPU as CfC Swarm Coprocessor + Speech/Diffusion Fabric Layer

## Stream of Consciousness

The GPU investigation for LLM inference is dead. We've proven it six ways. But Tripp asked the killer question: what about speech and diffusion models? And then the second question: could Yinsen's tiny ternary matvecs run on the GPU?

My gut says the combination of these two questions is where the real answer lives. Not "can the GPU do speech inference" (probably not well — same dispatch overhead, same memory bandwidth limits). But "can the GPU run a swarm of CfC cells as a parallel intelligence layer while the CPU does the heavy inference?"

Let me dump what I actually know:

**What the GPU is good at (proven)**:
- 128 threads running in perfect lockstep (subgroup=128)
- 16KB shared memory — fast, private, no bus contention
- Hardware texture interpolation (TMU) — never actually exploited
- Doing work that's hidden behind CPU matvec (coprocessor v2 showed GPU finishes in 0.8us median extra wait)
- The compute itself doesn't steal DRAM bandwidth — only the dispatch does

**What the GPU is bad at (proven)**:
- Anything touching DRAM during CPU matvec (bus contention)
- Single dispatches < 1ms (270us overhead dominates)
- Matmul at any size we tested (19x slower)
- Persistent spin-wait (driver watchdog)

**What speech models need**:
- STT (Whisper-tiny): encoder is heavy CNN + transformer, decoder is autoregressive
- TTS (Bark/VITS/Kokoro): typically vocoder (1D conv) + acoustic model
- Mel spectrograms: 2D FFT, 80 bins, overlapping windows
- Real-time constraint: 16kHz audio = 50 frames/sec at 320-sample hop

**What diffusion models need**:
- U-Net with residual blocks, cross-attention, downsampling/upsampling
- Multiple denoising steps (4-50 depending on model)
- Latent space typically 64x64x4 = 16K elements (fits in shared memory!)
- Each step processes full latent — embarrassingly parallel within a step
- Tiny diffusion models exist: Stable Diffusion Mobile, LCM-1.5 pruned

**What Yinsen CfC cells look like on GPU**:
- Sentinel cell: hidden_dim=8, concat_dim=10. Two 8x10 matvecs (gate+candidate), 2 activations, 1 ODE step.
- Sparse representation: ~2 non-zero weights per row at 81% sparsity. So each matvec is ~16 adds.
- One CfC step on CPU: 35ns (float) or 20ns (sparse). Absurdly fast.
- 128 independent cells on GPU: each thread does ~32 adds + 2 LUT lookups + ODE blending. Maybe 50 ops per thread. At ~1GHz effective GPU clock, that's ~50ns of compute inside a 270us dispatch envelope.
- The compute-to-dispatch ratio is terrible for a single invocation.
- BUT: what if you submit once and the shader does 1000 steps? 50ns × 1000 = 50us of compute inside 270us dispatch. Still bad.

Wait. I'm thinking about this wrong. The question isn't "is the GPU faster than CPU for CfC cells?" The CPU will always win per-cell. The question is "can the GPU do CfC work FOR FREE while the CPU is busy with inference?"

From coprocessor v2: the GPU completing SiLU during a CPU matvec adds only 80us at p50 (all dispatch overhead, zero compute contention). So 128 CfC cells running on GPU while CPU does a 758us matvec is effectively free — the CfC work happens in the dispatch shadow.

**But do we need 128 CfC cells?**

This is the real question. In the current fabric, we have ONE CfC cell running at 10-100Hz on the CPU. It takes 9.4us per step. That's nothing. Why would we need 128 of them?

Possible reasons:
1. Multi-modal pipeline: LLM + speech + vision, each needs its own controller swarm
2. Per-layer monitoring: 16 CfC cells watching 16 transformer layers
3. Per-head attention quality tracking: 16 heads × quality metric
4. Audio processing: 80 mel bins × temporal tracker
5. Diffusion step adaptation: per-spatial-region denoising scheduler
6. Thermal/power model: many-zone thermal prediction

But honestly, even 128 cells at 35ns each = 4.5us on CPU. That's trivial. The GPU doesn't save meaningful CPU time.

**So when DOES the GPU help?**

Maybe the answer isn't about CfC cells at all. Maybe it's about the speech/diffusion workloads themselves. Let me think about what operations in speech/diffusion are:
1. Large enough to amortize 270us dispatch
2. Parallel enough for 256 ALUs
3. Not memory-bandwidth-bound (or can use shared memory)

Candidates:
- Mel spectrogram: FFT of 80 bins × N frames. FFT is compute-heavy, parallel. Each bin's FFT is independent. 80 bins = not enough for 128 threads. But N frames × 80 bins could be 2400+ elements.
- Vocoder convolution: 1D conv with large kernel (K=7 to K=1024). Each output sample is independent. A 16K-sample output block would saturate 128 threads.
- Diffusion latent processing: 16K elements per step. SiLU, GroupNorm, attention QKV. Each fits in shared memory.
- Attention score computation in diffusion: matmul of Q×K^T where Q,K might be 64×64 per head. That's 4096 elements — GPU could help if there are many heads.

Actually wait — the 3.6GB RAM constraint is HUGE. Can we even fit a speech or diffusion model alongside the LLM?

- LFM2-350M Q4_0: 209 MB
- Whisper-tiny: ~75 MB (39M params, FP16)
- Whisper-small: ~488 MB
- LCM-1.5 pruned to 4-step: ~1.7 GB at FP16, ~900 MB at INT8
- Bark-small: ~800 MB
- VITS: ~80 MB
- Kokoro: ~80 MB (tiny TTS)

So: LFM2-350M + Whisper-tiny + Kokoro = 209 + 75 + 80 = 364 MB. Fits easily.
LFM2-350M + Whisper-small = 700 MB. Fits.
LFM2-350M + LCM-1.5 INT8 = 1.1 GB. Fits but tight with runtime overhead.

Multi-model on this device is real. The fabric manages model switching, memory, scheduling.

**The TMU angle**:
We still haven't exploited the hardware texture interpolation unit. For activation functions (sigmoid, tanh, GELU, SiLU), a 256-entry 1D texture with LINEAR interpolation gives hardware-accelerated function approximation. This is exactly what Yinsen's LUT+lerp does in software. The GPU has it in hardware. For speech/diffusion where activations are applied to large buffers (16K+ elements), the TMU could be genuinely useful.

**What scares me**:
- We might be looking for a use for the GPU when there isn't one. The CPU is just really good.
- 3.6GB RAM means multi-model is possible but tight. The fabric's real value might be memory management, not GPU compute.
- We've never actually run a speech or diffusion model on this device. We don't know the real bottlenecks.
- The dispatch overhead might be fundamentally too high for any mobile workload at this GPU tier.

**What excites me**:
- The CfC swarm idea is genuinely novel. Nobody is running neural controllers as GPU shaders.
- TMU for activation LUTs is a hardware feature we haven't exploited.
- Multi-model orchestration (LLM + STT + TTS) is a real product. The fabric managing all three is compelling.
- The GPU running a persistent "observation shader" — not per-inference-step, but as a continuous background monitor — could work if dispatch rate is low enough (1-10Hz).

## Questions Arising

- What's the minimum useful CfC swarm size that justifies GPU dispatch overhead?
- Can we batch many CfC steps into one dispatch to amortize the 270us?
- What speech/diffusion operations actually fit in 16KB shared memory?
- Has anyone done TMU-based activation functions on mobile GPUs?
- What does "multi-model fabric" actually look like? Hot-swap? Concurrent? Pipeline?
- Is the real product "fabric for LLM + speech" rather than "GPU for anything"?
- Could the GPU do mel spectrogram computation while CPU runs the LLM decoder?

## First Instincts

- The CfC swarm on GPU is intellectually cool but probably not practically needed — CPU handles it fine
- The real GPU opportunity (if any) is mel spectrogram or vocoder convolution during LLM generation
- The TMU activation LUT idea deserves a probe regardless — it's hardware we own and haven't tested properly
- Multi-model fabric is the real product expansion, and it's mostly a CPU/memory management problem
- We should try running Whisper-tiny through llama.cpp's whisper support (or whisper.cpp) to get real baseline numbers before designing anything
