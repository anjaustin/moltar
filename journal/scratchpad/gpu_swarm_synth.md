# Synthesis: GPU CfC Swarm + Multi-Model Fabric

## Architecture

The LMM revealed two clear conclusions and one actionable design:

### Conclusion 1: GPU for Inference Compute is CLOSED (All Modalities)

The PowerVR BXM-8-256 on the Dimensity 930 cannot contribute to inference compute for ANY model type — LLM, speech, or diffusion. The reasons are hardware-fundamental:

| Limitation | Measured | Implication |
|-----------|----------|-------------|
| Dispatch overhead | 270us mean | Operations < 2ms can't amortize |
| Matmul throughput | 19x slower than CPU | No crossover at any matrix size |
| DRAM bandwidth | Shared bus, contention | Large ops compete with CPU |
| Shared memory | 16KB | Can't hold model layers |
| ALU count | 256 | ~1 GFLOPS vs A78's ~8 GFLOPS |

This applies equally to:
- LLM token generation (proven across 6 probe types)
- Speech encoder convolutions (memory-bound, exceeds shared memory)
- Diffusion U-Net matmuls (same matmul limitation)
- Vocoder processing (dispatch overhead exceeds operation time)
- Mel spectrograms (too small to amortize dispatch)

**No further GPU inference probes should be conducted on this hardware class.**

### Conclusion 2: GPU Has One Viable Role — Background Intelligence Under CPU Saturation

When the fabric runs multiple models concurrently (LLM + STT + TTS), both A78 cores may be fully saturated with inference work. In this scenario, the GPU becomes the only available silicon for running the fabric's neural controller (CfC cells).

The viable pattern is the **Persistent Observer**:
- Dispatch rate: 10Hz (one GPU submission per 100ms)
- Dispatch tax: 270us / 100ms = 0.27% — negligible
- Shader: reads system state from unified memory, runs CfC swarm, writes decisions
- Communication: zero-copy via unified memory buffers
- CfC format: dense trit-packed (uniform control flow for 128-wide subgroup)

This is a contingent value — only matters when CPU is saturated. CPU handles CfC trivially when it has spare cycles.

### Conclusion 3: Multi-Model Fabric is the Real Next Step

The compelling product expansion is not GPU compute but multi-model orchestration:

```
┌──────────────────────────────────────────────────────┐
│                  TriX Fabric v1.0                     │
│                                                       │
│  ┌─────────┐  ┌───────────┐  ┌─────────┐            │
│  │ LFM2    │  │ Whisper   │  │ Kokoro  │  Models    │
│  │ 350M    │  │ tiny      │  │ TTS     │            │
│  │ 209 MB  │  │ 75 MB     │  │ 80 MB   │            │
│  └────┬────┘  └─────┬─────┘  └────┬────┘            │
│       │             │              │                  │
│  ┌────┴─────────────┴──────────────┴────┐            │
│  │         Memory Manager               │  364 MB    │
│  │  mmap + mlock + madvise + model swap │  of 3.6GB  │
│  └──────────────────┬───────────────────┘            │
│                     │                                 │
│  ┌──────────────────┴───────────────────┐            │
│  │         Scheduler                     │            │
│  │  core pinning + governor + pipeline   │            │
│  │  CfC controller (CPU or GPU)          │            │
│  └──────────────────┬───────────────────┘            │
│                     │                                 │
│  ┌──────────────────┴───────────────────┐            │
│  │         Hardware Layer                │            │
│  │  2x A78 + 6x A55 + PowerVR + 3.6GB  │            │
│  └──────────────────────────────────────┘            │
│                                                       │
└──────────────────────────────────────────────────────┘
```

**Total model memory: 364 MB.** Fits comfortably. The fabric manages:
1. **Model lifecycle**: load, pin, unpin, swap. mmap for lazy loading, mlock for hot models.
2. **Pipeline scheduling**: STT encodes while LLM generates while TTS decodes. Time-slice on 2 A78 cores.
3. **Memory pressure**: predict when to evict cold model pages. CfC thermal model for throttle prediction.
4. **Quality monitoring**: detect LLM degeneration, audio silence, TTS stuttering.

## Key Decisions

### Decision 1: Build multi-model fabric on CPU first, GPU later (if ever)

The CPU handles everything. The CfC daemon already runs at 10-100Hz with 9.4us overhead. Even 128 cells would be 4.5us. Add multi-model awareness to the existing daemon rather than building GPU infrastructure.

**Why**: Simpler, proven, no Vulkan complexity. GPU adds value only under sustained multi-model saturation, which we haven't proven exists.

### Decision 2: Get real speech baselines before designing anything

Before expanding the fabric for multi-model:
1. Build whisper.cpp for this device
2. Measure STT performance (real-time factor)
3. Determine if speech+LLM concurrent is feasible on 2 A78 cores
4. Understand the actual memory picture with multiple models loaded

**Why**: Node 10 — we have zero real numbers. Everything else is speculation.

### Decision 3: Preserve GPU CfC swarm as a future option, don't build it now

The dense-trit-packed CfC swarm shader is a clean design that could be built in a day. But there's no evidence we need it today. Document the design, shelf it, build it when multi-model saturation is proven.

**Why**: YAGNI. The CPU handles CfC trivially. Build the GPU swarm when we have proof of CPU saturation.

### Decision 4: Close GPU investigation permanently for this hardware class

The GPU has been thoroughly characterized:
- Dispatch latency: measured (270us mean, 160us min)
- Matmul: measured (19x slower)
- Activation overlap: measured (free compute, 80us dispatch tax)
- Bus contention: measured (zero for on-chip, severe for DRAM)
- Persistent spin: measured (driver kills it)
- Coprocessor pattern: measured (viable but marginal)

No further GPU probes are justified unless: (a) a new SoC with different GPU is targeted, or (b) multi-model CPU saturation is proven and GPU CfC swarm is needed.

## Implementation Plan

### Phase Next-A: Speech Baseline (Priority 1)

1. Cross-compile whisper.cpp with Android NDK (similar to llama.cpp build)
2. Download whisper-tiny.en model (75MB)
3. Deploy to device, run with test audio
4. Measure: tokens/sec, real-time factor, memory usage
5. Run concurrent with LFM2-350M generation to measure contention

### Phase Next-B: Multi-Model Fabric Daemon v0.5 (Priority 2)

Extend `trix_fabric_daemon_q15.c` with:
1. Multi-model memory tracking (mmap file descriptors per model)
2. Pipeline scheduler: time-slice STT → LLM → TTS on A78 cores
3. Model-aware CfC: one cell per active model, tracks throughput/health
4. Memory pressure observer: predict swap needs

### Phase Next-C: GPU CfC Swarm (Priority 3 — contingent)

Only if Phase Next-A proves CPU saturation under multi-model load:
1. Build dense-trit CfC shader (128 cells, 10Hz dispatch)
2. Deploy as fabric daemon subprocess
3. Measure: does GPU intelligence improve scheduling under load?

### Phase Next-D: TTS Integration (Priority 4)

1. Evaluate Kokoro/VITS/Piper for Android
2. Build or cross-compile TTS engine
3. Integrate into fabric pipeline
4. Complete voice-in → reason → voice-out loop

## Success Criteria

- [ ] Whisper-tiny runs on device with measured real-time factor
- [ ] LFM2-350M + Whisper-tiny concurrent memory < 1GB
- [ ] Fabric daemon v0.5 manages 2+ models
- [ ] End-to-end voice → text → LLM → text → voice demo (stretch goal)
- [ ] GPU CfC swarm built and measured (only if CPU saturation proven)

## GPU CfC Swarm Shader Design (Preserved for Future)

For when/if this is needed:

```glsl
#version 450
layout(local_size_x = 128) in;

// Each thread = one CfC cell
// 128 cells per dispatch

layout(set=0, binding=0) buffer State {
    float system_state[32];     // shared input: temps, freqs, mem, rates
    float cell_hidden[128][8];  // per-cell hidden state (h_dim=8)
    float cell_output[128][4];  // per-cell output decisions
};

layout(set=0, binding=1) buffer Weights {
    uint gate_trits[128][3];    // packed 2-bit: 8×10=80 trits = 20B = 5 uint32
    uint cand_trits[128][3];
    float gate_bias[128][8];
    float cand_bias[128][8];
};

// ... ternary matvec + sigmoid LUT + ODE step per thread
```

128 cells × (8 hidden + 2 input) × 2 gates × 20 bytes = 5KB weights. Fits in shared memory with room to spare. State: 128 × 8 × 4B = 4KB hidden. Total: ~10KB. Under 16KB limit.

Dispatch at 10Hz via timeline semaphore. Total GPU time per second: 10 × 270us = 2.7ms. 0.27% of wall time.
