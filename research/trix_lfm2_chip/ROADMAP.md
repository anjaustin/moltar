# TriX LFM2 Enhancement Layer — Roadmap

## Where We Are

The TriX LFM2 frozen chip produces **correct logits**, matching llama.cpp's
top-5 predictions exactly. The complete LFM2-350M forward pass runs end-to-end
in standalone C with zero external dependencies (no llama.cpp, no GGML, no
Python). All 32 tests pass.

**Current performance (unoptimized scalar C, macOS host):** 3.7 tok/s
**Baseline on device (llama.cpp + KleidiAI):** 26 tok/s generation, 63 tok/s prompt eval
**PRD target:** 40 tok/s (1.5x), stretch 52 tok/s (2x)

---

## Phase 1: NEON-Optimized Q4_0 Matvec (Critical Path)

Q4_0 matvec is ~80% of all compute. This is where the speed comes from.

### 1a. ARM NEON Q4_0 Dot Product Kernel

Replace the scalar loop in `trix_matvec_q4_0()` with a NEON-intrinsic kernel:

- **Target ISA:** ARMv8.2-a + dotprod (Cortex-A78/A55 on Dimensity 930)
- **No i8mm** — device lacks it, do NOT use SDOT4x4 instructions
- **Key operations:** `vld1q_u8` (load 16 quants), `vshrn` / `vand` (nibble split),
  `vsubl` (subtract 8), `vdupq_n_f16` (broadcast scale), `vfmaq_f32` (FMA accumulate)
- **Reference:** Ghost-stream kernel in `~/Projects/trix/trix.research/yinsen/neon/neon_ternary.c`
  uses LDNP for weight bypass + 12-accumulator pipeline to hide load latency
- **Block processing:** Each Q4_0 block = 18 bytes. Process 4 blocks at a time
  (128 elements) to fill NEON pipelines. Use `vdotq_s32` for int8 dot product
  where possible.

### 1b. Cache-Aware Tiling

- A78 L1D = 64KB, L2 = 256KB. A55 L1D = 32KB, L2 = 128KB.
- Weight tile: ~14KB fits in L1 (256 Q4_0 blocks = 8192 elements = 4608 bytes weight + input)
- Tile the M dimension (output rows) in groups of 4-8 to amortize input vector loads

### 1c. Memory Bandwidth Optimization

- LPDDR4X-4266 theoretical: ~13 GB/s. Measured effective: ~8-10 GB/s.
- Q4_0 weight reads per token: ~190 MB (354M params * 4.5 bits avg)
- Theoretical minimum: 190MB / 13GB/s = 14.6 ms/tok = 68 tok/s
- Target: 40 tok/s = 25 ms/tok = 7.6 GB/s sustained read (58% of peak)

**Expected outcome:** 15-25 tok/s on device (up from 3.7 scalar)

---

## Phase 2: Spline Activation Integration

Spline tables are built and tested (10/10 pass). Wire them into the forward pass.

### 2a. Replace `expf`/sigmoid in SiLU

- SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
- Current: uses `expf()` from libm (~3ns per call)
- Spline sigmoid: ~2.1ns, worst error 3.35e-4 (within Q4_0 noise)
- Schraudolph fast-exp: ~1.5ns, ~4% relative error (use for softmax denominator)

### 2b. Replace `expf` in Softmax (Attention Layers)

- 6 attention layers, each does softmax over KV cache
- For single-token generation, softmax dimension = sequence position (grows with context)
- Use Schraudolph for exp() in softmax numerator, spline for the log-sum-exp stability trick

**Expected outcome:** 5-10% overall speedup (activations are ~10% of compute)

---

## Phase 3: Android Cross-Compilation & On-Device Deployment

### 3a. NDK Cross-Compile

```bash
NDK=~/Library/Android/sdk/ndk/28.2.13676358
$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android35-clang \
    -march=armv8.2-a+dotprod+fp16 -O2 -ffast-math \
    -I include/ -o trix_lfm2_arm64 \
    src/lfm2_trix.c src/gguf_loader.c src/trix_spline.c tests/run_lfm2.c -lm
```

### 3b. Deploy and Benchmark

```bash
adb push trix_lfm2_arm64 /data/local/tmp/
adb push models/LFM2-350M-Q4_0.gguf /data/local/tmp/
adb shell "cd /data/local/tmp && ./trix_lfm2_arm64 LFM2-350M-Q4_0.gguf"
```

### 3c. A/B Comparison with llama.cpp Baseline

- llama.cpp on device: 26 tok/s generation
- Compare: same prompt, same temperature (0), verify token-for-token match
- Measure: tok/s, latency percentiles, peak RSS, thermal throttling over 100 tokens

**Expected outcome:** Baseline measurement for NEON-optimized chip on real hardware

---

## Phase 4: Advanced Optimizations

### 4a. Big/Little Core Scheduling

- Pin weight-heavy matvec to 2x A78 big cores (higher IPC, larger caches)
- Use A55 little cores for preprocessing (tokenization, sampling)
- `sched_setaffinity()` or `pthread_setaffinity_np()` for thread pinning

### 4b. Operator Fusion

- Fuse RMSNorm + Q4_0 matvec (avoid writing/reading the normed vector)
- Fuse SiLU + elementwise multiply in SwiGLU FFN (one pass over gate/up outputs)
- Fuse ShortConv: in_proj + split(b,c,x) + elementwise mul + conv + out_proj

### 4c. Weight Prefetch

- Use `__builtin_prefetch()` or `PRFM` instructions to prefetch next weight tile
  while computing current tile
- LDNP (non-temporal load) for weights that won't be reused (single-token generation)
- Ghost-stream pattern from Yinsen: overlap weight loads with FMA pipeline

### 4d. Memory Layout Optimization

- Repack weights into NEON-friendly layout at load time (one-time cost)
- Interleave 4 rows for `vdotq` processing (similar to llama.cpp's q4_0_4x8)
- Align weight rows to 64-byte cache lines

---

## Phase 5: Vulkan Compute (Stretch)

The PowerVR BXM-8-256 GPU shares the same memory bus as the CPU.
For batch=1 generation, CPU+NEON should saturate the bus.
Vulkan only helps if we can pipeline: GPU computes layer N while CPU
prefetches weights for layer N+1.

### 5a. Vulkan Q4_0 Matvec Shader

- 16KB shared memory, 512 max invocations
- Tile: 256 output rows per workgroup, each thread handles 4 rows
- Weight dequant in shader (avoid memory doubling from F32 conversion)

### 5b. CPU/GPU Pipeline

- Double-buffer: two weight tiles in shared memory
- CPU dispatches Vulkan compute while preparing next layer on CPU
- Use VK_KHR_external_memory for zero-copy access to mmap'd GGUF weights

**Only pursue if Phase 1-4 don't reach the 40 tok/s target.**

---

## Success Metrics

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| tok/s generation | 26 | 40 | 52 |
| Time-to-first-token | ~38ms | ~25ms | ~19ms |
| Peak RSS | ~300MB | ~220MB | ~220MB |
| Binary size | ~2MB (llama.cpp) | ~50KB | ~50KB |
| External deps | llama.cpp + KleidiAI | none | none |

## Key Principles

1. **Memory bandwidth is the wall.** Every optimization must reduce bytes
   read or increase effective bandwidth. Compute optimizations without
   bandwidth awareness are worthless.

2. **Measure on real hardware.** The Moto G Power with its Dimensity 930
   has specific cache sizes, bus widths, and thermal characteristics that
   simulators can't replicate.

3. **No facades.** Every number must be reproducible. Every claim must be
   backed by a test that can be run independently.

4. **Frozen shapes, not frameworks.** The chip is a fixed computation graph.
   No dynamic dispatch, no abstraction layers, no plugin systems. The shape
   of compute is known at compile time and baked into the binary.
