# PRD: TriX Enhancement Layer for LFM2 Inference on Dimensity 930

## Status: ACTIVE
## Authors: Tripp + Claude
## Date: February 1, 2026
## Target: Motorola moto g power 5G (2023) running LFM2-350M

---

## 0. Ground Truth

Device confirmed via `adb shell cat /proc/cpuinfo`:

| Component | Spec | Verified |
|-----------|------|----------|
| SoC | MediaTek MT6855 (Dimensity 930) | `getprop ro.hardware` = mt6855 |
| Big cores | 2x Cortex-A78 (part 0xd41) | cpuinfo CPU part |
| Little cores | 6x Cortex-A55 (part 0xd05) | cpuinfo CPU part |
| SIMD | NEON + dotprod (`asimddp`) | cpuinfo Features |
| i8mm | **NOT available** | Not in features list |
| A78 L1D/L2 | 64KB / 256KB per core | ARM spec |
| A55 L1D/L2 | 32KB / 128KB per core | ARM spec |
| L3 (shared) | 512KB (MediaTek config) | Estimated |
| GPU | PowerVR BXM-8-256 | /dev/pvr_sync present |
| RAM | 3.6GB LPDDR4X | /proc/meminfo |
| Memory BW | ~12-13 GB/s (LPDDR4X-4266) | Spec sheet |

### Current Baseline

llama.cpp with KleidiAI on 4 threads:
- **Prompt eval**: 63 tok/s (15.9 ms/tok)
- **Generation**: 26 tok/s (38.5 ms/tok)
- **Theoretical minimum**: ~16 ms/tok (190MB weights / 12 GB/s)
- **Overhead ratio**: 38.5 / 16 = **2.4x**

The 2.4x gap comes from: GGML graph dispatch overhead, cache pollution from
weight streaming, suboptimal activation compute (libm exp/tanh), and thread
synchronization costs.

### What This Is NOT

This is NOT a replacement for llama.cpp. llama.cpp handles:
- GGUF parsing, tokenization, sampling, KV cache management
- The GGML compute graph and operator dispatch
- Multi-platform support and correctness

This IS an enhancement layer that composes TriX frozen primitives into
hardware-shaped compute kernels that llama.cpp can call for the inner loops
on this specific device.

---

## 1. The Three Optimizations

### 1A. Ghost-Stream Q4_0 Matvec (Cache Coherency)

**Problem**: Q4_0 matvec reads ~190MB of weights per token. On A55's 32KB L1D,
every weight row load evicts the activation vector from cache. The activation
vector (4KB for D_MODEL=1024, 18KB for FFN_HIDDEN=4608) is read M times
(once per output row) but gets thrashed by weight data on every iteration.

**Solution**: Adapted from Yinsen's `neon_ghost12_matvec`:

1. **LDNP (Load Non-Temporal Pair)** for weight data. Bypasses L2 cache.
   Weights are read exactly once per token -- they're cold data that should
   NOT pollute the cache hierarchy.

2. **Pin activation vector in L1**. The activation vector is the "hot" data --
   read N times across all output rows. By using LDNP for weights, the
   activation stays resident in L1.

3. **Multi-output-channel accumulation**. Process 8 output rows simultaneously
   (8 accumulators in v0-v7). Each iteration loads the same activation chunk
   once and multiplies against 8 different weight rows. This amortizes the
   activation load cost 8x.

4. **Q4_0 dequant fused into the SDOT path**. Each Q4_0 block has 32 values
   in 18 bytes. Dequant to int8 (subtract 8 to center), then use SDOT for
   the accumulation. The FP16 scale gets applied once per block after the
   integer accumulation.

**Register budget** (A78 has 32 NEON registers):
```
v0-v7:   8 accumulators (int32x4, 8 output channels)
v8-v11:  activation vector (4 x 16 bytes = 64 activations per iter)
v12-v15: scratch for Q4_0 unpack
v16-v23: weight data (8 rows x 16 bytes, loaded via LDNP)
v24-v27: Q4_0 scales (FP16->FP32) 
v28-v31: constants (8-subtract, masks)
```

**Expected improvement**: 1.5-2x on matvec (closing the 2.4x gap to ~1.3-1.6x).
The A55 cores benefit more than A78 because their L1 is half the size.

**Key constraint**: No i8mm on this device. Must use SDOT (dotprod), not SMMLA.
This means 4-way MAC per instruction (vs 8-way for i8mm). The Yinsen I8MM
kernels are NOT applicable here -- only the SDOT and ghost-stream patterns.

### 1B. Spline Activation Tables (Eliminate exp())

**Problem**: SiLU (in FFN) calls `expf(-x)`. Softmax (in attention) calls
`expf(x - max)`. On A55 without hardware transcendental support, `expf()`
is a ~30-cycle libm routine. For LFM2-350M:
- SiLU: 16 layers x 4608 elements = 73,728 `expf()` calls per token
- Softmax: 6 layers x 16 heads x ~100 positions = ~9,600 `expf()` calls
- Total: ~83,000 `expf()` calls = ~2.5M cycles = ~1.2 ms at 2 GHz (A78)

That's ~3% of the 38.5ms generation time. Not dominant, but not free.

**Solution**: Piecewise cubic spline tables evaluated with pure FMA.

For sigmoid (from which SiLU derives as `x * sigmoid(x)`):
- Domain: [-8, 8] (outside this, sigmoid is 0 or 1 to float precision)
- 16 segments, each a cubic polynomial: `a*t^3 + b*t^2 + c*t + d`
- Coefficients fit at initialization from exact sigmoid values
- Evaluation: 1 integer truncation + 3 FMA + 1 table load = ~5 cycles

```c
// Spline sigmoid evaluation
float trix_sigmoid_spline(float x) {
    if (x <= -8.0f) return 0.0f;
    if (x >=  8.0f) return 1.0f;
    float t = (x + 8.0f) * (1.0f / 16.0f);  // map to [0, 1) over full range
    int seg = (int)(t * 16.0f);              // segment index [0..15]
    if (seg > 15) seg = 15;
    float u = t * 16.0f - (float)seg;        // local parameter [0, 1)
    const float *c = spline_table + seg * 4;  // 4 coefficients
    return ((c[0]*u + c[1])*u + c[2])*u + c[3]; // Horner's method: 3 FMA
}
```

**Spline table**: 16 segments x 4 coefficients x 4 bytes = **256 bytes**.
Fits in a single A55 cache line set. Stays permanently hot in L1.

For softmax's `exp()`, we use the same approach but over [-16, 0] (after
max subtraction, all values are <= 0):
- 32 segments, 4 coefficients each = 512 bytes
- Accuracy: max relative error < 0.1% (well within Q4_0 noise floor)

**Why not Schraudolph?** The bit-trick (`u.i = (int32_t)(x * 12102203.0f + 1064866805.0f)`)
works great on CPU. But:
1. It has ~4% relative error, vs <0.1% for 16-segment splines
2. On the PowerVR GPU (if we ever go that path), int-float reinterpret
   may serialize or be unavailable in GLSL
3. Splines are pure FMA -- the one thing every compute unit does fast
4. The spline table is constant data that can live in shared memory on GPU

**For this PRD, we use Schraudolph on CPU (where it's 3 instructions) and
keep the spline tables ready for the Vulkan path later.**

### 1C. Fused Layer Chips (Eliminate Dispatch Overhead)

**Problem**: llama.cpp's GGML dispatches each operation as a separate graph
node. A single ShortConv layer involves:
1. RMSNorm (graph node)
2. in_proj matvec (graph node)
3. Split b/c/x (graph node)
4. Element-wise b*x (graph node)
5. Conv state shift + convolve (graph node)
6. Element-wise c*conv (graph node)
7. out_proj matvec (graph node)

That's 7 dispatch events per conv layer. Each dispatch involves: checking
tensor sizes, finding the kernel, setting up threading, synchronizing
threads, writing results to memory. On a 4-thread system, thread sync
alone adds ~5-10 us per dispatch = 35-70 us per conv layer = 350-700 us
for all 10 conv layers.

**Solution**: Fuse each layer into a single function call that runs start
to finish without returning to the dispatcher. Intermediates live on the
stack (the TriX chip already does this in `trix_shortconv_chip`).

The fused chip:
- Eliminates 7 dispatch events per conv layer (10 layers = 70 events saved)
- Eliminates 5 dispatch events per attention layer (6 layers = 30 events saved)  
- Eliminates 5 dispatch events per FFN (16 layers = 80 events saved)
- Total: ~180 dispatch events eliminated per token
- At ~5-10 us per dispatch: **saves 0.9-1.8 ms per token** (~3-5%)

More importantly: fused layers keep intermediate activations in registers
or on the stack (L1-resident) instead of writing them to GGML tensor
buffers in main memory. For the FFN:
- Without fusion: gate_out[4608] written to RAM, then up_out[4608] written,
  then both read back for silu+mul. Total extra traffic: 36KB write + 36KB read.
- With fusion: gate_out and up_out stay in L1, consumed immediately.

**Combined with ghost-stream matvec**: The fused layer chip calls the
ghost-stream matvec for each projection, with the activation vector
already cache-resident from the previous operation. No cold-start
cache misses between operations within a layer.

---

## 2. Architecture

```
                    llama.cpp
                       |
                 [GGUF loading]
                 [Tokenization]
                 [KV cache mgmt]
                 [Sampling]
                       |
            +----------+----------+
            |                     |
      [GGML default]     [TriX enhancement]
      (fallback path)    (when available)
                                  |
                    +-------------+-------------+
                    |             |             |
              ghost-stream   spline        fused layer
              Q4_0 matvec   activations    chips
              (NEON+LDNP)   (FMA-only)    (zero-dispatch)
```

### File Structure

```
research/trix_lfm2_chip/
  include/
    lfm2_trix.h          -- existing: types, API
    trix_neon_q4_0.h     -- NEW: ghost-stream matvec kernels
    trix_spline.h        -- NEW: spline coefficient tables + eval
  src/
    lfm2_trix.c          -- existing: forward pass (update to use new kernels)
    gguf_loader.c         -- existing: GGUF parser
    trix_neon_q4_0.c     -- NEW: NEON intrinsic matvec implementation
    trix_spline.c        -- NEW: spline coefficient fitting + eval
  tests/
    test_trix_lfm2.c     -- existing: update with kernel benchmarks
    test_neon_matvec.c   -- NEW: correctness + perf test for ghost-stream
    test_spline.c        -- NEW: accuracy test for spline activations
    bench_layers.c       -- NEW: per-layer timing comparison vs llama.cpp
    run_lfm2.c           -- existing: token generator
  CMakeLists.txt         -- existing: update for new sources
```

---

## 3. Implementation Order

### Phase 1: Correctness (prerequisite)

Fix the existing forward pass to match llama.cpp token-for-token.
The current chip runs but produces wrong output (repetitive token 1323).
Without correctness, we can't validate that optimizations don't break things.

**Approach**: Layer-by-layer comparison. Run llama.cpp with debug logging
to dump intermediate activations after each layer. Compare against our
chip's output at the same points.

### Phase 2: Ghost-Stream Q4_0 Matvec

The hot path. Write the NEON kernel with:
- LDNP for weight loading (non-temporal)
- 8 output channels per iteration
- SDOT for Q4_0 dequant+accumulate
- Proper handling of Q4_0 block structure (18 bytes per 32 values)

Test on device, benchmark against KleidiAI baseline.

### Phase 3: Spline Activations

Implement cubic spline tables for sigmoid (=> SiLU) and exp (=> softmax).
Fit coefficients at init. Test accuracy against libm reference.
Integrate into fused layer chips.

### Phase 4: Fused Layer Chips

Update `trix_shortconv_chip`, `trix_attention_chip`, `trix_ffn_chip` to
use the ghost-stream matvec and spline activations. Benchmark full forward
pass on device.

### Phase 5: Integration

Wire the TriX chip as a callable from llama.cpp's forward pass, or
run standalone with llama.cpp's GGUF loader + tokenizer.

---

## 4. Success Criteria

| Metric | Baseline (llama.cpp) | Target | Stretch |
|--------|---------------------|--------|---------|
| Generation tok/s | 26 | 40 | 52 |
| Generation ms/tok | 38.5 | 25 | 19 |
| Overhead ratio | 2.4x | 1.6x | 1.2x |
| Correctness | reference | bit-accurate Q4_0 | -- |
| Binary size | N/A | < 64KB .text | < 32KB |
| Peak RAM above model | ~50MB | < 20MB | < 10MB |

The 40 tok/s target (1.5x improvement) comes from:
- Ghost-stream matvec: 1.3x on the 80% that is matvec = 1.24x overall
- Fused layers: 1.05x from eliminated dispatch
- Spline activations: 1.03x from faster exp()
- Combined: 1.24 x 1.05 x 1.03 = **1.34x** (conservative)
- Plus cache coherency compound effects: estimated additional 1.1x
- Total: 1.34 x 1.1 = **1.47x** => ~38 tok/s

The 52 tok/s stretch assumes perfect ghost-stream implementation closing
the remaining gap to memory bandwidth limits.

---

## 5. What We're NOT Doing

- **Vulkan persistent kernel**: Deferred. CPU with ghost-stream NEON is the
  faster path to results. GPU shares the same memory bus (LPDDR4X UMA) so
  can't add bandwidth. The Vulkan path is only useful if we can exploit
  PowerVR's tile memory in ways the CPU can't -- that's a research question,
  not an engineering task.

- **i8mm kernels**: Device doesn't have ARMv8.6 i8mm. Only SDOT (dotprod).

- **Ternary weights**: LFM2-350M uses Q4_0 weights from HuggingFace. We work
  with what the model ships as. The Yinsen ternary infrastructure is for
  ternary-native models (CfC), not for requantizing an existing Q4_0 model.

- **Custom tokenizer**: Use llama.cpp's existing LFM2 tokenizer.

- **Training or fine-tuning**: Pure inference optimization.
