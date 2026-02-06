# GPU as Spline Compute Engine

## The Pivot

Previous attempt: GPU as memory prefetcher → FAILED (coherency overhead)

New idea: GPU as **parallel spline evaluator** → Use its 128-wide SIMD for actual compute

## The PowerVR BXM-8-256

```
PowerVR BXM-8-256 Specs:
- 2 Unified Shading Clusters (USC)
- Each USC: 128-wide SIMD ALU
- FP16: ~200 GFLOPS
- FP32: ~100 GFLOPS
- Shared memory per workgroup
- Texture units with hardware interpolation
```

The key insight: **Texture units do hardware bilinear interpolation**. That's a 2D spline evaluation in hardware.

## Two Approaches

### Approach A: ALU-Based Spline Evaluation

Use the 128-wide SIMD to evaluate splines in parallel:

```glsl
// Vulkan compute shader
layout(local_size_x = 128) in;

layout(binding = 0) buffer Inputs { float x[]; };
layout(binding = 1) buffer Outputs { float y[]; };
layout(binding = 2) buffer Coeffs { vec4 spline_coeffs[256]; };  // 256 segments

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float xi = x[idx];
    
    // Find spline segment
    int seg = clamp(int(xi * 32.0 + 128.0), 0, 255);
    vec4 c = spline_coeffs[seg];
    
    // Evaluate cubic: y = c.x + c.y*t + c.z*t² + c.w*t³
    float t = fract(xi * 32.0 + 128.0);
    y[idx] = c.x + t * (c.y + t * (c.z + t * c.w));
}
```

128 spline evaluations per wavefront. But there's transfer overhead...

### Approach B: Texture Hardware as Spline Engine

The texture unit does **hardware bilinear interpolation**. A 1D texture with linear filtering IS a piecewise linear spline:

```glsl
layout(binding = 0) uniform sampler1D spline_tex;  // 256 texels = 256 segments

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float xi = x[idx];
    
    // Texture fetch with linear interpolation = FREE spline!
    float u = xi * 0.5 + 0.5;  // Map to [0, 1]
    y[idx] = texture(spline_tex, u).r;
}
```

The texture unit:
1. Computes the texel coordinate
2. Fetches two neighboring texels
3. Linearly interpolates between them
4. Returns the result

**One instruction, hardware interpolation, 128 parallel evaluations.**

## The Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CPU + GPU SPLINE HYBRID                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CPU (Cortex-A78):                                                          │
│    - Ternary GEMM (2.2x faster, proven)                                     │
│    - Sequential layer execution                                              │
│    - Feeds activation batches to GPU                                         │
│                                                                              │
│  GPU (PowerVR BXM-8-256):                                                   │
│    - Receives raw activations (pre-nonlinearity)                            │
│    - Evaluates SwiGLU/sigmoid/tanh via texture splines                      │
│    - Returns transformed activations                                         │
│    - 128-wide SIMD, hardware interpolation                                   │
│                                                                              │
│  Data Flow:                                                                  │
│                                                                              │
│    CPU: [ternary GEMM] → raw_act[4096] ──┐                                  │
│                                           │                                  │
│                                     ┌─────▼─────┐                           │
│                                     │    GPU    │                           │
│                                     │  Spline   │                           │
│                                     │  Texture  │                           │
│                                     └─────┬─────┘                           │
│                                           │                                  │
│    CPU: [next layer] ◄── act[4096] ◄─────┘                                  │
│                                                                              │
│  Memory:                                                                     │
│    - Spline texture: 256 × fp16 = 512 bytes (fits in GPU cache)            │
│    - Transfer per layer: 4096 × fp16 × 2 = 16 KB (act in + out)            │
│    - Transfer time: 16 KB / 8 GB/s = 2 µs                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Question: Is Transfer Overhead Worth It?

### Current Activation Cost (CPU)

From profiling:
- `ggml_vec_swiglu_f32`: 0.66% of total time
- Total inference: ~35 ms per token (28 tok/s)
- SwiGLU time: 0.66% × 35 ms = 0.23 ms per token

For 16 layers with SwiGLU:
- Per layer: 0.23 ms / 16 = 14 µs

### GPU Spline Cost

- Transfer to GPU: 2 µs
- GPU compute: ~1 µs (128-wide, 4096 elements = 32 wavefronts, ~30 cycles each)
- Transfer back: 2 µs
- Total: ~5 µs

**GPU is ~3x faster per layer for activation function!**

But wait - this is only 14 µs vs 5 µs = saving 9 µs per layer × 16 layers = 144 µs per token.

At 28 tok/s (35 ms/token), saving 144 µs is only 0.4% improvement. Not worth the complexity.

## Reframe: GPU for Heavier Compute

Activation functions are too light. What about something heavier?

### Option 1: Attention Score Computation

Attention: Q @ K^T is a matmul. For 512 context, 8 heads:
- Q: [8, 64] 
- K: [8, 512, 64]
- Scores: [8, 512]

That's 8 × 64 × 512 = 262K multiply-adds per layer per token.

For attention layers only (6 in LFM2):
- 262K × 6 = 1.57M ops per token
- GPU at 100 GFLOPS: 16 µs
- Plus transfer: 8 KB Q + 256 KB K = 264 KB
- Transfer at 8 GB/s: 33 µs
- Total: ~50 µs

CPU currently does this in... let's check the profile again.

### Option 2: Full Layer Offload

What if we ran entire FFN layers on GPU?

FFN for LFM2-350M:
- Up projection: 1024 → 4608 (4.7M weights)
- Gate projection: 1024 → 4608 (4.7M weights)  
- Down projection: 4608 → 1024 (4.7M weights)
- Total: ~14M weights per FFN

Transfer: 14M × 0.5 bytes (Q4) = 7 MB per layer
At 8 GB/s: 875 µs just for weights!

Not viable - weights too large.

### Option 3: Pre-Batched Spline Tables

What if the GPU precomputes **expanded** spline tables that the CPU can use directly?

```
GPU precomputes at init:
  For each unique activation function:
    For each possible int8 input (-128 to 127):
      Compute output via texture spline
    Store in 256-element LUT

CPU uses at runtime:
  output = lut[quantized_input]  // Simple table lookup
```

This is what we already proposed for CPU TBL! The GPU doesn't help at runtime.

But... the GPU could precompute **per-layer calibrated** splines:

```
GPU analyzes layer activations:
  1. Run calibration pass with sample inputs
  2. Measure activation distribution per layer
  3. Compute optimal spline knots for each layer's distribution
  4. Store 16 per-layer LUTs (one per layer)

CPU uses at runtime:
  output = layer_lut[layer_id][quantized_input]
```

This is a **one-time cost** at model load, improving accuracy of the spline approximation.

## The Real Opportunity: GPU for Speculative Decoding

What if GPU runs a **draft model** while CPU runs the main model?

```
Speculative Decoding:
  1. GPU runs tiny model (50M params?) to generate 4 draft tokens
  2. CPU verifies all 4 tokens in one forward pass
  3. If drafts are correct, 4x speedup
  4. If wrong, fall back to normal generation

Why this works:
  - GPU and CPU run DIFFERENT models (no data sharing)
  - No coherency overhead
  - GPU's 128-wide SIMD is perfect for tiny models
  - Can use aggressive quantization on draft model
```

This is architecturally cleaner than trying to share data between CPU and GPU.

## Synthesis: Where GPU Helps

| Use Case | Viable? | Why |
|----------|---------|-----|
| Memory prefetch | No | Coherency overhead |
| Activation splines | No | Too light, transfer overhead |
| Attention offload | Maybe | Transfer cost borderline |
| Full layer offload | No | Weight transfer too expensive |
| Speculative decoding | Yes! | Independent computation |
| Calibration/analysis | Yes | One-time cost at load |

## Recommendation

1. **Don't use GPU for runtime spline evaluation** - transfer overhead exceeds benefit
2. **Use CPU TBL for splines** - already have the mechanism, zero overhead
3. **Consider GPU for speculative decoding** - architecturally sound, potentially 2-4x speedup
4. **Use GPU for offline calibration** - generate per-layer spline tables at model load

## The CPU-Only Spline Path

Given that GPU doesn't help at runtime, the optimal path is:

```
CPU Ternary + Spline Architecture:
  1. Load model with ternary weights
  2. GPU (at load time) calibrates per-layer spline LUTs
  3. CPU stores 16-byte LUTs in NEON registers
  4. Runtime:
     a. TBL #1: Decode ternary weights
     b. SDOT: Accumulate
     c. TBL #2: Spline activation
  5. All compute stays on CPU, zero transfer overhead
```

This is the "headroom" you mentioned - GPU gives us better spline tables, CPU uses them efficiently.
