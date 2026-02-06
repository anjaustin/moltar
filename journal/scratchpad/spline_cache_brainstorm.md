# Brainstorm: Splines in Cache via Coherency Hardware

## The Idea

Instead of computing weight × activation on every token:
1. Precompute spline lookup tables that approximate the weight matrix behavior
2. Store splines in L1/L2/L3 cache (they're small, they fit)
3. Use cache coherency to keep splines synchronized across cores
4. Replace GEMM with spline interpolation (table lookup + lerp)

## Why This Might Work

### The Weight Matrix is Static
- Model weights don't change during inference
- We're computing `y = W @ x` where W is constant
- What if we precomputed `y = spline(x)` for common x patterns?

### Splines are Compact
- A cubic spline needs 4 coefficients per segment
- If we discretize the input range into N segments, we need 4N floats
- For 256 segments: 256 × 4 × 4 bytes = 4 KB per output neuron
- That fits in L1 cache (64 KB on A78)

### Cache Coherency is Free
- Once spline coefficients are loaded, they stay in cache
- Multiple cores can read the same cache lines (Shared state)
- No coherency traffic because splines are read-only

## The Architecture

```
Traditional GEMM:
  x[K] → W[N×K] → y[N]
  Each y[n] = Σ W[n,k] × x[k]  (K multiply-adds)

Spline Approximation:
  x[K] → hash/project to index i
  y[N] = spline_table[N][i]  (table lookup + interpolation)
  
The trick: project high-dim x to low-dim index
```

## The Problem: Dimensionality

A linear layer has K inputs and N outputs. The "function" we're approximating is:
- f: R^K → R^N

For LFM2-350M ffn_gate (N=4608, K=1024):
- Input space is 1024-dimensional
- We can't spline a 1024-D function directly

### Possible Solutions

#### Option A: Per-Neuron 1D Splines
Approximate each output neuron as a function of a single "projected" input:
```
z = x · v  (project K-dim x to 1D via learned vector v)
y[n] = spline_n(z)
```
This loses the matrix structure entirely. Probably too lossy.

#### Option B: Factored Splines
Factor W ≈ U @ V where:
- V: K → M (reduce dimension)
- U: M → N (expand dimension)

Then spline the intermediate representation:
```
z = V @ x      (small matmul, M << K)
h = spline(z)  (element-wise spline on M-dim vector)
y = U @ h      (small matmul, M << N)
```
For M=64: two 64×K and N×64 matmuls instead of one N×K.
With spline in the middle, the intermediate activation is "shaped".

#### Option C: Ternary + Spline Hybrid
Use ternary weights for the main matmul (2.2x speedup proven).
Use splines for the activation functions (SwiGLU, etc.).

Activation functions ARE 1D! Perfect for splines:
```
SwiGLU(x) = x × sigmoid(x)  // Two function evals
         ≈ spline_swiglu(x) // One table lookup + lerp
```

This is actually the most promising path.

## Deep Dive: Splined Activation Functions

### Current Activation Cost

From our profiling:
- `ggml_vec_swiglu_f32`: 0.66% of time
- `ggml_vec_soft_max_f32`: 0.33% of time
- `ggml_compute_forward_rms_norm`: 0.41% of time

These are small but they involve expensive ops (exp, div, sqrt).

### Spline Table Design

For SwiGLU = x × sigmoid(x):
```c
// Precompute at init time
#define SPLINE_SEGMENTS 256
#define SPLINE_MIN -8.0f
#define SPLINE_MAX 8.0f
#define SPLINE_STEP ((SPLINE_MAX - SPLINE_MIN) / SPLINE_SEGMENTS)

float swiglu_table[SPLINE_SEGMENTS + 1];  // 1 KB

void init_swiglu_table() {
    for (int i = 0; i <= SPLINE_SEGMENTS; i++) {
        float x = SPLINE_MIN + i * SPLINE_STEP;
        swiglu_table[i] = x / (1.0f + expf(-x));  // x * sigmoid(x)
    }
}

// Runtime: table lookup + linear interpolation
float swiglu_spline(float x) {
    if (x <= SPLINE_MIN) return swiglu_table[0];
    if (x >= SPLINE_MAX) return swiglu_table[SPLINE_SEGMENTS];
    
    float idx_f = (x - SPLINE_MIN) / SPLINE_STEP;
    int idx = (int)idx_f;
    float frac = idx_f - idx;
    
    return swiglu_table[idx] * (1.0f - frac) + swiglu_table[idx + 1] * frac;
}
```

### NEON Vectorization

The spline lookup can be vectorized:
```c
// Process 4 values at once
float32x4_t swiglu_spline_neon(float32x4_t x) {
    // Clamp to table range
    x = vmaxq_f32(x, vdupq_n_f32(SPLINE_MIN));
    x = vminq_f32(x, vdupq_n_f32(SPLINE_MAX));
    
    // Compute indices
    float32x4_t idx_f = vmulq_f32(
        vsubq_f32(x, vdupq_n_f32(SPLINE_MIN)),
        vdupq_n_f32(1.0f / SPLINE_STEP)
    );
    
    // Integer and fractional parts
    int32x4_t idx = vcvtq_s32_f32(idx_f);
    float32x4_t frac = vsubq_f32(idx_f, vcvtq_f32_s32(idx));
    
    // Gather from table (this is the tricky part - no native gather)
    // Would need scalar fallback or TBL tricks
    // ...
}
```

The problem: ARM NEON doesn't have efficient gather instructions like x86 AVX2. Table lookups need to be done scalar or via TBL (which works on bytes, not floats).

### Alternative: Polynomial Approximation

Instead of table lookup, use a polynomial that's fast to evaluate:

```c
// SwiGLU ≈ polynomial (fitted)
// For x in [-4, 4], this is accurate to ~0.1%
float swiglu_poly(float x) {
    // Coefficients from least-squares fit
    float x2 = x * x;
    float x3 = x2 * x;
    return 0.5f * x + 0.25f * x2 - 0.02f * x3;  // Example coefficients
}
```

This is just FMA operations - very fast on NEON.

## The Real Opportunity: Cache-Resident Lookup Tables

### What Fits in L1 (64 KB)?

| Table | Size | Fits? |
|-------|------|-------|
| SwiGLU spline (256 points) | 1 KB | Yes |
| Sigmoid spline (256 points) | 1 KB | Yes |
| Tanh spline (256 points) | 1 KB | Yes |
| Softmax exp table (256 points) | 1 KB | Yes |
| RMSNorm rsqrt table (256 points) | 1 KB | Yes |
| **Total activation tables** | **5 KB** | **Yes** |

### What Fits in L2 (256 KB)?

| Table | Size | Fits? |
|-------|------|-------|
| Trit decode LUT (16 bytes) | 16 B | Yes |
| Ternary weight scales (per layer) | ~10 KB | Yes |
| Activation splines | 5 KB | Yes |
| **Total lookup tables** | **~15 KB** | **Yes, 6% of L2** |

### What Fits in L3 (1 MB shared)?

This is where we could cache larger structures:
- Partial weight matrices?
- KV cache hot pages?
- Prefetched next-layer weights?

## The Coherency Play

### Scenario: Spline Tables Shared Across Cores

```
Core 6 (inference thread 1):
  Load swiglu_table → L1 (Exclusive)
  Compute SwiGLU for half the neurons
  
Core 7 (inference thread 2):
  Want swiglu_table → Snoop finds Core 6 has it
  Cache-to-cache transfer → Both cores now Shared
  
Result: Table is in both L1 caches, NO memory traffic for subsequent accesses
```

This is exactly how coherency should work - the table gets replicated where needed.

### Scenario: L3 as Broadcast Medium

```
LITTLE Core 0 (prefetch worker):
  Load next-layer spline coefficients from DRAM → L3
  
BIG Core 6 (inference):
  When ready for next layer → L3 hit instead of DRAM
```

The L3 acts as a staging area. LITTLE cores warm it, BIG cores consume from it.

## Combining with Ternary

The most promising architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPLINE + TERNARY INFERENCE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   L1 Cache (64 KB):                                                         │
│     - Trit decode LUT (16 bytes)                                            │
│     - Activation splines (5 KB)                                             │
│     - Hot weight blocks (~50 KB)                                            │
│                                                                              │
│   L2 Cache (256 KB):                                                        │
│     - Current layer ternary weights                                         │
│     - Block scales                                                          │
│                                                                              │
│   L3 Cache (1 MB):                                                          │
│     - Next layer weights (prefetched)                                       │
│     - KV cache hot region                                                   │
│                                                                              │
│   Compute Path:                                                             │
│     1. Load 2-bit ternary weights from L2                                  │
│     2. TBL decode to {-1, 0, +1} via L1 LUT                                │
│     3. SDOT accumulate                                                      │
│     4. Spline activation from L1 LUT                                        │
│     5. Next layer weights already in L3 (LITTLE prefetched)                │
│                                                                              │
│   Coherency:                                                                │
│     - Spline tables: Shared across cores (read-only, replicated)            │
│     - Ternary weights: Streaming (no reuse, prefetch helps)                 │
│     - Scales: Shared (small, fits everywhere)                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Estimated Impact

| Optimization | Speedup | Status |
|--------------|---------|--------|
| Ternary weights (proven) | 2.2x on GEMM | Kernel ready, integration needed |
| Spline activations | ~1.1x on activation | Easy to implement |
| L3 prefetch | ~1.05x (hide latency) | Needs testing |
| **Combined** | **~2.4x overall?** | Theoretical |

The GEMM is 70% of time. If we speed that up 2.2x:
- GEMM: 70% → 32% (0.7 / 2.2)
- Other: 30% → 30%
- New total: 62% of original → 1.6x overall

Add spline activations (small win) and prefetch (small win) → maybe 1.7-1.8x overall.

## Questions to Probe

1. **Can NEON do efficient table lookups?**
   - TBL works on bytes, need float version
   - Or use polynomial approximation instead

2. **Does L3 prefetch actually help?**
   - We measured 99% cache hit rate already
   - Might be hitting L3 ceiling, not L2

3. **What's the quality impact of spline activations?**
   - Linear interpolation on 256 points is very accurate
   - Need to verify no precision issues

## Next Steps

1. **Implement spline activation kernels** - Replace SwiGLU/sigmoid with table lookup
2. **Measure spline accuracy** - Verify <0.1% error vs true function
3. **Benchmark spline vs original** - Is it actually faster on ARM?
4. **If yes: Integrate with ternary** - Full optimized inference path
