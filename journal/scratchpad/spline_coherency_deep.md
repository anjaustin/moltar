# Deep Dive: Splines as Cache-Coherent Computation

## Reframing the Question

Not "store splines in cache" but rather:
**Can we use the cache coherency state machine itself to perform spline interpolation?**

The coherency protocol (MESI/MOESI) is essentially a distributed state machine. What if we encoded computation into its transitions?

## The Cache Coherency State Machine

```
        ┌─────────────────────────────────────────────────────────────────┐
        │                    MESI State Machine                            │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                  │
        │     ┌───────┐    Read     ┌───────┐                            │
        │     │Invalid│ ──────────→ │Shared │                            │
        │     └───┬───┘             └───┬───┘                            │
        │         │                     │                                 │
        │    Write│                Write│                                 │
        │         ▼                     ▼                                 │
        │     ┌───────┐   Snoop    ┌────────┐                            │
        │     │Exclusive│ ◄──────── │Modified│                            │
        │     └───────┘            └────────┘                            │
        │                                                                  │
        └─────────────────────────────────────────────────────────────────┘
```

Each state transition involves hardware that:
- Reads/writes data
- Broadcasts to other cores
- Updates state bits

## Crazy Idea: Spline Coefficients as Cache Line Data

What if each cache line held spline coefficients, and the "computation" was the act of reading them?

```
Cache Line Layout (64 bytes):
┌────────────────────────────────────────────────────────────────┐
│ a0 │ a1 │ a2 │ a3 │ b0 │ b1 │ b2 │ b3 │ ... │ (16 floats)    │
└────────────────────────────────────────────────────────────────┘
     Segment 0         Segment 1         ...

To evaluate spline at x:
1. Compute segment index i = floor(x / segment_width)
2. Load cache line containing segment i → coefficients arrive
3. Compute: y = a[i] + b[i]*t + c[i]*t² + d[i]*t³
```

The cache hardware does the "routing" - selecting which coefficients to load based on the address we access.

## The Address as Computation

Here's where it gets interesting. The cache is indexed by address. The address is computed by the CPU. So:

```c
// Traditional spline evaluation
float spline_eval(float x, float* coeffs, int n_segments) {
    int i = (int)(x * n_segments);      // Compute index
    float t = x * n_segments - i;        // Fractional part
    float* c = &coeffs[i * 4];           // Pointer arithmetic
    return c[0] + t*(c[1] + t*(c[2] + t*c[3]));  // Horner's method
}

// What if we precomputed ALL outputs?
// For 8-bit input quantization (256 values):
float spline_lut[256];  // 1 KB

float spline_eval_lut(uint8_t x_quantized) {
    return spline_lut[x_quantized];  // Just a load!
}
```

The spline computation is "baked" into the table. The cache hardware fetches the right value based on the quantized input.

## The Coherency Angle

Now, what if we use coherency to **share** these precomputed values across cores efficiently?

### Scenario: Spline LUTs in Shared State

```
Initialization:
  Core 0: Fill spline_lut[256] with precomputed values
  Core 0: Flush to memory
  
Runtime (all cores):
  Any core reads spline_lut[x] →
    - First access: Cache miss, fill from memory, state = Exclusive
    - Other cores access same line → Snoop, state = Shared
    - Now ALL cores have the line, NO further memory traffic
```

For a 1 KB LUT with 64-byte cache lines:
- 16 cache lines total
- After warmup, all 16 lines are Shared in every core's L1
- Zero coherency traffic, zero memory traffic

### Scenario: Different Splines for Different Layers

What if each layer has slightly different activation function parameters?

```
Layer 0: spline_lut_0[256]  // Tuned for layer 0's distribution
Layer 1: spline_lut_1[256]  // Tuned for layer 1's distribution
...
```

This is 1 KB per layer. For 16 layers = 16 KB total, easily fits in L2.

But here's the clever part: we can use **cache coloring** to ensure different layers' LUTs don't conflict:

```
Memory layout (assuming 8-way L1, 64B lines, 64KB L1):
  Address bits: [tag | set index (10 bits) | offset (6 bits)]
  
  Place spline_lut_0 at address 0x10000 (set 0)
  Place spline_lut_1 at address 0x10400 (set 16)
  ...
  
Each LUT occupies 16 sets (1 KB / 64B = 16 lines).
With 1024 sets available, we can have 64 non-conflicting LUTs.
```

## Even Crazier: Using Cache Line Size as Interpolation

A cubic spline segment needs 4 coefficients = 16 bytes. A cache line is 64 bytes. That's 4 segments per line.

What if we organized so that **neighboring segments are in the same cache line**?

```
Cache Line N:
  [Seg 4N coeff] [Seg 4N+1 coeff] [Seg 4N+2 coeff] [Seg 4N+3 coeff]

For input x, compute segment index i = floor(x * scale).
If i % 4 != 3, then segment i and i+1 are in the SAME cache line.

This means: linear interpolation between segments is "free" - one cache line fetch gives both values.
```

## The NEON Connection

ARM NEON has the TBL instruction which does byte-level table lookup:

```
TBL Vd.16B, {Vn.16B}, Vm.16B

- Vn contains the lookup table (16 bytes)
- Vm contains the indices (16 bytes, each 0-15)
- Vd contains the results (16 bytes)
```

This is a 16-entry LUT evaluated 16 times in parallel, in ONE instruction.

For splines, we could:
1. Quantize inputs to 4 bits (16 levels)
2. Store 16 spline outputs in a NEON register
3. Use TBL to lookup all 16 outputs simultaneously

```c
// 16 inputs, 16 outputs, one instruction
int8x16_t spline_neon_tbl(int8x16_t x_quantized, int8x16_t lut) {
    return vqtbl1q_s8(lut, x_quantized);
}
```

This is exactly what the ternary kernel uses! TBL to decode {-1, 0, +1} from 2-bit encoding.

## Combining Everything: The Spline-Ternary-Coherency Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPLINE-COHERENT INFERENCE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  L1 Cache (per core, 64 KB):                                                │
│    ┌────────────────────────────────────────────────────────────────────┐   │
│    │  Trit Decode LUT (16B) ← Loaded once, stays forever               │   │
│    │  Activation Spline LUTs (16 × 16B = 256B per function)            │   │
│    │    - sigmoid_lut: 16 points, NEON TBL accessible                   │   │
│    │    - swiglu_lut:  16 points                                        │   │
│    │    - tanh_lut:    16 points                                        │   │
│    │  State: SHARED (all cores have identical copy)                     │   │
│    │  Coherency traffic: ZERO after warmup                              │   │
│    └────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  L2 Cache (per cluster, 256 KB):                                            │
│    ┌────────────────────────────────────────────────────────────────────┐   │
│    │  Per-layer fine-grained LUTs (if needed)                           │   │
│    │  Current layer's ternary weights (streaming)                       │   │
│    │  Block scales                                                       │   │
│    └────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  L3 Cache (shared, 1 MB):                                                   │
│    ┌────────────────────────────────────────────────────────────────────┐   │
│    │  Next layer's weights (LITTLE cores prefetch here)                 │   │
│    │  KV cache (attention)                                               │   │
│    └────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Compute Flow:                                                              │
│    1. Load ternary weights (2-bit) from L2                                  │
│    2. TBL decode to {-1, 0, +1} using L1-resident LUT (16B)                │
│    3. SDOT accumulate                                                       │
│    4. Quantize activation to 4-bit index                                    │
│    5. TBL spline lookup for activation function (one instruction!)          │
│    6. Continue to next layer (weights already in L3)                        │
│                                                                              │
│  Coherency Behavior:                                                        │
│    - LUTs: SHARED state, zero snoop traffic                                 │
│    - Weights: EXCLUSIVE→INVALID (streaming, no reuse)                       │
│    - L3 prefetch: No coherency with L1/L2 (different cores)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Key Insight

**The spline computation becomes a cache access pattern.**

Instead of:
- Load coefficients → Compute polynomial → Store result

We do:
- Compute address from input → Load precomputed result

The cache hardware handles the "routing" (address decode) and "sharing" (coherency). We just need to organize data so the access patterns are cache-friendly.

## What This Means for Ternary + Activation

The ternary GEMM already uses TBL for weight decode. We can extend this:

```c
// Current ternary kernel inner loop
int8x16_t weights = vqtbl1q_s8(trit_lut, packed_trits);  // Decode weights
int32x4_t acc = vdotq_s32(acc, weights, activations);    // SDOT

// Extended with spline activation
// ... after accumulation ...
int8x16_t act_quantized = vqshrn_n_s32(acc, 4);  // Quantize to 4-bit
int8x16_t act_splined = vqtbl1q_s8(swiglu_lut, act_quantized);  // Spline!
```

The activation function becomes a SINGLE TBL instruction, using an L1-resident 16-byte LUT.

## Accuracy Concern

16 levels (4-bit) for activation function quantization is coarse. But:

1. The LUT can be per-layer, tuned to that layer's activation distribution
2. We can use 256-level (8-bit) for higher accuracy at the cost of larger LUT (256B)
3. The ternary weights are already lossy - activation quantization might not be the bottleneck

## Next Step: Probe It

Build a probe that:
1. Implements TBL-based spline activation
2. Compares accuracy to true activation function
3. Measures throughput (should be essentially free - one instruction)

If accuracy is acceptable, integrate with ternary kernel for combined speedup.
