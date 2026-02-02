# Synthesis: Matvec Shootout — Ternary vs Q2_K vs Q4_0 on Dimensity 930

## Core Insight

This is a bandwidth problem, not a compute problem. At 76% of theoretical LPDDR4X bandwidth, the only way to go faster is to read fewer bytes per token. Three paths achieve ~2 bits per weight: Q4_0 with zero-skipping (no bandwidth savings), Q2_K via llama.cpp (proven format, unknown kernel quality on this SoC), and ternary via Yinsen's NEON kernels (2-bit packed, zero-multiply, cache-optimized). We probe all three. The device picks the winner.

## Architecture: The Three-Way Matvec Probe

A single C file (`matvec_shootout.c`) that:

1. Loads one large tensor from the 350M GGUF (e.g., `blk.0.ffn_gate.weight`: 1024 x 4608 = 4.7M weights)
2. Prepares three representations of that tensor:
   - **Q4_0** (original): block-of-32 with f16 scale + 16 nibble bytes = 18 bytes/block
   - **Ternary packed** (converted): 2-bit encoding, Yinsen's blocked-8 layout = 2 bytes per 8 weights
   - **Int8 ternary** (expanded): {-1, 0, +1} as int8, Yinsen's blocked-8 layout = 1 byte per weight
3. Generates a random int8 activation vector (1024 elements)
4. Times each matvec 1000 iterations:
   - **Kernel A:** Scalar Q4_0 dequant-and-dot (reference)
   - **Kernel B:** NEON SDOT Q4_0 (approximating KleidiAI's approach)
   - **Kernel C:** Yinsen's `neon_ternary_matvec_sdot` (2-bit packed + TBL decode)
   - **Kernel D:** Yinsen's `neon_ternary_matvec_blocked8` (2-bit packed + cache-optimized)
   - **Kernel E:** Yinsen's `neon_int8_matvec_blocked8` (int8 expanded, SDOT, blocked)
5. Reports: throughput (GOP/s), effective bandwidth (GB/s), time per matvec (us)
6. Validates correctness: compares all kernels against scalar reference

## Ternary Conversion Strategy

For each Q4_0 block (32 weights, 1 f16 scale):
1. Dequantize: `float_val = (nibble - 8) * scale`
2. Ternarize: `ternary = (nibble == 8) ? 0 : (nibble > 8) ? +1 : -1`

This is the simplest possible mapping: the sign of the quantized integer, with zero preserved. No threshold tuning needed. The 4-bit quantizer already decided which weights are "near zero" (nibble=8) vs positive (9-15) vs negative (0-7).

Quality note: This discards magnitude information. A weight with nibble=15 (value +7) and nibble=9 (value +1) both become +1. But the BLOCK SCALE is preserved — if we store one f16 scale per ternary block, the ternary dot product result gets multiplied by scale, recovering the per-block magnitude. This is effectively 2-bit-with-scale, similar in spirit to Q2_K.

**Critical realization:** We don't need to throw away the scale. Ternary dot with per-block scale is:
```
result += scale * ternary_dot(ternary_weights, activations)
```
Yinsen's kernel computes `ternary_dot` via add/subtract. Then ONE multiply by scale per block-of-32. That's 1 multiply per 32 weights instead of 32 multiplies. This is 97% multiply-free while preserving per-block scaling.

## Key Decisions

1. **Use the 350M model for probing** — smaller, faster iteration, same distribution
2. **Probe on-device** — cross-compile with NDK, deploy via adb push, run via fabric daemon
3. **Include Yinsen's actual kernel source** — inline from `neon/neon_ternary.c` (the kernels are self-contained)
4. **Measure absolute bandwidth** — this tells us how close to the ceiling each approach gets
5. **Don't modify llama.cpp** — this is a standalone probe. Integration comes later if results warrant it.

## Implementation Plan

### Step 1: Build the probe (host)
Single file: `research/trix_fabric/probes/matvec_shootout.c`
- Include Yinsen's NEON kernels (copy the relevant functions, they're header-style)
- Include a minimal Q4_0 dequant kernel
- GGUF loader: reuse the tensor offset logic from our sparsity probe
- Target matvec: N=4608, K=1024 (ffn_gate from blk.0)

### Step 2: Cross-compile for device
```bash
$CC -O2 -march=armv8.2-a+dotprod -o matvec_shootout matvec_shootout.c -lm
adb push matvec_shootout /data/local/tmp/
```

### Step 3: Run on device
```bash
adb shell "/data/local/tmp/matvec_shootout /data/local/tmp/LFM2-350M-Q4_0.gguf"
```

### Step 4: Analyze results
- If ternary+Yinsen > Q4_0+KleidiAI by >20%: pursue full LD_PRELOAD integration
- If ternary+Yinsen ~ Q4_0+KleidiAI (+/- 10%): consider Q2_K as the easier path
- If ternary+Yinsen < Q4_0+KleidiAI: Yinsen's cache tricks don't beat Arm's, focus elsewhere

### Step 5 (conditional): Full ternary model conversion
If Step 4 is positive:
- Write `q4_to_ternary.c`: reads Q4_0 GGUF, writes ternary GGUF (custom tensor type)
- Build LD_PRELOAD shim that intercepts ggml matvec, routes ternary tensors to Yinsen
- Run full inference with ternary model through the fabric daemon
- Compare quality (perplexity) and speed against Q4_0 baseline

## Success Criteria

- [ ] Probe compiles and runs on device
- [ ] All kernel results match scalar reference within tolerance
- [ ] Throughput numbers reported for all 5 kernels
- [ ] Clear winner identified (or clear "they're all at the bandwidth ceiling" result)
- [ ] Numbers are reproducible (low variance across 1000 iterations)

## What We Expect

The A78 at 2.2 GHz with LPDDR4X at ~13 GB/s theoretical:
- Q4_0 matvec (4.5M weights, 0.5 bytes/weight effective): ~2.25 MB per matvec
- Ternary matvec (4.5M weights, 0.25 bytes/weight): ~1.13 MB per matvec
- At 13 GB/s theoretical: Q4_0 floor = ~173us, Ternary floor = ~87us
- Real-world at 76% efficiency: Q4_0 = ~228us, Ternary = ~114us
- **If ternary is ~2x faster on a single matvec, the bandwidth hypothesis is confirmed**

This is the clean cut. One probe file. Five kernels. Real hardware. The numbers decide.
