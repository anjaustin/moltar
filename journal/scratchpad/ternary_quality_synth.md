# Synthesis: Rescuing Ternary Inference Quality

## Core Insight

We tested "ternary with misaligned scales" and declared ternary dead. The experiment was flawed. We clamped Q4_0 nibbles but preserved scales that were calibrated for the original 16-level distribution — producing systematic magnitude errors of up to 7x per weight. The model collapsed from scale misalignment, not from ternary's information capacity.

Two things must be fixed simultaneously:
1. The weight values (ternary sign selection)
2. The block scale (recalibrated for the ternary distribution)

## Architecture: Staged Falsification

Three tests, in order of effort. Each one gates the next. If a test fails, we stop.

### Test 1: Q2_K Existence Proof (5 minutes)

**Purpose:** Prove that 2-bit weight density is viable on LFM2-350M with properly calibrated scales.

llama.cpp has `llama-quantize` which converts F32/F16 GGUF to Q2_K (2.6 BPW). Q2_K uses a two-level scale structure specifically designed for low-bit precision. If Q2_K produces coherent text on LFM2-350M, then 2-bit density works and the issue is purely our scale calibration.

**Method:**
```bash
# We already have the Q4_0 GGUF. Requantize to Q2_K.
# llama-quantize reads any GGUF and requantizes.
./llama-quantize LFM2-350M-Q4_0.gguf LFM2-350M-Q2_K.gguf Q2_K
adb push LFM2-350M-Q2_K.gguf /data/local/tmp/
# Run same test prompt
adb shell "... ./llama-completion -m LFM2-350M-Q2_K.gguf -p 'What is the capital of France?' ..."
```

**Gate:** If Q2_K produces coherent text -> proceed to Test 2. If Q2_K is incoherent -> 2 bits is too few for this model at 350M parameters, stop.

**Note:** llama-quantize typically wants the original F16/F32 model to requantize, not a Q4_0. Quantizing from Q4_0 to Q2_K would double-quantize. We may need to use the original safetensors or find a pre-made Q2_K. If unavailable, this test is blocked and we skip to Test 2.

### Test 2: Dequant-Requant Ternary (30 minutes)

**Purpose:** Test ternarization with mathematically optimal per-block scales.

**Method:** Build `q4_to_ternary_v2.c` that:

1. For each Q4_0 block (32 weights):
   a. Dequantize all 32 weights to float32: `float_i = (nibble_i - 8) * original_scale`
   b. Compute ternary signs: `sign_i = (float_i > 0) ? +1 : (float_i < 0) ? -1 : 0`
   c. Compute optimal ternary scale: `new_scale = sum(|float_i| where sign_i != 0) / count(sign_i != 0)`
      - This minimizes MSE: the optimal scale for ternary reconstruction is the mean absolute value of non-zero weights
   d. Encode back to Q4_0 format:
      - Write `new_scale` as fp16 in the block header
      - Write ternary nibbles: -1 -> 7, 0 -> 8, +1 -> 9

The key difference from v1: the block scale is RECALIBRATED. A block that had weights spanning [-0.35, +0.35] with original_scale=0.05 will now have new_scale=0.20 (the mean |weight| of non-zero entries). Ternary +1 * 0.20 = 0.20, which is much closer to the original weights than +1 * 0.05 = 0.05.

**What this fixes:**
- v1: ternary +1 reconstructed as 1 * old_scale = 0.05 (when original was 0.35). Error: 0.30.
- v2: ternary +1 reconstructed as 1 * new_scale = 0.20 (when original was 0.35). Error: 0.15.

The error is still there (ternary can't distinguish 0.10 from 0.35), but it's 2x smaller on average.

**Gate:** If dequant-requant ternary produces coherent text -> the 2.2x speed win is accessible. Proceed to integration. If incoherent -> ternary's 3-value constraint is fundamentally insufficient for this model. Proceed to Test 3.

### Test 3: Learned 2-Bit Codebook (hours, only if Test 2 fails)

**Purpose:** Test whether 4 learned values per block (instead of fixed {-1,0,+1}) can carry the model.

**Method:** For each Q4_0 block:
1. Dequantize all 32 weights to float32
2. Run k-means with k=4 on the float values (or use quantile-based initialization)
3. Assign each weight to its nearest centroid -> 2-bit code (00, 01, 10, 11)
4. Store: 4 fp16 centroid values (8 bytes) + 32 2-bit codes (8 bytes) = 16 bytes per block
   - vs Q4_0's 18 bytes per block — actually SMALLER

This can't be encoded in Q4_0 format (different block structure), so it would require a custom GGUF type or an LD_PRELOAD shim. More complex but tests the fundamental question: can 2 bits per weight carry this model with optimal value placement?

**Gate:** If coherent -> the 2-bit density works, build the integration. If incoherent -> 2 bits is fundamentally too few for post-hoc quantization of LFM2-350M. Accept the Q4_0 fabric speedup (+41%) and wait for natively ternary models.

## Key Decisions

1. **Test Q2_K first** — it's free, it's already implemented, and it answers the density question.
2. **Dequant-requant is the primary bet** — it fixes the exact mechanical error we identified.
3. **The codebook path is insurance** — only needed if ternary's 3-value constraint is the problem.
4. **We do NOT pursue training-time solutions** — we are a hardware fabric, not a training pipeline.
5. **All tests use unmodified llama.cpp** — the fabric principle: applications run unaware.

## Implementation: Dequant-Requant Converter

`research/trix_fabric/tools/q4_to_ternary_v2.c`

Key difference from v1:
```c
// v1 (what failed): clamp nibble, keep original scale
uint8_t new_nibble = clamp(old_nibble, 8-L, 8+L);
// scale unchanged

// v2 (the fix): dequant to float, compute sign, compute optimal scale
float original = (old_nibble - 8) * original_scale;
int8_t sign = (original > 0) ? +1 : (original < 0) ? -1 : 0;
// After processing all 32 weights in block:
float new_scale = sum_abs_nonzero / count_nonzero;
// Convert new_scale to fp16 and write to block header
// Write sign as nibble: +1->9, 0->8, -1->7
```

## Success Criteria

- [ ] Q2_K existence proof: coherent text from LFM2-350M-Q2_K (or determination that Q2_K isn't available for this model)
- [ ] Dequant-requant converter built and tested
- [ ] Ternary v2 model produces grammatical, topically-correct responses for at least 64 tokens
- [ ] Speed measurement: confirm 44+ tok/s (same as Q4_0 — format is identical, just different values)
- [ ] If successful: per-block MSE comparison between v1 (naive) and v2 (recalibrated)

## What We Expect

The recalibrated ternary model will be significantly better than v1 but still worse than Q4_0. The question is whether it crosses the coherence threshold. Based on the 13-level result (7.1% of weights changed, coherent for ~30 tokens), there IS a coherence threshold, and it's reachable.

The optimal ternary scale should reduce per-block MSE by roughly 3-4x compared to the naive approach (from ~7x magnitude errors on tail weights to ~2x magnitude errors on average). Whether this 3-4x error reduction is enough to cross from "Eiffiffiffiff" to "Eiffel Tower" is the open question.

If it works, the path to production is:
1. Convert model on host with `q4_to_ternary_v2`
2. Deploy to device (same file size, same format)
3. llama.cpp reads it natively at 44 tok/s
4. Future: LD_PRELOAD shim routes ternary tensors to Yinsen's blocked-8 kernel for 2.2x speedup

If it doesn't work, we accept:
- The fabric's +41% speedup from core pinning + pre-faulting is the product
- The 2.2x ternary kernel speedup waits for natively ternary models
- The shootout, converter, and kernels are assets ready for that day
