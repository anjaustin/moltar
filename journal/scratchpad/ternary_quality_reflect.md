# Reflections: Ternary Quality Collapse

## The Three Whys

**Why did the model collapse?**
Because we destroyed weight magnitude while preserving a scale calibrated for the original magnitude.

**Why does that matter so much? Other quantization methods reduce precision too.**
Because we didn't reduce precision — we reduced levels from 16 to 3 (or 5, or 7) without recalibrating the scale. It's as if you resized a photograph to 3 pixels wide but kept the frame the same size. The container (scale) no longer matches the content (ternary values).

**Why is the cliff so sharp?**
Because the residual stream has a finite capacity to absorb error. Below the threshold, layer errors are additive noise on a strong signal. Above it, the error becomes the signal. It's a signal-to-noise transition, not a gradual degradation.

## The Core Insight

**We committed two errors simultaneously and conflated them.**

Error 1: Reducing weight levels (16 -> 3). This is inherently lossy but is the same thing BitNet does successfully. It's survivable with proper calibration.

Error 2: Keeping the original Q4_0 block scales after clamping the values. This is NOT what any working quantization scheme does. Every quantization method — Q4_0, Q2_K, GPTQ, AWQ, BitNet — computes scales that are calibrated for the actual value distribution. We broke this contract.

**We tested "ternary with wrong scales" and concluded "ternary doesn't work." That's not what the experiment showed.** The experiment showed that wrong scales kill the model. We haven't yet tested ternary with RIGHT scales.

## Resolved Tensions

**Node 1 (Scale Misalignment) vs Node 2 (Sharp Cliff):**
Both are real, and they compound. Scale misalignment creates a per-block bias. The sharp cliff is the point where accumulated bias exceeds the residual stream's correction capacity. Fixing the scale should move the cliff — maybe dramatically.

**Node 5 (BitNet works) vs Node 1 (our ternary doesn't):**
BitNet trains with ternary from scratch, which means its scales are learned jointly with the ternary values. We're doing post-hoc ternarization with stale scales. The gap between these two is Node 11 — dequant-requant with optimal scale computation.

**Node 6 (Learned Codebook) vs Node 11 (Dequant-Requant):**
These are on a spectrum of sophistication:
- Level 0 (what we did): Clamp nibbles, keep original scale. **Failed.**
- Level 1 (Node 11): Ternary values, recompute optimal scale per block. **Untested.**
- Level 2 (Node 6): 4-value codebook per block, 2-bit codes. **Untested.**
- Level 3: Per-tensor sensitivity analysis, selective quantization. **Untested.**
- Level 4: Fine-tuning with ternary constraints. **Requires training infrastructure.**

We jumped from Level 0 to "ternary is dead" without testing Level 1. That's the gap.

**Node 7 (Q2_K exists) vs everything else:**
This is the elephant in the room. llama.cpp already has a 2-bit quantization format that handles scale calibration correctly. It uses a two-level scale structure (superblock + sub-block) specifically designed for low-bit precision. We should just... try it. If Q2_K on LFM2-350M produces coherent text, it proves that 2-bit density CAN work on this model with proper calibration, and the question becomes whether Yinsen's kernels can outperform KleidiAI's Q2_K kernels.

## What I Now Understand

The problem is not "ternary can't work." The problem is "we served ternary values with Q4_0 scales."

The Q4_0 format stores: `reconstructed = (nibble - 8) * scale`. When nibble was 15, reconstructed was 7 * scale. When we clamp nibble to 9, reconstructed becomes 1 * scale. We've divided that weight by 7 but the model still expects 7x the magnitude.

The fix is straightforward: after clamping, recompute the scale so the clamped values produce the best approximation of the ORIGINAL float values.

For a ternary block, the optimal scale is the mean absolute value of all non-zero original weights in the block. Then:
- Original weight = +0.35 -> ternary +1 * 0.35-ish -> reconstructed ~ +0.35 (not +0.05)
- Original weight = -0.10 -> ternary -1 * 0.35-ish -> reconstructed ~ -0.35 (this is wrong!)

Wait. That's the real problem with ternary. A weight of -0.10 and a weight of -0.35 both become -1. With any single scale, you can match one of them but not both. The ternary representation has only 3 distinct output values per block: {-scale, 0, +scale}. That's fundamentally 1.58 bits of information.

The question is whether the block scale recalibration gives enough headroom to cross the "residual stream absorption threshold" from Node 8. It won't make ternary perfect — but it might make it functional.

**The testable hypothesis:** Dequant-requant ternarization with optimal scales per block will produce measurably better inference than our naive nibble-clamping. If it crosses the coherence threshold (i.e., produces grammatical, topically-correct text), then ternary is viable for this model and we can deploy the speed win.

**The fast parallel test:** Convert LFM2-350M to Q2_K using llama.cpp's built-in quantizer and run it. If Q2_K works, 2-bit density is proven viable, and we know the ceiling for post-hoc ternary quality.

## The Structure Beneath the Content

What appeared to be "ternary is too aggressive for this model" is actually "we miscalibrated the scale factor." This is an engineering error, not a fundamental limit. The 2.2x speed win is still accessible — we just need to quantize correctly.

The hierarchy of what to test:

1. **Q2_K baseline** (5 minutes) — proves 2-bit density is viable on LFM2-350M
2. **Dequant-requant ternary** (30 minutes) — tests whether proper scale calibration rescues ternary
3. **Learned 2-bit codebook** (hours) — optimal post-hoc 2-bit representation
4. **Per-layer sensitivity** (hours) — find which layers can be ternarized safely

If step 1 fails (Q2_K is incoherent), then 2 bits is genuinely too few for this model and we stop. If step 1 succeeds and step 2 fails, the issue is ternary's 3-value constraint specifically, and we pursue the codebook path. If both succeed, we ship.
