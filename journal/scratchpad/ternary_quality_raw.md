# Raw Thoughts: Why Ternary Killed the Model

## Stream of Consciousness

We proved the speed. Kernel D (Yinsen's blocked-8 ternary) is 2.2x faster than Q4_0 NEON on real hardware. The bandwidth hypothesis is confirmed: same GB/s, half the data, half the time. 347us vs 764us on a 4608x1024 matvec. This is real.

Then we converted the actual model and it died.

Not gracefully. Not "slightly worse." Dead. Pure ternary {-1,0,+1}: question marks. 5-level {-2..+2}: question marks. 7-level {-3..+3}: question marks. Even 11-level {-5..+5} (only 11.9% of weights touched) produced incoherent word salad. 13-level {-6..+6} (7.1% changed) got "Paris" right, started a real sentence, then collapsed into "Eiffiffiffiffiff..."

The cliff is between 7.1% and 11.9% of weights perturbed. That's an absurdly sharp boundary. Change 7% and the model wobbles. Change 12% and it's gone.

Why? What's actually happening?

First instinct: the extreme-value weights are doing outsized work. In a bell curve centered on zero, the tails carry the most information in an information-theoretic sense — they're the least probable values, so they encode the most surprise. Clipping them destroys the highest-information weights first.

But wait. Q4_0 only has 16 levels. The "tails" are values -8 and +7 (nibbles 0 and 15). These are already coarsely quantized. They're not precise — they're the best the Q4_0 quantizer could do for the largest-magnitude weights. And we're clipping them further.

Second thought: it's not about individual weights. It's about the aggregate effect across a layer. A matvec sums K=1024 products. If you clip 7% of those products, you're introducing a systematic bias in the sum. The sum shifts. Across 16 layers, the shift compounds.

But that doesn't explain the sharp cliff. 7% clipped = barely alive, 12% = dead. The transition is nonlinear.

Third thought: the softmax in attention is exponentially sensitive. If the QK^T dot products shift by even a small amount due to weight clipping, the exponential in softmax can dramatically redirect attention. One wrong attention pattern cascades through all subsequent layers.

But the ShortConv layers don't have softmax. And there are 10 of those vs 6 attention layers. If the model dies from softmax sensitivity, we'd expect the 6 attention layers to be the bottleneck and the 10 ShortConv layers to be more robust.

Fourth thought: SiLU gating in the FFN. The FFN is SwiGLU: output = SiLU(gate(x)) * up(x). SiLU is sigmoid(x) * x. If gate(x) shifts because the gate projection weights were clipped, the sigmoid crosses its midpoint at a different place. Sigmoid is steep at x=0 — small shifts in input cause large shifts in output. Then this gets multiplied by up(x). Two clipped projections multiplied together.

Fifth thought: the block scale is preserved, but it was calibrated for the full 16-level distribution. When we clamp to fewer levels, the scale is now wrong. A block that had weights spanning [-8,+7] * scale now only spans [-1,+1] * scale (for ternary). The scale was chosen to minimize quantization error for the original distribution. For ternary, it's too large — the few +1 values are being multiplied by a scale that was meant for +7.

Wait. That's it. That might be THE problem.

The scale factor is a single number per 32-weight block. It was computed during quantization as: scale = max(abs(original_floats)) / 7 (approximately). So for a block where the largest weight was 0.35, scale = 0.05. A nibble of 15 means +7 * 0.05 = 0.35. A nibble of 9 means +1 * 0.05 = 0.05.

When we ternarize, that nibble-15 weight becomes nibble-9 (+1). So instead of 0.35, it's now 0.05. We've destroyed 86% of that weight's magnitude. And the scale is still 0.05 because we preserved it.

For the nibble-8 (zero) weights, nothing changes. For nibble-9 (+1), nothing changes. For everything else, the magnitude is crushed.

The scale was the anchor for the ORIGINAL distribution. After ternarization, the scale is misaligned — it's too large for what the ternary values actually need. The ternary weights that should be +1 are represented as +1 * (scale_for_original_range), which is correct for the sign but incorrect for the magnitude.

Actually, no. The ternary +1 is encoded as nibble 9, which llama.cpp reads as (9-8)=1, then multiplies by scale. So the weight becomes 1 * scale = scale. The original +7 weight was 7 * scale. We've replaced 7*scale with 1*scale — a 7x reduction. That's massive.

But here's the thing: BitNet b1.58 works. Those models use ternary {-1,0,+1} with per-block (or per-tensor) scales. The difference is: they were TRAINED that way. The training process learned scale values that are calibrated for ternary, and the model's other weights compensated.

Our model was trained with 16 levels of magnitude resolution. The inter-layer dynamics, the attention patterns, the gating functions — everything learned to expect fine-grained magnitude from the weights. Ripping that away post-hoc breaks the implicit contracts between layers.

Sixth thought: what about the 13-level result? Only 7.1% of weights changed (the extreme tails: nibbles 0,1 and 14,15). It got the first sentence right but then collapsed. This suggests the first few tokens of generation use prompt context heavily (prompt eval is a matmul, not a matvec, and the errors average out more). But as generation continues and the model relies on its own outputs, the small errors compound. Token N depends on tokens 1..N-1, each of which was generated with slightly wrong logits. By token 30-40, the accumulated drift causes repetition collapse.

This is the autoregressive amplification problem. Errors don't add — they multiply through the feedback loop.

## Questions Arising

- Is the scale misalignment the primary cause, or is it the loss of inter-weight magnitude relationships?
- Could we recompute optimal scales for the ternary distribution per block?
- Would that make ternary work? Or is the fundamental issue that 3 levels can't represent 16 levels of structure?
- BitNet trains with ternary from scratch. What if we fine-tune just a few layers to adapt?
- Are some layers more sensitive than others? Could we selectively ternarize only robust layers?
- The 13-level result started coherent — does this mean per-block scale recalibration could push the threshold further?
- What about the approach where we don't ternarize but instead use Yinsen's kernel structure (blocking, prefetch, SDOT) on the native Q4_0 format?

## First Instincts

- The scale misalignment is the core mechanical failure
- But even with perfect scales, 3 levels probably can't carry a model trained for 16 levels
- The path forward is NOT post-hoc ternarization of existing models
- The speed win (2.2x) is real but inaccessible without ternary-native models
- There may be a middle path: use Yinsen's kernel architecture (cache blocking, prefetch patterns) to accelerate the existing Q4_0 format
- Or: wait for ternary-native LFM models and the kernels are ready
- Or: the spline idea — a learned nonlinear mapping from 2-bit codes to optimal reconstruction values
