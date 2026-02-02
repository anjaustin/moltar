# Nodes of Interest: Ternary Quality Collapse

## Node 1: The Scale Misalignment

The Q4_0 block scale was computed to minimize error for a 16-level distribution. After ternarization, the scale is calibrated for the wrong distribution. A weight that was +7 * scale becomes +1 * scale — a 7x magnitude reduction. The scale is an anchor for the original range; with ternary values, it's an anchor for nothing.

Why it matters: This is a mechanical, fixable error. If we recompute per-block scales to minimize reconstruction error for the clamped values, the model might recover significantly.

## Node 2: The Sharp Cliff (7% vs 12%)

The transition from "barely coherent" to "total collapse" happens between 7.1% and 11.9% of weights changed. This is not a gradual degradation. It's a phase transition.

Why it matters: Phase transitions in neural networks usually indicate a tipping point in some critical subcomputation — likely attention or gating. Below the threshold, errors are correctable by the residual stream. Above it, they cascade.

Tension with Node 1: If scale misalignment were the only problem, we'd expect gradual degradation, not a cliff. The cliff suggests something structural.

## Node 3: Autoregressive Amplification

Token N depends on tokens 1..N-1. The 13-level model got the first sentence right but collapsed by token ~30. Each generated token carries forward a small error from the clipped weights. In autoregressive generation, errors compound multiplicatively through the feedback loop.

Why it matters: This explains why the 13-level model starts well and degrades — prompt eval (parallel, single-pass) is more resilient than generation (sequential, self-feeding).

## Node 4: The Kernel Speed is Real

347us ternary vs 764us Q4_0. 2.2x. Confirmed at 3.41 GB/s vs 3.47 GB/s — same bandwidth ceiling, half the data. The blocked-8 prefetch pattern is well-optimized. This speed win is stranded unless we find a way to use it.

Why it matters: The engineering is done. The kernels work. The bottleneck is now purely a weight representation problem.

## Node 5: BitNet's Existence Proof

BitNet b1.58 models work with ternary {-1,0,+1}. They achieve reasonable quality at 2-bit weight density. But they are trained from scratch with ternary-aware quantization. The training process co-adapts all layers to compensate for the limited weight resolution.

Why it matters: Ternary CAN work. It just can't be imposed post-hoc on a model that learned 16-level dynamics.

Tension with Node 1: Even with perfect scale recalibration, a post-hoc ternary model lacks the co-adaptation that training provides. Fixing the scale might help but won't make it equivalent to BitNet.

## Node 6: The Middle Ground — Learned Codebook

Between "naive ternary" and "full Q4_0" there's a spectrum. What if instead of mapping nibbles to fixed {-1,0,+1}, we learn a per-block (or per-tensor) codebook of 4 optimal reconstruction values for a 2-bit code? Like a 2-bit k-means. This preserves the 2-bit bandwidth advantage while giving the values room to be non-uniform.

Why it matters: This is the spline connection. A 4-entry lookup table (00->a, 01->b, 10->c, 11->d) where a,b,c,d are learned per-block. One VLD + one TBL + one SDOT — Yinsen's exact kernel structure.

Tension with Node 5: This still lacks training-time co-adaptation. But it minimizes reconstruction error per block, which is the best we can do post-hoc.

## Node 7: What Q2_K Already Does

llama.cpp has Q2_K: 2-bit quantization with per-block superblocks and fine-grained scales. Q2_K achieves ~2.6 BPW. It exists, it runs in llama.cpp natively, and it already uses optimized NEON kernels via KleidiAI. We never tested it.

Why it matters: Before inventing something new, we should test the thing that already exists. If Q2_K runs on LFM2-350M with acceptable quality, and its speed is comparable to our ternary kernel, the problem is already solved.

## Node 8: The Residual Stream as Error Corrector

In transformer-like architectures, each layer adds its output to the residual stream: x = x + layer(x). This means small errors in layer(x) are diluted by the residual x. The model can tolerate per-layer errors as long as they're below the residual's magnitude.

Why it matters: This explains the cliff — when errors in layer outputs exceed the residual signal, the corrections flip sign and the model enters a chaotic regime. Below the threshold, the residual absorbs errors. Above it, errors dominate.

## Node 9: Selective Layer Ternarization

Not all layers are equally sensitive. Attention Q/K projections compute dot products for softmax — exponential sensitivity. FFN gate projections hit SiLU — sigmoid sensitivity near zero. V/O projections and up/down projections are more linear.

Why it matters: If we could identify which tensors tolerate clamping and which don't, we could ternarize the robust ones (saving bandwidth) while keeping the sensitive ones at Q4_0. A mixed-precision approach.

Tension with Node 4: Mixed precision complicates the kernel dispatch. Each matvec needs to know its tensor's quant format. Not impossible, but not the clean "replace all matvec with Yinsen" story.

## Node 10: Scale Recalibration is Testable Right Now

We don't need to speculate about whether scale recalibration helps. We can write a converter that:
1. For each Q4_0 block, clamp nibbles to [-L, +L]
2. Recompute the block scale to minimize MSE between the original dequantized values and the clamped-then-dequantized values
3. Write the new scale and clamped nibbles back

This is a 30-minute test. If it helps significantly, it confirms Node 1 as the primary failure mode. If it doesn't help, the problem is structural (Node 2/8) and no post-hoc fix will work.

## Node 11: The Dequant-Requant Approach

Instead of clamping nibbles, what if we:
1. Dequantize the Q4_0 block to float32
2. Requantize to ternary with an OPTIMAL scale (chosen to minimize MSE)

The optimal ternary scale for a block is: scale = mean(|float_values| where float_value != 0). The ternary reconstruction is: reconstructed = sign(float) * scale. This is the mathematically correct post-hoc ternarization.

Why it matters: Our current converter clamps nibbles but keeps the ORIGINAL scale. That's doubly wrong — wrong values AND wrong scale. The dequant-requant approach fixes both.

Tension with Node 6: Even optimal ternary has only 3 reconstruction points per block. The codebook approach has 4 points with arbitrary placement. Strictly more expressive for the same 2 bits.
