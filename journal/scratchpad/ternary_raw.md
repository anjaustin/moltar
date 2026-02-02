# Raw Thoughts: Ternary/Sparse Kernels for LFM2 On-Device Inference

## Stream of Consciousness

The user (Tripp) looked at the Q4_0 sparsity data and asked "why don't we do it?" — meaning, why don't we leverage the zero-weight sparsity and near-ternary distribution in the Q4_0 models to use Yinsen's zero-multiply kernels?

The data says: 12% exact zeros, 35% ternary-compatible (|val| <= 1), 55% within |val| <= 2, 70% within |val| <= 3. This is consistent across 350M and 1.2B models. The distribution is a clean bell curve centered on zero.

My first instinct was "you need ternary-trained models for Yinsen's kernels." But Tripp is pushing me to think harder. He's right to push.

What if we DON'T need pure ternary? What if we can exploit the sparsity that's ALREADY THERE in Q4_0?

The key bottleneck on this device is memory bandwidth. We measured it: ~76% of theoretical LPDDR4X bandwidth on the 1.2B. We're reading 663MB per token at 14.85 tok/s. The ALU is waiting for data.

So the question isn't "can we compute faster" — it's "can we read less data per token?"

Wait. That reframes everything.

Option 1: Re-quantize to ternary. Collapse Q4_0 to {-1, 0, +1}. Model goes from 4 bits/weight to 2 bits/weight. That's 50% less data to read. But quality loss would be catastrophic — you're throwing away 65% of the weight information.

Option 2: Sparse-skip the zeros in Q4_0. 12% of weights are zero. Skip the multiply. Saves 12% compute but reads the SAME data (you still have to read the nibble to know it's zero). No bandwidth savings.

Option 3: Something in between. What about a mixed-precision approach? What if the weights near zero could be stored more compactly, and the larger weights stored normally?

Option 4: What if we exploit the STRUCTURE of sparsity? Not random zeros but patterns. If zeros cluster in certain rows or blocks, we could use block-sparse formats.

Option 5: What if we forget about ternary entirely and focus on what Yinsen's NEON kernels actually give us? The blocked weight layouts (block-8, block-16) with prefetch are cache-optimized. KleidiAI uses its own layout. What if Yinsen's layout is better for THIS specific memory subsystem?

Option 6: What about the ghost-stream LDNP kernels? Non-temporal loads bypass cache. On a device with limited L2, this could prevent cache thrashing between activation data (hot) and weight data (cold). This is bandwidth optimization, not compute optimization.

Hmm. Option 6 is interesting. The ghost-stream idea is specifically designed for memory-bandwidth-bound scenarios. That's exactly us.

But wait — KleidiAI presumably already handles cache well. Would our kernels actually beat Arm's own tuned library?

What about a completely different angle: we know the distribution. What if we create a NEW quantization format that's optimized for this specific distribution? Something like Q2_SPARSE or TERNARY_PLUS — where the 35% ternary weights get 2-bit encoding and the remaining 65% get 4-bit, with a bitmap to distinguish them?

That's essentially variable-length encoding for weights. Could reduce model size by ~15-20% while keeping quality close to Q4_0. Less data to read = faster inference on bandwidth-bound hardware.

## Questions Arising

- Is KleidiAI already exploiting any sparsity in Q4_0?
- What's the actual L2 cache size on the A78 cores in the Dimensity 930?
- Could we do a simple experiment: NEON matvec with the same Q4_0 format but Yinsen's blocked layout + prefetch hints, vs KleidiAI?
- Are the zeros uniformly distributed or do they cluster? Block sparsity only helps if there are dense-zero regions.
- What does llama.cpp's Q4_0 matvec kernel actually look like? Is there low-hanging fruit?
- Could we do ternary quantization of JUST the ShortConv layers (10/16 layers) since those are smaller ops and more tolerant of quantization?
- The attention layers (6/16) had HIGHER sparsity (15-17% zeros in V projections). Could we selectively apply sparse kernels to just those tensors?
- What about Q2_K or IQ2 quantization from llama.cpp? Those already exist and are very aggressive. How do they compare to ternary?
- Could we write a custom GGUF tensor type that llama.cpp loads but we intercept via LD_PRELOAD to route through Yinsen's kernels?

## First Instincts

- Pure ternary re-quant from Q4_0 will destroy quality. Don't do it naively.
- The ghost-stream (LDNP) idea has merit independent of weight format — it's about cache policy, not quantization.
- A hybrid format (2-bit for near-zero, 4-bit for others, with bitmap) is the "correct" answer but requires the most engineering.
- The fastest path to an experiment: build a standalone matvec benchmark that compares KleidiAI's Q4_0 kernel vs a Yinsen-style blocked NEON kernel on the same Q4_0 data, on the actual device.
- The LD_PRELOAD approach from the fabric roadmap Phase 4 is the deployment mechanism regardless of which kernel we choose.
- We should probe before we build. Measure KleidiAI's actual throughput on a single matvec on the device, then we know the target to beat.

## What Scares Me

- We might build a beautiful kernel that's 2% faster than KleidiAI, and the engineering cost isn't worth it.
- The quality loss from aggressive quantization might make the model useless.
- We're not experts in KleidiAI's internals — maybe they're already doing everything smart.
- The device has no perf/simpleperf — we can't profile at the instruction level. We'd be tuning blind.
