# Raw Thoughts: W4A8 Integer Path Optimization for LFM2 on A78

## Stream of Consciousness

So the profiling reveals the actual situation. MUL_MAT is 96% of everything. Within that:
- KleidiAI handles 99% of the calls (828/837)
- KleidiAI takes 77% of MUL_MAT time
- The F32→Q8 quantization is basically FREE (0.02%)
- The standard vec_dot path only catches 9 edge cases

This means... the question Tripp asked about "integer math to skip multiplies" is interesting but maybe misframed? The SDOT instruction already does 4 int8 multiplies per cycle. The multiplies aren't the bottleneck — DRAM bandwidth is.

But wait. Let me think about what Yinsen actually does. Ternary sparse:
- Weights are {-1, 0, +1}
- Zeros are SKIPPED entirely — no load, no compute
- Non-zeros are just add/subtract

The win isn't "avoid multiply" — SDOT makes multiplies essentially free. The win is:
1. Skip loading zero weights (bandwidth savings)
2. Skip the entire operation for zero rows/columns (compute savings)
3. Simpler decode (no block scale, just sign bit)

But we proved in Phase 5 that post-hoc ternarization kills quality. The model wasn't trained that way.

What about the CURRENT Q4_0 path? Each weight is 4 bits (-8 to +7 after offset). The distribution shows ~12% exact zeros. That's not enough sparsity to exploit.

The real question: what CAN we do with the existing Q4_0 model and KleidiAI?

Looking at the KleidiAI code:
- It takes F32 activations, packs them into some internal format
- Then runs its optimized NEON kernel
- The "packing" (line 401 in kleidiai.cpp) is where F32→internal happens

Wait — KleidiAI already does implicit activation quantization in the pack step! It's not doing F32 × Q4 directly. It must be converting to int8 or similar for SDOT.

So the path is:
1. F32 activations → KleidiAI LHS pack (quantize to int8-ish)
2. Pre-packed Q4_0 weights (already in device memory)
3. NEON SDOT kernel
4. Accumulate → F32 output

The F32→int8 quantization is happening inside KleidiAI's pack function, which is where the 76.75% of time is spent. But that includes the ACTUAL COMPUTE too, not just quantization.

So what could we do differently?

Option A: Keep activations in int8 between layers
- Problem: RMS_NORM and GLU need float precision
- Could we do "fused" norm+quant? 
- RMS_NORM = x / sqrt(mean(x^2) + eps)... hard to keep in int8

Option B: Lower quantization (IQ2, IQ3)
- Reduces bytes loaded per weight
- But requires different kernels, may not have KleidiAI support

Option C: Speculative decoding
- Draft model generates candidates
- Big model verifies in batch
- Amortizes memory loads across multiple tokens
- Liquid AI doesn't have a smaller draft model... unless we use LFM2-350M to draft for LFM2-1.2B?

Option D: Weight pruning / sparsity
- If we could identify near-zero weights and skip them...
- But Q4_0 is already quantized — the zeros are baked in
- Would need to re-train with structured sparsity

Option E: Context caching / KV optimization
- Not about MUL_MAT, more about memory management
- Already explored in fabric work

Option F: Look inside KleidiAI's pack function
- Is there overhead we could eliminate?
- Unclear without deeper instrumentation

## Questions Arising

1. What exactly does KleidiAI's `pack_func_ex` do? How expensive is it vs the kernel?
2. Can we keep activations quantized between layers somehow?
3. Would speculative decoding with 350M drafting for 1.2B work? What's the acceptance rate?
4. Are there any Liquid AI models with built-in sparsity?
5. What's the theoretical minimum bytes-per-token for LFM2-350M? Are we close?
6. Could we fuse RMS_NORM into the previous layer's dequantization?

## First Instincts

My gut says:
- The F32→int8 quantization IS happening, just inside KleidiAI
- We're probably already near-optimal for Q4_0 on this hardware
- The real gains would come from smaller quantization or native sparsity
- Speculative decoding might be the most practical next step

What scares me:
- We might be at the wall — KleidiAI is Arm's best shot at this
- Without native sparse models, Yinsen's advantage can't be used
- The DRAM bottleneck is fundamental physics

What's probably wrong with my first instinct:
- There might be overhead in KleidiAI we haven't measured
- The pack function might be doing more work than necessary
- Maybe there's a way to keep int8 activations for some ops
