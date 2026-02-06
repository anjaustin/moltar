# Raw Thoughts: FFN Optimization for LFM2-350M

## Stream of Consciousness

We just tested the obvious stuff - hierarchical FFN, spatial grouping, MoE-style experts, predictive neuron selection. All failed or gave marginal gains. The model is too small and too dense. SwiGLU doesn't create natural sparsity like ReLU would.

But wait - what HAVEN'T we tried? I keep thinking about this in terms of "compute less" but maybe the frame is wrong. What about "compute smarter"? Or "compute earlier"?

Thread optimization gave us 2.6x by doing NOTHING to the algorithm - just using the right cores. That's the biggest win so far. What other "free" optimizations are hiding?

Memory access patterns. We're at 69% DRAM bandwidth. Where's the other 31%? Cache misses? TLB misses? Branch misprediction? The profiler showed barrier sync is tiny (0.19%), so it's not thread coordination.

What about the STRUCTURE of the FFN computation itself? Right now it's:
1. gate = x @ W_gate 
2. up = x @ W_up
3. hidden = gate * sigmoid(gate) * up
4. out = hidden @ W_down

That's 3 matmuls + 2 elementwise. The matmuls dominate. But they're done SEQUENTIALLY. Could we overlap them? Prefetch the next layer's weights while computing?

Thinking about the phone's memory hierarchy:
- L1: 64KB per core, ~200 GB/s
- L2: 256-512KB per cluster
- L3: 1-2MB shared
- DRAM: 4GB, ~13 GB/s

The FFN layer is 27MB. It doesn't fit in any cache. So every token reads 27MB from DRAM. That's unavoidable unless... we change the model?

Distillation. What if we distill LFM2-350M to an even smaller model? Like LFM2-100M? Half the params, half the bandwidth. But probably much worse quality.

Or what about WIDTH vs DEPTH tradeoff? LFM2-350M is 16 layers. What if we had 32 shallower layers? Or 8 deeper layers? The bandwidth would be the same, but the computation pattern changes.

Hmm. What about SHARING weights across layers? "Universal transformer" style. Same weights, different layers. That would be a huge bandwidth win - only load weights once, use 16 times. But requires retraining.

Actually wait - we have shortconv layers too. What's their bandwidth cost? They're much smaller than FFN. Could we lean more on shortconv and less on FFN?

Let me think about what shortconv does. It's a 1D conv with kernel_size=3(maybe 4?), groups=dim. So it's depthwise - very cheap. It mixes adjacent tokens in each channel. The FFN then mixes across channels.

Could we do MORE channel mixing in shortconv and LESS in FFN? Like, make shortconv do some of FFN's job? Would need to add point-wise conv to shortconv...

Oh. That's basically making shortconv into a full MobileNet-style block: depthwise conv + pointwise conv. Then FFN becomes redundant or smaller.

But again, requires retraining. Can't do this post-training.

POST-TRAINING options only:
1. Quantization (done - Q4_0)
2. Pruning (tried - weights aren't sparse enough)
3. Low-rank approximation (tried - full rank)
4. Knowledge distillation (requires training small model)
5. Speculative decoding (requires draft model)
6. Caching / KV optimization (already good for LFM2)
7. Batching (doesn't help latency)
8. Thread optimization (done - 2.6x)
9. Memory prefetching (?)
10. Kernel fusion (?)
11. Custom NEON kernels (?)

The ones with "?" are unexplored.

Memory prefetching: while computing layer N, prefetch layer N+1's weights. The CPU has prefetch instructions. Does llama.cpp use them? Does KleidiAI? 

Kernel fusion: instead of gate, up, swiglu, down as separate ops, fuse them. Reduces memory round-trips. But KleidiAI probably already does this?

Custom NEON kernels: the Yinsen ternary kernel was 2.2x faster than SDOT for ternary. What about a custom Q4_0 kernel? Is KleidiAI already optimal?

Actually - let me question KleidiAI. It handles 99% of MUL_MAT calls but only 77% of MUL_MAT TIME. The remaining 23% is vec_dot for "odd" dimensions. What dimensions are "odd"?

The attention projections might have different dimensions than FFN. Q/K/V are smaller. Maybe those aren't optimized.

What if we FORCE all dimensions to be KleidiAI-friendly? Pad to multiples of 32 or 64 or whatever KleidiAI wants?

Let me think about the memory bandwidth more carefully.

Model = 209 MB
Token rate = 44 tok/s (with 2 threads)
Bandwidth = 209 * 44 = 9.2 GB/s
DRAM peak = 13 GB/s
Utilization = 71%

Where's the other 29%?

Possibilities:
1. We're not actually memory-bound; we're compute-bound
2. Memory accesses aren't perfectly sequential (cache line waste)
3. There's compute between memory loads (the sigmoid, the multiply)
4. Thread scheduling overhead (even with 2 threads)
5. The "vec_dot" fallback path is slower

Actually (1) is unlikely - SDOT gives 4 int8 MACs per cycle, that's absurd throughput. We measured that memory is the bottleneck.

(2) seems likely. Q4_0 packing might not align with cache lines. Or the strided access pattern causes issues.

(3) is minimal - elementwise ops are memory bandwidth limited too, not compute.

(4) possible but we measured barrier sync at 0.19%.

(5) could be significant! 23% of MUL_MAT time in vec_dot.

Let me focus on (5). If we could get vec_dot to 0%, we'd save 23% of MUL_MAT time. MUL_MAT is 94% of total. So 0.23 * 0.94 = 21.6% total speedup. That's ~1.3x.

1.3x on top of our 2.6x = 3.4x total speedup from baseline.

How to eliminate vec_dot? Make all dimensions KleidiAI-compatible. That means understanding what KleidiAI requires.

## Questions Arising

1. What dimensions cause vec_dot fallback instead of KleidiAI?
2. Can we pad weights to eliminate vec_dot?
3. What does KleidiAI actually require (alignment, tile size)?
4. Is there a custom Q4_0 kernel that's faster than KleidiAI?
5. Can we prefetch next layer while computing current?
6. What's the actual cache miss rate during inference?
7. Is there memory access pattern optimization possible?
8. Could we reorder operations to improve locality?

## First Instincts

- The 23% vec_dot is the low-hanging fruit
- Memory prefetching might help but hard to implement
- Custom kernels are a lot of work for uncertain gain
- The "free" wins (like thread count) are probably exhausted
- Real remaining gains might require model changes (retraining)

## What Scares Me

- We might be at the hardware limit already
- Any further optimization might require deep llama.cpp surgery
- KleidiAI is ARM's optimized library - hard to beat
- The 29% "missing" bandwidth might be unfixable overhead

## Wild Ideas

- Run inference on GPU instead (Vulkan)? Phone has Mali-G610
- Use the NPU? MediaTek has APU but llama.cpp doesn't support it
- Hybrid CPU+GPU? Split model across processors
- Memory-mapped IO tricks? Pin model in specific memory region
- Compile model to native code? (Like Apache TVM)
