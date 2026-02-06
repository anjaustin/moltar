# Nodes of Interest: W4A8 Integer Path Optimization

## Node 1: The Multiply Myth
The original question was "can we use integer math to avoid multiplying?" But SDOT already does 4 int8 multiplies per cycle — multiplies are essentially free on A78. The bottleneck is memory bandwidth, not ALU throughput.
**Why it matters:** Reframes the entire problem. We're not compute-bound.

## Node 2: KleidiAI Already Quantizes Activations
The profiling shows F32→Q8 quantization takes 0.02% of MUL_MAT time. But KleidiAI's `pack_func_ex` is doing implicit quantization inside its 76.75% chunk. The activation quantization IS happening — it's just fused into KleidiAI's internal path.
**Why it matters:** We can't optimize what's already been optimized by Arm.

## Node 3: The 76.75% Black Box
KleidiAI takes 76.75% of MUL_MAT time but we don't know the breakdown inside. Is it:
- LHS packing/quantization?
- Actual SDOT compute?
- Memory stalls waiting for weight loads?
**Tension with Node 2:** If packing is expensive, maybe we CAN optimize. If it's memory stalls, we can't.

## Node 4: Bytes-Per-Token is the True Metric
For GEMV (single token): we load the entire weight matrix (204 MB for 350M Q4_0) per token. At 6.6 GB/s effective bandwidth, that's ~31ms minimum per token just for weight loading. We're measuring ~17ms/tok — which means we're doing better than streaming everything?
**Why it matters:** Either KleidiAI has good caching or weights are partially reused.

## Node 5: The RMS_NORM Barrier
Even if we kept activations in int8, RMS_NORM requires float: `x / sqrt(mean(x^2) + eps)`. The square and sqrt break integer precision. Every layer needs float somewhere.
**Tension with Node 2:** Can't keep pure integer pipeline end-to-end.

## Node 6: Speculative Decoding as Bandwidth Amortization
If LFM2-350M drafts for LFM2-1.2B:
- Draft generates N candidates (cheap — 350M is fast)
- Verify runs N tokens through 1.2B in one batch
- If acceptance rate is high, we amortize 1.2B's memory loads across N tokens
**Why it matters:** This is a SOFTWARE solution to a HARDWARE bottleneck.

## Node 7: Native Sparsity vs Post-Hoc
Post-hoc ternarization of Q4_0 = garbage output (proven Phase 5). But if Liquid AI released a natively sparse model, Yinsen's 2.2x kernel could apply. Current LFM2 has 12% zeros — not enough structured sparsity.
**Why it matters:** The optimization exists but the model doesn't.

## Node 8: The Weight Repack at Load Time
KleidiAI uses `CPU_KLEIDIAI model buffer` (154 MB for 350M). This is pre-repacked weights in KleidiAI's optimal layout. The repacking happens at load time, not inference time.
**Why it matters:** No per-inference weight transformation cost.

## Node 9: GEMM vs GEMV Split
828 calls through KleidiAI (GEMV for generation), 9 calls through standard path (GEMM for prompt). KleidiAI's GEMV is optimized; GEMM less so. The prompt eval is slower per-token (7ms vs ~17ms/tok) but that's batch effects.
**Why it matters:** Different code paths for different batch sizes.

## Node 10: Theoretical Bandwidth Limit
LFM2-350M Q4_0: 204 MB of weights
LPDDR4X theoretical: 13 GB/s
Minimum time to stream all weights: 204MB / 13 GB/s = 15.7 ms
Achieved: ~17 ms/tok for generation
**We're at 92% of theoretical bandwidth utilization.**
**Why it matters:** There's almost no room left to optimize at the memory level.

## Node 11: The F16 mmproj Discovery
For VL, we found Q8 mmproj was broken but F16 worked. The SigLIP2 vision encoder has dimensions not divisible by 32. This suggests quantization can break specific tensor shapes.
**Why it matters:** Some operations may require specific precision/alignment.

## Node 12: The CfC Controller is Irrelevant Here
The fabric daemon's CfC controller runs at 100Hz with ~7us/step — completely decoupled from inference. It optimizes scheduling, not compute. Wrong lever for this problem.
**Why it matters:** Don't confuse system optimization with kernel optimization.

## Node 13: OpenMP Threading in KleidiAI
KleidiAI uses OpenMP (`GGML_USE_OPENMP`). The 2-thread configuration on A78 big cores is optimal for our setup. More threads = more cache contention.
**Why it matters:** Thread scaling is already tuned.

## Node 14: The Attention Layers Use Standard Path?
LFM2 has 6 attention layers (n_head_kv != 0) and 10 recurrent layers. The 9 standard path calls might be the attention Q/K/V projections during prompt eval (6 layers × ~1.5 batched calls)?
**Why it matters:** Understanding which ops hit which path.

