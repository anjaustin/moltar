# Nodes of Interest: Ternary/Sparse Kernels for LFM2 On-Device

## Node 1: The Bottleneck Is Memory Bandwidth, Not Compute
We measured: 1.2B model reads ~663MB per token at 14.85 tok/s = ~9.9 GB/s, which is 76% of LPDDR4X theoretical bandwidth. The ALU is waiting for data. Any optimization that doesn't reduce bytes-read-per-token is rearranging deck chairs.
**Why it matters:** This filters out most "faster kernel" ideas immediately. Compute tricks only help if they also reduce data movement.

## Node 2: The Distribution Is Already Peaked at Zero
Q4_0 histogram: 12% exact zeros, 35% ternary-band, 55% within |val|<=2, 70% within |val|<=3. Clean bell curve. Consistent across 350M and 1.2B. This isn't noise — the model's weight landscape genuinely concentrates near zero.
**Why it matters:** The model is TELLING us most of its weights don't carry much information. The question is whether we can exploit that without destroying what information they do carry.

## Node 3: Q4_0 Already Loses Information — How Much More Can We Lose?
The original model is FP32/BF16. Q4_0 quantizes to 16 levels per block-of-32. We already accepted a quality tradeoff going to 4 bits. Going to 2 bits (ternary) is another 2x compression. But the relationship between bits and quality isn't linear — the first bits lost hurt less than the last bits.
**Tension with Node 2:** The 35% that's already in {-1,0,+1} wouldn't lose ANY precision going ternary. The remaining 65% would lose a lot. Could we keep them separate?

## Node 4: Three Levels of Ambition
- **Level A (Conservative):** Keep Q4_0 format, write better NEON kernels with cache-optimized layouts and zero-skipping. Target: 5-15% speedup.
- **Level B (Moderate):** Create a hybrid format — ternary for weights near zero, Q4_0 for the rest. Per-block bitmap. Target: 20-30% speedup through reduced bandwidth.
- **Level C (Aggressive):** Full ternary re-quantization with quality-aware calibration. Target: 2x speedup through 2-bit encoding, at the cost of quality study.
**Why it matters:** We need to pick a level or define a progression through them.

## Node 5: Yinsen's NEON Kernels Are Built for Int8/Ternary, Not Q4_0
All 31 kernels operate on either 2-bit packed ternary or pre-expanded int8. None of them handle the Q4_0 format (f16 scale + packed nibbles per block-of-32). To use Yinsen's kernels, we'd need to either:
(a) Convert Q4_0 to int8 at load time (expands data 2x, worse bandwidth)
(b) Convert Q4_0 to ternary at load time (collapses data, loses quality)
(c) Write NEW kernels that handle Q4_0 natively but with Yinsen's layout/scheduling tricks
(d) Create a new format that bridges the two worlds
**Tension with Node 1:** Options (a) and (c) don't reduce bandwidth. Only (b) and (d) do.

## Node 6: The Ghost-Stream LDNP Idea Is Format-Independent
Yinsen's ghost-stream kernels use non-temporal loads (LDNP) to bypass L2 cache for weight data. This is a cache *policy* optimization, not a data *format* optimization. It could be applied to ANY weight format, including Q4_0. If the A78's L2 is being thrashed by 663MB of weights cycling through it every token, LDNP could help by treating weights as streaming data.
**Why it matters:** This is potentially the easiest path to a real speedup, independent of quantization format.

## Node 7: The LD_PRELOAD Deployment Path
The fabric roadmap Phase 4 says: "Optimized compute libraries via LD_PRELOAD (Yinsen's NEON kernels for matvec)." This is the deployment mechanism — intercept ggml's matvec calls at the dynamic linker level. The application (llama.cpp) never changes.
**Why it matters:** Whatever kernel we build, LD_PRELOAD is how it gets deployed. This constrains the interface: we need to match ggml's matvec function signatures.

## Node 8: We Can't Profile on This Device
No perf, no simpleperf, no ETM, no CoreSight. Production Motorola locks down all profiling. We can only measure wall-clock time for end-to-end inference. We're tuning blind at the instruction level.
**Tension with Node 6:** We can't verify that LDNP actually helps without cache-miss counters. We'd need a proxy: "does wall-clock improve?"

## Node 9: Selective Layer Quantization
The attention V projections have 15-17% zeros vs 12% for FFN layers. The ShortConv layers are smaller operations. Different layers have different sensitivity to quantization. We could apply aggressive quantization only where the model can tolerate it.
**Connection to Node 4:** This is a middle path — not all-or-nothing ternary, but ternary WHERE IT'S SAFE.

## Node 10: The Probe-First Principle
"Probe first, design second." We should measure before building. The minimum viable probe: a standalone matvec benchmark running on the Motorola that compares:
(a) ggml's current Q4_0 matvec (via KleidiAI)
(b) A simple NEON matvec with blocked layout + prefetch on the same Q4_0 data
(c) A ternary matvec on the same data collapsed to {-1,0,+1}
This gives us three numbers. Those numbers tell us whether any of this is worth pursuing.
**Why it matters:** We could spend weeks building a beautiful hybrid kernel only to discover KleidiAI is already at the bandwidth ceiling.

## Node 11: What llama.cpp Already Has — IQ2/Q2_K
llama.cpp already supports very aggressive quantization formats: IQ2_XXS (2.06 bpw), IQ2_XS (2.31 bpw), Q2_K (2.63 bpw). These are heavily optimized with importance matrices and non-uniform quantization. We could just quantize LFM2 to Q2_K and run it with the existing llama.cpp + KleidiAI pipeline. No new kernels needed.
**Tension with Node 5:** If Q2_K already exists and KleidiAI optimizes it, what does a custom ternary kernel add?
**Counter-tension:** Q2_K still does multiplications. Ternary does add/subtract. On THIS device where compute and bandwidth are both tight, zero-multiply could matter.

## Node 12: The Real Question
Is this about making inference faster? Or is it about proving that Yinsen's zero-multiply architecture has practical value on real hardware with real models? Those are different goals with different success criteria.
**Why it matters:** If the goal is speed, Q2_K in llama.cpp might be the pragmatic answer. If the goal is validating Yinsen's approach, we need the custom kernels even if they only match Q2_K's speed.
