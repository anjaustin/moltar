# Reflections: FFN Optimization for LFM2-350M

## The Core Tension

Looking at the nodes, there's a fundamental tension:

**The easy optimizations are done. The remaining ones require either:**
1. Deep runtime surgery (Nodes 1, 3, 7) - high effort, uncertain gain
2. Hardware access (Nodes 4, 11) - blocked by root/driver issues  
3. Model changes (Nodes 13, 15) - requires training

We've exhausted the "configure differently" space. What remains is "build differently" or "use different hardware."

---

## Resolved Tensions

### Vec_Dot (Node 1) vs Bandwidth Ceiling (Node 2)

After reflection: These are likely THE SAME problem. Vec_dot fallback happens for "odd" dimensions. These odd dimensions probably also have poor cache behavior.

**Resolution:** Focus on understanding WHY vec_dot is triggered. Fixing that likely fixes both issues.

### Prefetch (Node 3) vs Kernel Fusion (Node 7)

These seem complementary but are actually competing. Prefetch assumes we're memory-bound and need to hide latency. Fusion assumes we're overhead-bound and need fewer kernel launches.

**Resolution:** Given we're at 71% bandwidth, we're NOT at memory saturation. The issue isn't latency hiding - it's something else eating 29%. Prefetch probably won't help. Focus on understanding the 29%.

### Thread Insight (Node 9) vs Batch Size (Node 12)

2 threads beat 8 threads. But batch=8 would want 8 threads to parallelize across sequences.

**Resolution:** For single-user latency (our goal), 2 threads is optimal. Batching doesn't help latency. These aren't in tension - they're different use cases.

---

## Key Insights

### Insight 1: The 29% Gap is Suspicious

71% bandwidth utilization feels low for a memory-bound workload. On desktop GPUs, we'd expect 85-95%. What's different about mobile?

Possibilities:
- **Memory controller overhead**: Mobile LPDDR4X has different characteristics than desktop DDR4/5
- **Mixed precision overhead**: Dequantizing Q4_0 to FP32 for compute takes cycles
- **Activation functions**: SiLU (sigmoid * x) isn't free
- **Small tile sizes**: Mobile caches are small, forcing more DRAM round-trips

**New hypothesis:** The 29% isn't "missing" - it's spent on dequantization and activation compute between memory loads.

### Insight 2: The Dimension Alignment is Likely the Vec_Dot Cause

4608 = 576 * 8 = 18 * 256. It's not a power of 2, not a multiple of 32 (NEON vector), but IS a multiple of 256.

KleidiAI probably tiles at 32, 64, or 128. 4608 / 128 = 36. 4608 / 64 = 72. 4608 / 32 = 144. All clean!

So why vec_dot fallback? Let me reconsider...

The fallback might be for SMALL matmuls, not big ones. What small matmuls exist?
- Embedding lookup: [batch, 1] -> [batch, 1024]
- Output projection: [batch, 1024] -> [batch, vocab]
- Attention: various shapes depending on sequence length

The "11 vec_dot calls" (from earlier profiling) vs "4784 KleidiAI calls" suggests it's rare cases, not the main FFN.

**New hypothesis:** Vec_dot handles edge cases (batch=1? odd dimensions in attention?). It's not the FFN.

### Insight 3: Root Access is the Highest-ROI Next Step

Looking at all nodes, the ROOT question (Node 11) has the best effort/reward ratio:
- Potential gain: 1.5x (from frequency)
- Effort: Fix Magisk shell access (1 hour?)
- Risk: Low (reversible)

Compare to:
- Runtime surgery: weeks of work, uncertain gain
- Different runtime: days of work, might be slower
- Model surgery: requires training infrastructure

**Decision:** Focus on getting root working before any other optimization.

### Insight 4: The "Free" Optimizations Pattern

Our best win (2.6x) came from understanding the hardware (big.LITTLE) and configuring appropriately. No code changes.

Are there more "free" wins hiding?
- **CPU affinity**: We're using 2 threads, but ARE they on big cores? Or does the scheduler put them on LITTLE?
- **Memory placement**: Is the model in the best memory region?
- **Process priority**: Is llama-completion getting CPU priority?

**New experiment idea:** Use `taskset` to force execution on CPU 6-7 (big cores only).

### Insight 5: The Mobile vs Desktop Gap

We keep comparing to desktop ML performance, but mobile is fundamentally different:
- No dedicated ML accelerator (like Apple's Neural Engine)
- Shared memory (CPU and GPU fight for bandwidth)
- Thermal constraints (can't sustain max frequency)
- Power constraints (battery life matters)

**Reframe:** 44 tok/s on a $200 phone is actually impressive. We might be closer to optimal than we think.

---

## What I Now Understand

### The Optimization Landscape

```
                    EFFORT
                    Low ────────────────────────── High
                    │                               │
         ┌──────────┼───────────────────────────────┤
    High │ ★ ROOT   │                    Model      │
         │   ACCESS │                    Surgery    │
  IMPACT │          │                               │
         ├──────────┼───────────────────────────────┤
    Med  │ CPU      │ Vec_dot    Kernel   Runtime   │
         │ Affinity │ Fix        Fusion   Switch    │
         ├──────────┼───────────────────────────────┤
    Low  │ Tuning   │ Prefetch   Cache    Custom    │
         │          │            Opt      Kernel    │
         └──────────┴───────────────────────────────┘
```

**Priority order:**
1. Root access (high impact, low effort) - Get Magisk shell working
2. CPU affinity (medium impact, low effort) - Pin to big cores
3. Vec_dot investigation (medium impact, medium effort) - Understand and fix
4. Everything else is diminishing returns

### The Real Bottleneck Chain

```
Theoretical max: 64 tok/s (at 13 GB/s DRAM, 209 MB model)
     │
     ├─ 29% lost to overhead (dequant, activations, etc)
     │  → Achievable: 45 tok/s
     │
     ├─ Current: 44 tok/s (2 threads)
     │  → We're at 98% of achievable!
     │
     ├─ With root + max frequency: ~66 tok/s?
     │  → Exceeds theoretical? Means we're compute bound at higher freq
     │
     └─ Real limit: ~50-55 tok/s (bandwidth + overhead)
```

**Key realization:** We're probably at 98% of what's achievable without root. The remaining gains are in hardware control (frequency, affinity).

---

## Remaining Questions

1. Can we verify CPU affinity (are threads actually on big cores)?
2. What exactly triggers vec_dot fallback?
3. Can Magisk shell access be fixed by granting permission in app?
4. What would a "tuned" Android power profile give us?
5. Is 50-55 tok/s the real ceiling, or is there more?

---

## The Synthesis Direction

The synthesis should focus on:
1. **Immediate action**: Fix root access for frequency control
2. **Verification**: Confirm we're on big cores, measure actual frequencies during inference  
3. **Ceiling estimation**: Calculate true maximum given hardware constraints
4. **Documentation**: Record what works and what doesn't for future reference

The FFN optimization quest has reached a natural conclusion: the model is dense, the runtime is efficient, and the hardware is the limit. Further gains require either different hardware or model retraining.
