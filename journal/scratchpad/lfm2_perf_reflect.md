# Reflections: LFM2 Performance on Moto G Power 5G

## Core Insight

**We've been solving a solved problem.**

The 47 tok/s baseline at 90% memory efficiency means the current implementation is already near-optimal for this hardware. The compute optimizations we attempted were fundamentally misguided because the workload is memory-bound, not compute-bound.

The real question isn't "how do we go faster?" but rather:
1. "Is faster even necessary?" (Node 8)
2. "What's the power cost of the current speed?" (Node 5)
3. "Are there different tradeoffs worth exploring?" (Node 10, 11)

## Resolved Tensions

### Node 1 (90% efficient) vs Node 3 (need less data)
**Resolution:** These aren't in tension - they're complementary. We're 90% efficient at READING what we have. To go faster, we must have LESS to read. 2-bit quant or sparsity are the only paths.

### Node 5 (power) vs Node 8 (good enough)
**Resolution:** Both can be true. If 47 tok/s is good enough for UX, the question becomes: can we get ~40 tok/s at half the power using little cores? This is a different optimization target entirely.

### Node 4 (not a transformer) vs Node 9 (haven't profiled)
**Resolution:** We must profile before proceeding. Our mental model of LFM2 might be wrong. The state-space structure could have compute-bound components we haven't identified.

## What I Now Understand

### The Moneyball Might Be Power, Not Speed

In baseball Moneyball, the insight was that on-base percentage was undervalued relative to batting average. Everyone optimized for the wrong metric.

Here: **Everyone optimizes for tok/s. The undervalued metric might be tok/watt.**

A mobile device that runs inference 2x longer on battery might be more valuable than one that runs 20% faster.

### The Little Core Opportunity Reframed

Our RAID 0 experiments failed for speed. But what if we reframe?

6 A55 cores at ~2.5 GB/s aggregate bandwidth = ~25 tok/s (estimate)
Power: maybe 1.5W for 6 A55s vs 3W for 2 A78s

That's: 25/1.5 = 16.7 tok/watt (A55) vs 47/3 = 15.7 tok/watt (A78)

**The little cores might win on efficiency!** This needs measurement.

### The Profile Gap is Critical

We've built an entire mental model without ever profiling the actual model. This is the biggest gap. Before any more optimization:

1. Profile LFM2-350M inference end-to-end
2. Identify which ops dominate
3. Measure actual memory bandwidth utilization
4. Find if any ops are compute-bound

### The 2-Bit Question

Q4_0 → Q2 halves memory bandwidth = potentially 2x throughput.

But:
- Does llama.cpp support Q2 for LFM2?
- What's the quality impact?
- Is there a Q3 middle ground?

This is the highest-leverage investigation remaining.

## The Structure Beneath

```
                    ┌─────────────────┐
                    │   PHYSICS WALL  │
                    │   (~10 GB/s)    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  GO FASTER      │           │  GO CHEAPER     │
    │  (less data)    │           │  (same speed)   │
    │                 │           │                 │
    │  - 2-bit quant  │           │  - Little cores │
    │  - Sparsity     │           │  - Lower freq   │
    │  - Smaller model│           │  - Thermal mgmt │
    └─────────────────┘           └─────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   MUST PROFILE  │
                    │   FIRST!        │
                    └─────────────────┘
```

## Remaining Questions

1. **What does LFM2's compute graph actually look like?**
   - Which ops dominate runtime?
   - What's the state size and update cost?

2. **What's the power draw of 2xA78 vs 6xA55?**
   - Need actual measurements
   - Battery impact over sustained use

3. **Does Q2/ternary quant exist for LFM2 in llama.cpp?**
   - If not, how hard to implement?
   - Quality benchmarks needed

4. **What's the thermal envelope?**
   - How long can A78s sustain full speed?
   - Do A55s maintain better sustained throughput?

## The Path Forward

1. **PROFILE** - Before anything else, profile the actual model
2. **MEASURE POWER** - Get actual wattage for different configs
3. **TEST SUSTAINED** - 60-second runs, not 2-second benchmarks
4. **EXPLORE Q2** - The only path to significantly faster
5. **REFRAME SUCCESS** - tok/watt might matter more than tok/s
