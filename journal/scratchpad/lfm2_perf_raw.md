# Raw Thoughts: LFM2 Performance on Moto G Power 5G

## Stream of Consciousness

We've been chasing the wrong rabbit. We started with "shift-add is multiplication-free, therefore faster" and proved it wrong - SDOT wins, the hardware already does small-integer multiply efficiently. Then we tried "little cores in RAID 0" and hit the memory controller ceiling. Then "A78 prefetch for A55" and it made things WORSE due to memory contention.

The whole session has been about compute tricks on a memory-bound problem. We're at 47 tok/s with 2 A78 threads, and we keep trying to squeeze more compute out of cores that are already waiting for memory.

What actually matters? The model is 190MB. Each token needs to read most of those weights. At 47 tok/s and 190MB, that's... 47 * 190 = ~9 GB/s. And we measured the A78 can do about 10 GB/s. We're already at 90% memory efficiency!

So why are we trying to make compute faster when we're memory bound?

The real levers are:
1. Read less data (smaller quant, sparsity)
2. Read data faster (can't - hardware limit)
3. Read data fewer times (caching, but model >> cache)

Wait. What about the STATE? LFM2 is a state-space model, not a transformer. It doesn't have KV cache - it has recurrent state. How big is that state? Is the state update the bottleneck or is it the projection matrices?

Also - we tested pure matmul benchmarks. But inference has other ops: RMSNorm, activations, state updates. Are those hiding somewhere?

What's the actual profile of LFM2 inference? We've been benchmarking matvec in isolation but never profiled the real model.

The 47 tok/s baseline - is that actually good? What's the theoretical max? If weights are 190MB and bandwidth is 10 GB/s... that's 52 tokens/second theoretical maximum if we could read weights perfectly. 47/52 = 90% efficiency. That's REALLY good already.

So... are we done? Is 47 tok/s actually near-optimal?

But wait - the user mentioned energy. We're running 2 A78 cores at full blast. What about using 6 A55 cores at lower power? Same throughput but less battery? The A55s have "capacity 299" vs A78's "1024". That's 3.4x less capable but also probably 3x less power.

Also - thermal. If the A78s throttle, the A55s might sustain better.

## Questions Arising

- What's LFM2's actual compute graph? What ops dominate?
- How big is the recurrent state? Is state update bandwidth-bound too?
- Is 47 tok/s actually near theoretical maximum?
- What's the power consumption of 2xA78 vs 6xA55?
- Are there any ops that are compute-bound where little cores could help?
- What does the Dimensity 7020's power management look like?
- Could we trade latency for efficiency (batching)?

## First Instincts

- We're probably already near optimal for throughput
- The real opportunity might be power efficiency, not speed
- We haven't profiled the actual model
- State-space models have different bottlenecks than transformers
- 2-bit quant could double throughput but quality loss?
- The "Moneyball" might be: accept 47 tok/s is good, optimize for watts/token

## What Scares Me

- We've burned a lot of time on compute optimizations for a memory-bound problem
- Maybe the answer is "you can't beat physics"
- The device might already be doing the optimal thing
