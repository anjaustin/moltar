# Nodes of Interest: LFM2 Performance on Moto G Power 5G

## Node 1: We're 90% Memory-Efficient Already
47 tok/s * 190MB = ~9 GB/s actual throughput
A78 measured bandwidth = ~10 GB/s
Efficiency = 90%

**Why it matters:** We can't get more than ~10% improvement through compute optimizations. The ceiling is physics.

## Node 2: We've Been Optimizing the Wrong Thing
Every experiment this session was about compute:
- Shift-add (compute trick) - failed
- RAID 0 little cores (more compute) - failed  
- A78 prefetch pipeline (compute orchestration) - failed

**Why it matters:** We were sharpening a knife to cut water.

## Node 3: The Only Way to Win is Less Data
If bandwidth is the constraint, the only levers are:
- 2-bit quantization (halves bandwidth, ~2x throughput)
- Sparsity (skip zeros, proportional speedup)
- Smaller model (but quality loss)

**Why it matters:** This reframes the entire problem space.

## Node 4: LFM2 is NOT a Transformer
We keep assuming transformer patterns. LFM2 is a state-space model:
- No KV cache
- Has recurrent state
- Different compute graph

**Why it matters:** We haven't profiled the actual model. The bottleneck might not be where we think.

## Node 5: Power Efficiency is Unexplored
2 A78 cores at 2.2GHz vs 6 A55 cores at 2.0GHz
- A55 capacity is 29% of A78
- But power is probably 30-40% per core

What if: 6 A55s give similar throughput at 1/2 the power?

**Why it matters:** Mobile = battery. Watts/token might matter more than tokens/second.

## Node 6: The State Update Question
LFM2's recurrent state:
- How big is it?
- How often updated?
- Is it compute or memory bound?

If state update is compute-bound, little cores COULD help there.

**Why it matters:** We've benchmarked matvec but not the full inference loop.

## Node 7: Thermal Throttling Reality
A78 at full blast will throttle over time.
A55s run cooler, sustain longer.

Sustained throughput over 60 seconds might favor little cores.

**Why it matters:** Benchmarks are short. Real usage is long.

## Node 8: The "Good Enough" Threshold
47 tok/s = 2820 tokens/minute = reading speed+
Is optimization even necessary? What's the user-perceived threshold?

**Why it matters:** Might be solving a non-problem.

## Node 9: We Never Profiled the Real Model
All our benchmarks are synthetic matmul.
We don't know:
- Actual op breakdown of LFM2
- Memory access patterns
- Cache hit rates
- Which layers dominate

**Why it matters:** We might be optimizing 10% of runtime while ignoring 90%.

## Node 10: The 2-Bit Opportunity
Q4_0 = 4 bits per weight
Q2 (ternary-ish) = 2 bits per weight

Half the bandwidth = potentially 2x throughput
BUT: Quality degradation unknown for LFM2

**Why it matters:** This is the only proven path to significant speedup.

## Node 11: Speculative/Batching Options
Instead of faster single-stream:
- Draft model speculation
- Batch multiple requests
- Pipeline across requests

**Why it matters:** Throughput vs latency tradeoff.

## Tensions Between Nodes

- **Node 1 vs Node 3**: We're efficient, but smaller data is still the only lever
- **Node 5 vs Node 8**: Power matters, but maybe speed is already good enough
- **Node 4 vs Node 9**: We assume we know the model, but we haven't profiled it
- **Node 7 vs Node 1**: Sustained vs burst performance

## The Delta (Boundary Cases)

- State update: Is it bandwidth or compute bound? (boundary between Node 4 and Node 6)
- Quality vs speed: What's acceptable degradation? (boundary between Node 3 and Node 10)
- User perception: When does faster stop mattering? (boundary between Node 8 and everything else)
