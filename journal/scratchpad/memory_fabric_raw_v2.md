# RAW v2: Memory Fabric - Going Wild

## Stream of Consciousness

Wait. Stop. I'm being too conventional.

We discovered uncached memory is faster for large random access. But I immediately jumped to "use uncached for KV cache." That's the OBVIOUS answer. What's the NON-OBVIOUS answer?

What if the whole framing is wrong?

The question was: "Can we create a fabric that unifies the paths in a way that eliminates cache coherency overhead?"

We found: Yes, use uncached memory.

But that's not a FABRIC. That's just a different memory type. A fabric implies COORDINATION. ORCHESTRATION. Something ACTIVE.

What are we actually trying to do? We're trying to run inference on a 4GB RAM phone with a 2GB model. The bottleneck is memory bandwidth and latency. We have:
- CPU (good at sequential, has cache)
- GPU (good at parallel, non-coherent path to DRAM)
- DRAM (the actual data)
- UFS swap (slow but big)

What if the GPU isn't for prefetch OR compute? What if it's for TRAFFIC SHAPING?

Holy shit. What if the GPU's job is to CREATE ARTIFICIAL MEMORY PRESSURE that KEEPS DRAM ROWS OPEN?

Like... the GPU doesn't prefetch specific data. It just reads PATTERNS that keep the memory controller in a state that benefits the CPU.

That's insane. But is it?

---

What else did we learn that we didn't fully process?

1. Combined CPU+GPU bandwidth was 18 GB/s, EXCEEDING what either could do alone
2. Row buffer locality is 2.66x
3. Uncached access has 1-2% variance vs 5-30% for cached
4. The memory controller (EMI) does its own optimization

Wait. The EMI. What does it actually do? It's an ARBITER. It decides who gets memory when. What if we can INFLUENCE its arbitration?

MediaTek has MMQOS - Memory Quality of Service. We saw `mtk_mmqos_scen` - different scenarios for camera, video, etc. What if there's an INFERENCE scenario we can set?

Or what if we CREATE one by behaving like a camera?

---

Another wild thought: The GPU is non-coherent. What if we use it as a DMA ENGINE?

Not for compute. Not for prefetch. For MOVEMENT.

Like: CPU wants data from location A. Instead of CPU reading A, GPU reads A into a shared buffer, CPU reads shared buffer. Why would this help?

Because the GPU path might have DIFFERENT LATENCY CHARACTERISTICS. Different queue depths. Different burst patterns.

This is probably stupid. But maybe not?

---

What about the LITTLE cores?

We've been focusing on big cores (A78) for inference. But we have 6 LITTLE cores (A55) doing nothing. What if they become the "fabric"?

LITTLE cores read ahead, big cores compute. LITTLE cores are in the same coherency domain but... wait, that's what we tested and it failed.

Unless... what if LITTLE cores use uncached access? They're not coherent with themselves for this purpose. They read uncached, big cores read uncached, no coherency needed.

But then we're back to "just use uncached."

---

The EMI has 2 channels (2x16 bit = 32 bit bus). What if there's channel affinity? What if we can steer certain accesses to certain channels?

Like... weights on channel 0, KV cache on channel 1. Parallel paths. No contention.

Is that even controllable? Probably not from userspace. But maybe from kernel? The `emichn` driver...

---

What about the DVFSRC? It controls DRAM frequency. We set it to 4266 MHz. But it's DYNAMIC. It wants to scale based on demand.

What if we FOOL it into thinking there's always high demand? Force it to stay at 4266 MHz by generating artificial traffic?

That's basically what we're doing with the boot script. But what if we do it SMARTER? Generate traffic patterns that make the DVFSRC happy while also warming useful memory regions?

---

UNCACHED MEMORY IS FASTER FOR RANDOM ACCESS.

But why? Let's really think about this.

Cached access: Check L1 → miss → check L2 → miss → check L3/system cache → miss → DRAM
Uncached access: DRAM

So uncached is faster because it SKIPS THE CACHE CHECKS? That seems backwards. Cache checks should be fast (ns).

Unless... the cache check isn't just a check. It's a COHERENCY PROTOCOL. It has to snoop. It has to potentially invalidate. It has to manage state machines.

For LARGE random access, every access is a miss anyway. So you're paying the coherency overhead for NO BENEFIT.

This is why uncached wins: it's not faster at DRAM access. It's faster by AVOIDING USELESS CACHE PROTOCOL.

So the question becomes: what's the overhead of the cache protocol vs the benefit of caching?

For repeated access: cache wins (obviously)
For sequential access: HW prefetch wins (it fills the cache ahead of you)
For large random access: uncached wins (no point paying protocol overhead for guaranteed misses)

The CROSSOVER POINT was ~16MB. That's roughly the size of L2 cache (probably L2 is 512KB-2MB, L3 might be shared ~4MB). Above that, cache is useless for random access.

---

NEW IDEA: What if we create a TIERED memory system?

Tier 1: L1/L2 cache - Activations (small, repeated)
Tier 2: Uncached DRAM - Weights and KV cache (large, streaming/random)
Tier 3: UFS swap - Overflow KV cache (huge context)

The GPU's job: MANAGE TIER 2→3 TRANSITIONS.

When KV cache grows beyond DRAM, GPU handles the eviction/loading to swap. It runs ASYNC. It uses its non-coherent path to move data without disturbing CPU cache.

This isn't a "memory fabric." It's a "MEMORY MANAGER AS COPROCESSOR."

---

What about DMA engines? The SoC probably has dedicated DMA controllers. Can we use those instead of GPU?

Hmm, DMA is usually for device↔memory, not memory↔memory. But some SoCs have memory-to-memory DMA...

Actually, GPU IS a DMA engine. With compute. We're just using the DMA part.

---

Let me think about the ORIGINAL question differently.

"Eliminate cache coherency overhead"

Coherency overhead happens when multiple agents access shared data. The overhead is:
1. Snoop requests (broadcast to all caches)
2. State transitions (MESI/MOESI)
3. Data transfers (cache-to-cache)

We can eliminate this by:
A) Not sharing data (separate regions) - boring
B) Using uncached access (no cache states to manage) - what we found
C) Making all agents use the SAME cache (impossible on this arch)
D) Somehow SYNCHRONIZING access so coherency is cheap

D is interesting. What if we time GPU and CPU access to NEVER overlap on the same cache line? Then coherency protocol still runs but there's never any actual state to manage.

That's... basically a software implementation of cache partitioning. We manually ensure no conflicts.

Actually, isn't that just option A with extra steps?

---

WILD THOUGHT: What if we use the GPU's TEXTURE CACHE?

GPUs have specialized caches for texture sampling. They're optimized for spatial locality in 2D. Model weights can be thought of as 2D matrices...

If we could lay out weights as "textures" and read them through the texture path, we might get:
- GPU-side caching (separate from CPU)
- Optimized for 2D access patterns
- Non-coherent (no CPU interaction)

Then CPU reads the OUTPUT of GPU texture operations. Not the raw weights.

This is getting into actual GPU compute territory. But with a twist: we're using GPU cache, not GPU ALUs.

The PowerVR BXM-8-256 has texture units. Does it have a texture cache we can exploit?

---

Back to basics. What do we KNOW for sure?

1. DRAM bandwidth is ~11-18 GB/s depending on access pattern
2. Uncached has lower latency for random access >16MB
3. Uncached has dramatically lower variance
4. Cache coherency adds ~50+ ns for cross-core transfers
5. GPU has non-coherent path to DRAM
6. EMI does arbitration, DVFSRC controls frequency
7. MediaTek has ION/dma_heap for uncached allocation

What do we NOT know?

1. Can we control EMI arbitration priorities?
2. Can we use GPU texture cache for weight access?
3. What's the actual benefit in tok/s for uncached KV cache?
4. Are there other non-coherent agents we can leverage?
5. What's the power impact of all this?

---

THE LAUNDRY METHOD SAYS: Check the delta. The boundaries.

What's at the boundary of cached/uncached? 
- 16MB working set (crossover point)
- Sequential vs random access pattern
- Repeated vs streaming access

What's at the boundary of coherent/non-coherent?
- GPU memory space
- ION/dma_heap regions
- Device memory (MMIO)

What's at the boundary of CPU/GPU/memory controller?
- The SMI interconnect
- The EMI arbiter
- The DVFSRC frequency controller

THE INTERESTING STUFF IS AT THESE BOUNDARIES.

---

What if the REAL fabric is the SMI (Smart Multimedia Interface)?

SMI connects everything: display, camera, video codec, ISP... and GPU and CPU.

Each subsystem has a LARB (Local Arbiter). There are 13 LARBs on this SoC. They all connect to the SMI common, which connects to EMI.

What if we can INFLUENCE the SMI scheduling? Make it prioritize inference traffic?

The MMQOS (Memory Quality of Service) might let us do this. There was `mtk_mmqos_scen` we couldn't read...

---

GETTING FUNKY:

What if we pretend to be a VIDEO DECODER?

Video decoders get high priority on mobile SoCs. They need consistent bandwidth for smooth playback. The system is tuned to prioritize them.

What if our inference engine presents itself as a video decoder to the SMI? Gets allocated LARB slots? Gets QoS priority?

This is probably impossible without kernel modifications. But the CONCEPT is: use the system's existing optimization for a different workload.

---

What if the solution isn't about MEMORY at all?

What if it's about HIDING memory latency through COMPUTATION?

Classic computer architecture: overlap compute and memory access. While the CPU waits for one memory load, it computes on previously loaded data.

In inference, this means: while loading layer N weights, compute layer N-1. Pipeline the model.

We sort of do this already (transformer layers are naturally sequential). But are we doing it OPTIMALLY?

What's the ratio of compute time to memory time for one layer? If compute >> memory, we're compute bound (good, memory is hidden). If memory >> compute, we're memory bound (bad, need prefetch/optimization).

On this phone, with Q4_0 quantization... probably memory bound. Hence all this effort.

---

RECURSIVE IDEA:

What if we run the LMM on THIS VERY DOCUMENT?

Phase 1 (RAW) is what I just wrote.
Phase 2 (NODES) would extract the key insights.
Phase 3 (REFLECT) would find the patterns.
Phase 4 (SYNTH) would give us the clean cut.

Let's do it. This RAW is done. Time for NODES.

---

## Questions Arising

1. Can we control EMI/SMI/MMQOS from userspace?
2. Is GPU texture cache exploitable for inference?
3. What IS the compute/memory ratio for our model?
4. Can we pretend to be a video decoder?
5. What's at the BOUNDARY of all these domains?
6. Is there a "fabric" or are we just doing memory tiering?

## First Instincts

- Uncached memory is the obvious win; implement it first
- But there's something more here about the INTERCONNECT
- The SMI/EMI/MMQOS system is underexplored
- GPU as "traffic shaper" is weird but might work
- The real question is: what workload profile triggers the best memory controller behavior?
