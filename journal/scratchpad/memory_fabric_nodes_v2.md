# NODES v2: Memory Fabric - The Grain

## Node 1: The MMQOS System is a Black Box

We found `mtk_mmqos_scen` but couldn't read it. MediaTek's Memory QoS system controls how bandwidth is allocated between subsystems (camera, display, video, etc.).

**The delta:** Inference isn't a recognized "scenario." We're getting default/generic memory treatment.

**Wild question:** Can we BECOME a recognized scenario? Can we register as a high-priority memory consumer?

---

## Node 2: Uncached Wins Because Cache Protocol is Expensive

The reason uncached beats cached for large random access isn't that DRAM is faster - it's that we AVOID the cache coherency protocol overhead.

For random access on a 64MB region, EVERY access is a cache miss. So we pay:
- Cache lookup (~2-5ns) 
- L2 lookup (~10-20ns)
- Coherency snoop (~50ns+)
- DRAM access (~50-100ns)

Vs uncached:
- DRAM access (~50-100ns)

The first path is ~2x longer for NO BENEFIT when working set >> cache size.

**The delta:** The crossover point is ~16MB. Below: cache wins. Above: uncached wins.

---

## Node 3: Combined CPU+GPU Exceeds Individual Bandwidth

We measured 18 GB/s combined when CPU and GPU read concurrently. This EXCEEDS what either could do alone (~10-11 GB/s each).

**This is weird.** It suggests the memory controller has separate paths/queues for CPU vs GPU. They're not competing - they're parallel.

**Wild question:** Can we use this to get MORE than 18 GB/s with better orchestration? What's the theoretical max?

---

## Node 4: GPU as Traffic Shaper

Forget prefetch. Forget compute. What if GPU's job is to CREATE A MEMORY ACCESS PATTERN that optimizes the memory controller's behavior?

Memory controllers do their own optimization:
- Row buffer management
- Bank interleaving  
- Request reordering
- Burst coalescing

What if GPU traffic can INFLUENCE these optimizations in a way that benefits CPU?

**Concrete example:** GPU reads addresses A, A+64, A+128... (sequential). This opens DRAM row A. CPU reads A+32 (same row, different offset). CPU benefits from row being open.

This is what we tried before and it failed due to cache coherency. BUT WHAT IF BOTH USE UNCACHED?

---

## Node 5: Uncached + Uncached = No Coherency

We tested:
- Cached CPU + Cached GPU = coherency overhead (0.75x)
- Uncached CPU = faster (1.3x for random)

We DIDN'T test:
- Uncached CPU + Uncached GPU together

If both are uncached, there's NO cache state to manage. No coherency protocol. Just two agents hitting DRAM through the same controller.

**This might work.** GPU and CPU both use uncached access to the same region. GPU reads to warm rows. CPU reads to use data. No coherency overhead because there's no cache.

---

## Node 6: The SMI Has 13 LARBs

The Smart Multimedia Interface connects 13 Local Arbiters (LARBs) to the EMI. Each LARB serves different subsystems:
- LARB0/1: Display
- LARB2: MDP (MediaTek Display Processor)
- LARB4: VDEC (video decoder)
- LARB7/8: VENC (video encoder)
- etc.

**Wild question:** Which LARB does the GPU use? Which does the CPU use? Can we influence LARB allocation?

If CPU and GPU are on different LARBs, they have separate arbitration. This could explain the 18 GB/s combined bandwidth.

---

## Node 7: Video Decoder Gets Priority

Mobile SoCs are optimized for video playback. VDEC (video decoder) gets high QoS priority because dropped frames are visible to users.

**Wild thought:** What if inference could "impersonate" video decoding? Access memory in patterns that look like video decode? Get treated with similar priority?

This is probably impossible without kernel mods. But the PRINCIPLE is: the system already has optimized paths for certain workloads. Can we exploit them?

---

## Node 8: The Variance Gap

Cached: CV 5-65%
Uncached: CV 1-7%

The variance difference is MORE significant than the average latency difference. Uncached is PREDICTABLE.

**For real-time applications, predictability > raw speed.**

If you're streaming tokens to a user, consistent 100ms is better than variable 50-200ms. The perceived quality is higher.

---

## Node 9: Tiered Memory is the Obvious Answer

The boring-but-correct answer:
- Tier 1: Cached (activations, small repeated data)
- Tier 2: Uncached DRAM (weights, KV cache)  
- Tier 3: UFS swap (overflow)

GPU manages Tier 2↔3 transitions asynchronously.

This is the "memory manager as coprocessor" model. GPU doesn't help with compute or prefetch - it manages memory lifecycle.

---

## Node 10: What if We Test GPU Texture Cache?

GPUs have specialized caches for texture sampling. PowerVR has texture units. What if we:
1. Lay out weights as "textures" (2D arrays)
2. Access them through texture sampling operations
3. Benefit from GPU-side caching without CPU coherency

This would require:
- Vulkan/OpenCL implementation
- Reformatting model weights as textures
- Measuring if texture cache hit rate matters

**This is actual GPU compute, not fabric.** But with a twist: we're using GPU's specialized memory path, not its ALUs.

---

## Node 11: The 18 GB/s Mystery

At 4266 MT/s with a 32-bit bus: theoretical max is 17.1 GB/s.

We measured 18 GB/s combined. This EXCEEDS theoretical single-channel max.

**Possible explanations:**
1. Measurement error
2. Burst mode exceeding sustained rate
3. Two 16-bit channels acting as parallel paths
4. Memory controller has more capability than spec

If (3), then we might be able to push further with better access patterns.

---

## Node 12: The DVFSRC Wants to Scale Down

The DRAM frequency controller (DVFSRC) wants to save power by reducing frequency when load is low. We force it to 4266 MHz.

But forcing might be FIGHTING the system. What if we WORK WITH the system?

**Wild idea:** Instead of forcing 4266 MHz, generate access patterns that make DVFSRC WANT to stay at 4266 MHz. The system optimizes itself.

This is "cooperative" vs "coercive" optimization.

---

## Node 13: The Real Question is Latency Hiding

Classic architecture: overlap compute and memory access. While waiting for memory, compute on previous data.

**Question:** What's the compute/memory ratio for our model?

If compute >> memory: we're compute bound, memory optimization has limited benefit.
If memory >> compute: we're memory bound, all this effort matters.

On a phone with Q4_0 quantization, probably memory bound. But we should MEASURE.

---

## Node 14: The LITTLE Cores Are Idle

We have 6 A55 cores doing nothing during inference (we use 2 A78 big cores).

What if LITTLE cores become the "fabric"? They pre-read data (uncached) that big cores will need. Different cores, no coherency issue with uncached.

**Concrete architecture:**
- Thread on LITTLE core reads next layer's weights (uncached)
- Thread on BIG core computes current layer
- Overlap: LITTLE is always one layer ahead

This is software prefetch with dedicated cores.

---

## Node 15: The Delta is the Boundary

The Laundry Method says: check the boundaries.

Boundaries in our problem:
- 16MB: cached/uncached crossover
- CPU/GPU: coherent/non-coherent boundary
- LARB: arbitration boundaries
- DRAM row: locality boundary (~8KB)
- Cache line: 64 bytes

**The interesting solutions live at these boundaries.**

---

## Tensions

**Tension A:** Uncached is faster (Node 2) vs HW prefetch is faster for sequential (Probe B v8)
→ Resolution: Different patterns need different strategies

**Tension B:** GPU prefetch failed (original tests) vs maybe uncached GPU+CPU works (Node 5)
→ Resolution: Must test uncached-uncached scenario

**Tension C:** Simple tiered memory (Node 9) vs complex fabric coordination (Node 4)
→ Resolution: Try simple first, escalate if needed

**Tension D:** Force DRAM frequency (current) vs cooperative optimization (Node 12)
→ Resolution: Forcing works, but cooperative might be better

---

## Connections

- Node 3 (18 GB/s) + Node 6 (13 LARBs) → Maybe separate LARBs explain parallel bandwidth
- Node 5 (uncached both) + Node 14 (LITTLE cores) → LITTLE cores as uncached prefetchers
- Node 7 (VDEC priority) + Node 1 (MMQOS) → Can we get VDEC-like priority?
- Node 8 (variance) + Node 9 (tiers) → Uncached tier gives predictable latency
