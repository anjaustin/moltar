# Raw Thoughts: GPU as Memory Fabric Layer

## Stream of Consciousness

The idea is to use the GPU not for compute but as a memory system assistant. The GPU shares the same DRAM as the CPU. When GPU reads memory, it opens DRAM rows. When CPU follows, those rows might still be open = row buffer hit instead of miss.

But wait — I'm making assumptions. Do I actually know:
- That DRAM row buffers are shared between GPU and CPU access paths?
- That the memory controller doesn't have separate queues that invalidate each other's row opens?
- That the GPU read latency is low enough to actually get ahead of the CPU?
- That the prefetch doesn't HURT by consuming memory bandwidth the CPU needs?

The dispatch overhead is 270us. A matvec is 758us. So GPU has ~500us to prefetch before CPU catches up. At 25.6 GB/s (3200 MHz DRAM), that's ~12.8 MB of prefetch possible. A layer's weights are ~2.65 MB. So we could prefetch 4-5 layers ahead. That seems like overkill.

But here's the problem: GPU and CPU share the bus. If GPU is reading weights at 12 GB/s, that's bandwidth the CPU can't use. We might slow down the current matvec to speed up the next one. Net effect could be zero or negative.

What am I actually trying to solve? The CPU matvec is memory-bound — we measured 51% of theoretical DRAM bandwidth. The other 49% is... what? Cache misses? Bank conflicts? Row buffer misses?

If row buffer misses are significant, then GPU prefetch could help. If it's cache misses or bank conflicts, GPU prefetch does nothing useful.

I don't actually know the breakdown of the 49% loss.

The PowerVR probe showed the GPU can read HOST_COHERENT memory with zero-copy. The GGUF can be imported as a Vulkan buffer via external memory. This part is real and proven.

But there's a deeper question: is the memory controller smart enough to keep rows open across requestors? Or does a GPU access close the row and a CPU access has to reopen it?

On high-end systems (Intel, AMD), there's usually a row buffer policy — open page vs closed page. Open page keeps rows open for subsequent accesses. Closed page closes after each access. Mobile SoCs typically use... I don't know. MediaTek's memory controller is opaque.

What if the GPU access actually INVALIDATES the CPU's prefetch? ARM CPUs have hardware prefetchers. What if GPU traffic confuses them?

The worst case: GPU consumes bandwidth, pollutes whatever shared state exists, and the CPU gets slower. We've traded compute capability we weren't using for bandwidth we were.

The best case: GPU opens rows ahead of CPU, CPU hits warm rows, matvec gets faster by 10-20%.

How would I even measure this? The matvec variance might change — fewer row misses = more consistent timing. But I'd need many samples and careful statistical analysis.

What scares me:
- The memory controller is a black box
- We could make things worse
- The improvement might be within measurement noise
- This is a lot of complexity for uncertain gain

What excites me:
- It's a novel use of the GPU that sidesteps its compute weaknesses
- If it works, it's "free" performance — GPU was idle anyway
- It would validate the unified memory architecture as actually useful
- This is the kind of thing nobody has tried on mobile

## Questions Arising

1. What is the DRAM row buffer policy on MT6855?
2. Does GPU access warm or invalidate CPU row buffer state?
3. What percentage of matvec time is row buffer misses vs other stalls?
4. Does the ARM A78 have hardware prefetchers? How do they behave?
5. Can we measure row buffer hit rate somehow?
6. What happens to CPU bandwidth when GPU is actively reading?
7. Is there a memory controller performance monitor we can access?
8. Does the PowerVR driver have any prefetch or bandwidth hints?

## First Instincts

- This is probably a dead end, but it's an interesting dead end
- The right first step is to measure memory controller behavior, not build a prefetch shader
- We need a probe that measures bandwidth interference between GPU and CPU
- If interference is high, this idea is DOA
- If interference is low, we need a second probe to measure row buffer warming
- The actual prefetch shader is the last thing to build, not the first
- We're potentially fighting the hardware prefetcher, not helping it
