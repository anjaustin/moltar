# RAW: Fabric Integration Findings

## Stream of Consciousness

The fabric integration worked. Code compiles, runs, allocates uncached memory from dma_heap. But the benchmark shows 5% SLOWER with uncached KV cache. 26.96 vs 28.45 tok/s.

Why? The LFM2 model has only 3 MiB KV cache. That's tiny. Fits in L2/L3 easily. Our earlier probing showed uncached only wins for LARGE random access (>16MB). For small stuff, hardware prefetch on cached memory is excellent.

So we built the right thing but tested with the wrong model. The implementation is correct - the hypothesis needs a bigger test.

What did we learn about the codebase? The offload parameter is always true even on CPU-only devices. That was surprising. Had to check if buft == CPU instead of checking offload flag. Design assumption didn't match reality.

The build system complexity - had to match dotprod+fp16 flags to get same performance. Without those flags, 19 tok/s. With them, 28 tok/s. That's a 47% difference just from ARM SIMD instructions. Way bigger than any memory optimization.

Cross-core prefetch - our earlier probes showed 1.04x speedup with LITTLE core prefetch on uncached. But that was with overlapped prefetch. The libfabric implementation doesn't do overlapped prefetch yet - it's just uncached allocation. The prefetch thread pool exists but isn't integrated with llama.cpp attention loop.

The variance story - uncached had CV 0.8-4.7% vs 10-21% for cached. That's still true. If you need predictable latency (real-time systems), uncached might be worth the speed hit.

What's the actual bottleneck? At 28 tok/s on a 700M model, we're doing ~20B operations per token (rough estimate). At 2.2 GHz with 2 big cores, that's maybe 4.4 GFLOPS effective. The theoretical peak for Cortex-A78 is much higher. So we're memory bound, not compute bound. Which means memory optimizations SHOULD matter.

But the KV cache access pattern for this model might be too sequential. LFM2 is a hybrid architecture with shortconv and attention. The attention layers are sparse (only 6 out of 16). Maybe the access pattern isn't random enough to benefit from uncached.

The dma_heap allocation itself works perfectly. Opens /dev/dma_heap/mtk_mm-uncached, ioctls for allocation, mmaps the buffer. Clean integration with ggml backend system. The infrastructure is solid.

One thing I didn't test - latency variance during inference. llama-bench reports average tok/s but not variance per token. Would need custom instrumentation to measure that.

The context size matters. We tested with n=32 tokens. With longer sequences, the KV cache grows and access becomes more random. At 128K context (LFM2's max), the KV cache would be much larger.

Actually wait - let me recalculate. 6 attention layers, 512-dim KV (padded), 256 cells (small context for bench), f16... that's 6 * 2 * 512 * 256 * 2 bytes = 3 MiB. If we used full 128K context, it would be 6 * 2 * 512 * 128K * 2 = 1.5 GB. Way too big for 4GB RAM.

More realistic - 4K context would give 48 MiB KV cache. That's where fabric might help. The benchmark uses tiny context by default.

The LITTLE core prefetch isn't being used at all in this integration. We just changed the memory allocation, not the access pattern. The real win from fabric requires coordinating LITTLE cores to prefetch during compute. That's Phase 2 - haven't done it yet.

What about the tensor data layout? ggml uses row-major tensors. KV cache access during attention is... need to trace through the code. If it's sequential enough, cached prefetch handles it.

The fact that KleidiAI gave such huge speedup suggests the bottleneck is in matrix ops, not memory. KleidiAI optimizes GEMM/GEMV. If those are fast enough, memory access might not be the limiting factor.

Thermal considerations - ran many benchmarks, phone might be throttling. Though temps showed 30.8C which is fine.

One odd finding - the new llama.cpp code was 33% slower (19 vs 28 tok/s) until we added dotprod+fp16 flags. That's because cmake detection failed for those features. The original build had them in CMAKE_C_FLAGS directly. Build system fragility.

The buffer type comparator uses strcmp on names. That's fine but means buffer types must have unique names. Our "CPU_Fabric" works.

The ctx_map groups layers by buffer type. With fabric enabled, all CPU layers use CPU_Fabric, so they share one allocation. That's efficient.

Memory alignment - fabric uses 4096 byte alignment (page size). ggml's default is 32 bytes. Shouldn't matter for correctness but might affect performance.

The fallback path works - if dma_heap fails, falls back to ggml_aligned_malloc. Good defensive coding.

What would make fabric actually faster?
1. Larger model with bigger KV cache
2. Longer context length
3. Integrated LITTLE core prefetch during attention
4. Access patterns that defeat hardware prefetch

Or maybe the hypothesis is just wrong for this hardware. MediaTek's memory controller might be good enough that uncached doesn't help.

The phone has 4GB RAM. After model loading (~500MB), OS, etc., maybe 2GB free. KV cache for long context would compete with model for memory. Might get swap pressure.

We set up zram earlier - 2GB swap. That would interact badly with uncached memory since zram compresses. Though KV cache values probably don't compress well.

The variance reduction is real though. For production systems where p99 latency matters, fabric might be worth it even at lower throughput.
