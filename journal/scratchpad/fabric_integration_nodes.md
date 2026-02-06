# NODES: Fabric Integration Key Insights

## Node 1: The Size Threshold
Uncached memory only benefits large allocations (>16 MB). The 3 MiB KV cache in LFM2 is below this threshold. Hardware prefetch on cached memory handles small sequential/semi-random access efficiently.

## Node 2: Implementation vs Hypothesis Gap
The fabric backend is implemented correctly - allocates uncached memory, integrates with ggml buffer system. But testing with wrong model size. Implementation is Phase 1; the hypothesis needs Phase 2 (larger context) to validate.

## Node 3: Build System Sensitivity
ARM SIMD flags (dotprod+fp16) account for 47% performance difference (19 vs 28 tok/s). Memory optimizations are secondary to instruction-level optimization. Build configuration matters more than expected.

## Node 4: Offload Flag Mismatch
`offload_kqv=true` even on CPU-only devices. Design assumption that offload=false means CPU was wrong. Had to check actual buffer type instead. Lesson: verify assumptions against runtime behavior.

## Node 5: Missing Prefetch Integration
The LITTLE core prefetch mechanism exists in libfabric but isn't integrated with llama.cpp attention. Current integration is allocation-only, not access-pattern-aware. The 1.04x speedup from probe_c requires overlapped prefetch during compute.

## Node 6: Variance vs Throughput Tradeoff
Uncached: 5% slower throughput, but CV 0.8-4.7% vs 10-21%. For latency-sensitive applications, predictability might be worth the speed cost. Different optimization goals lead to different choices.

## Node 7: Context Length Scaling
At default benchmark context (256 cells), KV cache is 3 MiB. At 4K context, it would be 48 MiB. At 128K context (LFM2 max), 1.5 GB. The benefit zone is somewhere in the middle - large enough to exceed cache, small enough to fit in RAM.

## Node 8: Hybrid Architecture Sparsity
LFM2 has only 6 attention layers out of 16 (rest are shortconv). KV cache is sparse. Traditional transformer would have denser KV access patterns where uncached might help more.

## Node 9: Memory Bandwidth Reality
At 28 tok/s, we're memory bound not compute bound. Memory optimizations SHOULD matter. But the access pattern might be too sequential for uncached to help - hardware prefetch is excellent at sequential.

## Node 10: Infrastructure Completeness
The dma_heap integration works cleanly. Opens device, allocates, maps. Fallback to malloc on failure. Buffer type system integrates well. The infrastructure is ready; the workload needs to match.

## Node 11: KleidiAI Dominance
KleidiAI GEMM optimization gave huge gains. Matrix multiply speed might be more important than memory access pattern. If compute is fast enough, memory latency gets hidden.

## Node 12: Per-Token Variance Unmeasured
llama-bench reports average tok/s, not per-token variance. To truly validate variance reduction, need custom instrumentation measuring each token's generation time.

## Node 13: Phase 2 Requirements
To complete the fabric story:
1. Integrate LITTLE core prefetch with attention loop
2. Test with larger context (4K+ tokens)
3. Test with pure transformer model (not hybrid)
4. Measure per-token latency variance

## Node 14: Defensive Design
The code handles edge cases - NULL buffer type falls back, failed allocation falls back, buffer type comparison by name works. Production-ready error handling.

## Node 15: The Real Question
Is the memory fabric concept valid for mobile LLM inference? Evidence says: yes for large random access, no for small/sequential. Need to find the crossover point for this specific hardware.
