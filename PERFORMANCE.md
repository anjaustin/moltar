# Performance

All numbers measured on Motorola Moto G Power 5G 2023 (MT6855V / Dimensity 930), Cortex-A78 big cores @ 2.2 GHz, Android framework stopped (`su -c 'stop'`), `taskset c0` (cores 6-7).

## LLM Inference

Custom llama.cpp fork with row-scaled quantization and NEON DOTPROD GEMV/GEMM kernels.

### Token Generation (tg32, 2 threads)

| Model | Quant | Weight size | tok/s | GB/s DRAM | Bottleneck |
|-------|-------|-------------|-------|-----------|------------|
| LFM2-350M | Q4_0 | 190 MiB | **77.1** | 14.4 | DRAM BW |
| LFM2-350M | Q8_0 | 359 MiB | **40.9** | 14.4 | DRAM BW |
| LFM2-700M | Q4_0 | 423 MiB | **33.0** | 13.7 | DRAM BW |
| LFM2-1.2B | Q4_0 | 661 MiB | **21.3** | 13.8 | DRAM BW |

**Note**: Numbers corrected after removing a buggy activation quantization cache that was inflating throughput by 2-4% while producing gibberish output (see Changelog).

All models are DRAM-bandwidth-bound during token generation. The sustained DRAM bandwidth of the MT6855V with Android stopped is **15.5 GB/s** (2 threads, LPDDR4X @ 4266 MHz).

### Prompt Processing (pp32, 2 threads)

GEMM kernels process 4 activation rows simultaneously with weight reuse.

| Model | Quant | tok/s (pp32) |
|-------|-------|-------------|
| LFM2-350M | Q4_0 | ~304 |

### Optimization Timeline

| Optimization | LFM2-350M Q4_0 tok/s | Improvement |
|-------------|----------------------|-------------|
| Baseline (upstream llama.cpp + DOTPROD) | ~58 | — |
| Row-Scaled Q4_0 format | ~56 | format change, slight regression |
| Pure-integer SDOT GEMV | ~56 | correctness, no speed change |
| GEMM dispatch (pp) | ~56 tg / 304 pp | 3x prompt speedup |
| Barrier-skip | 56.8 | +1.4% |
| Graph dispatch + thread fast-forward | 58.3 | +2.6% |
| **Android framework stopped** | **77.1** | **+32%** |

The single biggest gain was discovering that Android framework processes consume ~4 GB/s of DRAM bandwidth. Stopping them (`su -c 'stop'`) immediately raised throughput from 58 to 77 tok/s.

**Note**: An earlier "activation quant caching" optimization was removed — it caused stale quantized activations due to buffer address reuse by the allocator, producing gibberish output. The 2-4% "speedup" came from skipping re-quantization with stale data.

### Kernel Breakdown (simpleperf)

| Function | % cycles | Notes |
|----------|----------|-------|
| `forward_mul_mat` (GEMV) | ~35% | Row-scaled Q4_0, SDOT inner loop |
| Token embedding (Q6_K) | ~18% | Single tensor, not optimized |
| GEMM (prompt) | ~11% | 4-row batch, weight reuse |
| Thread sync (barriers) | ~7% | Reduced by barrier-skip |
| Tensor repack | ~3% | One-time cost |

## L-Cache Vector Database

HNSW graph search with NEON int8 dot products. Split storage layout: topology (64 bytes/node, M=16) separate from vectors (64 bytes/node). Total: 128 bytes/node.

### Search Latency

| N | k=1 (ns) | k=5 (ns) | k=10 (ns) | QPS | Cycles |
|---|----------|----------|-----------|-----|--------|
| 32 | 2,363 | 2,362 | 2,367 | 423K | 5,198 |
| 64 | 6,686 | 6,682 | 6,683 | 150K | 14,709 |
| 128 | 13,711 | 13,697 | 13,720 | 73K | 30,164 |
| 256 | 23,460 | 23,437 | 23,412 | 43K | 51,612 |
| 512 | 27,185 | 27,177 | 27,193 | 37K | 59,807 |
| 1024 | 27,115 | 27,125 | 27,050 | 37K | 59,653 |

Note: k value barely affects latency — beam search dominates, not output copy. Search is ~10-20% slower than the old brute-force-insert graph because the better-connected graph causes beam search to explore more nodes. The tradeoff is dramatically better recall.

### Recall

| N | recall@1 | recall@5 | recall@10 |
|---|----------|----------|-----------|
| 32 | 100% (50/50) | 100% (250/250) | 100% (500/500) |
| 64 | 100% (50/50) | 100% (250/250) | 100% (500/500) |
| 128 | 100% (50/50) | 100% (250/250) | 100% (500/500) |
| 256 | 100% (50/50) | 100% (250/250) | 100% (500/500) |
| 512 | 100% (50/50) | 100% (250/250) | 99.0% (495/500) |
| 1024 | 96% (48/50) | 93.2% (233/250) | 94.6% (473/500) |

100% graph connectivity at all N. Beam search during insert (build_ref.c) with NEON dot products produces much better graphs than the old brute-force candidate collection. Previous numbers (84.8% recall@5 at N=1024) were from the pre-beam-search-fix era.

Remaining recall gap at N=1024 likely caused by sparse upper layers (P(layer>=1) = 1/8 vs standard HNSW 1/3) and aggressive diversity pruning (>= threshold). Not blocking for the working memory use case (N=64-512, where recall is 100%).

### Build Time

| N | Total build | Per insert |
|---|-------------|------------|
| 32 | 30 us | 953 ns |
| 64 | 215 us | 3,371 ns |
| 128 | 1,057 us | 8,260 ns |
| 256 | 4.4 ms | 16,999 ns |
| 512 | 13.8 ms | 27,046 ns |
| 1024 | 34.5 ms | 33,650 ns |

Build uses HNSW beam search during insert + NEON dot products (via `lcvdb_dot_i8` from distance.S). Per-insert cost is higher than old brute-force at small N because beam search does more graph traversal, but graph quality is dramatically better.

### Memory Budget (Split Storage)

| N | Topology (64B/node) | Vectors (64B/node) | Total | Fits in |
|---|----------|---------|-------|---------|
| 256 | 16 KB | 16 KB | 32 KB | L1D (64 KB) |
| 512 | 32 KB | 32 KB | 64 KB | L1D/L2 |
| 1024 | 64 KB | 64 KB | 128 KB | L2 (256 KB) |
| 4096 | 256 KB | 256 KB | 512 KB | L2/L3 |
| 65534 | 4 MB | 4 MB | 8 MB | DRAM |

During search, only topology is traversed continuously. Vectors are loaded on-demand for distance computation. Effective hot working set = topology array only.

## ColBERT RAG Pipeline

End-to-end on-device retrieval-augmented generation using LFM2-ColBERT-350M (Q4_0, 209 MB) for embedding and LFM2-1.2B (Q4_0, 661 MB) for generation.

### Latency Budget

| Step | Time | Notes |
|------|------|-------|
| Embed query (8-15 tokens) | ~66 ms | LFM2-ColBERT-350M Q4_0 |
| MaxSim search (10 chunks) | ~15 ms | Brute-force, NEON SDOT |
| LLM generate (short answer) | ~1-2 s | LFM2-1.2B Q4_0, ~21 tok/s |
| **Total (short answer)** | **~2 s** | |

### MaxSim Scoring

ColBERT uses late-interaction scoring: each query token's 128D embedding is dotted against all document token embeddings, taking the max per document token, then summing across query tokens.

- **NEON kernel**: `colbert_maxsim_i8` in `maxsim_neon.S` — uses SDOT for 128D int8 dot products
- **Quantization**: float32 embeddings quantized to int8 per-vector (max-abs scaling)
- **Scaling**: O(Q * D * T) where Q=query tokens, D=documents, T=avg tokens/doc. Currently brute-force over all documents.

### ColBERT Correctness Tests (on device)

All 4 tests pass: dot product, MaxSim scoring, quantization, search ranking.

## DRAM Bandwidth

Measured with custom membw_test using NEON `ld1` streaming loads on big cores:

| Threads | Android running | Android stopped |
|---------|----------------|-----------------|
| 1 | ~8 GB/s | ~10 GB/s |
| 2 | ~11 GB/s | **15.5 GB/s** |

Theoretical max for LPDDR4X @ 4266 MHz, dual-channel: ~17 GB/s. We achieve 91% of theoretical with Android stopped.

## Methodology

- All measurements use `CLOCK_MONOTONIC_RAW` on the device
- CPU governors set to `performance` (2.2 GHz locked)
- `taskset c0` pins to big cores (6-7)
- Android framework stopped (`su -c 'stop'`)
- Warmup iterations run before timing (2000 for VDB, varies for LLM)
- VDB bench uses 100K iterations for N<=256, 50K for N=512, 10K for N=1024
- LLM numbers from llama-bench with `-r 5` (5 repetitions)
