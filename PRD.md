# Moltar — Product Requirements Document

## What Moltar Is

Moltar is an embedded AI platform for running LLM inference on a **Motorola Moto G Power 5G (2023)** — a $99 phone with a MediaTek Dimensity 930 SoC. Three memory layers on one device: **LCVDB** for microsecond working memory (conversation recall, entity tracking), **ColBERT** for millisecond knowledge retrieval (document search), and **LFM2** for generation at 21-77 tok/s. Everything runs on 2x Cortex-A78 cores.

**Target device**: Motorola Moto G Power 5G 2023 (codename `devonn`, SKU `XT2311-4`)
- **SoC**: MediaTek MT6855V / Dimensity 930
- **CPU**: 2x Cortex-A78 @ 2.2 GHz (big) + 6x Cortex-A55 (little)
- **ISA**: ARMv8.2-a with DOTPROD, FP16, NEON. No I8MM, no SVE.
- **Memory**: 6 GB LPDDR4X @ 4266 MHz, 15.5 GB/s sustained bandwidth (with Android stopped)
- **Cache**: L1D 64 KB/core, L2 256 KB/A78, L3 ~1 MB shared, 64-byte cache lines
- **GPU**: PowerVR BXM-8-256 — OpenCL 3.0, 128-wide SIMD, 28 KB local mem, 950 MHz

---

## Current State

### LLM Inference (`research/llama.cpp/`)

Custom llama.cpp fork with hand-tuned kernels for the MT6855V. All optimizations target the DRAM-bandwidth bottleneck of token generation.

| Model | Quant | Size | tok/s (tg32) |
|-------|-------|------|-------------|
| LFM2-350M | Q4_0 | 190 MiB | **77.1** |
| LFM2-350M | Q8_0 | 359 MiB | **40.9** |
| LFM2-700M | Q4_0 | 423 MiB | **33.0** |
| LFM2-1.2B | Q4_0 | 661 MiB | **21.3** |

**Key discovery**: Android framework (SurfaceFlinger, SystemUI) consumes ~4 GB/s of DRAM bandwidth. Running `su -c 'stop'` frees this bandwidth and eliminates the "thermal throttling" we originally diagnosed. A Magisk boot script at `/data/adb/service.d/moltar_perf.sh` auto-stops Android on boot.

**Optimizations shipped** (in `ggml/src/ggml-cpu/`):
1. Row-Scaled Q4_0 format (`block_q4_0x4_rs`) — 64-byte cache-line-aligned blocks
2. Row-Scaled Q8_0 format (`block_q8_0x4_rs`) — 128-byte blocks
3. Pure-integer SDOT GEMV kernels — `vdotq_laneq_s32` inner loop
4. Power-of-2 shift activation quantization — IEEE754 bit manipulation
5. GEMM dispatch for prompt processing — 4 activation rows with weight reuse
6. Barrier-skip, graph dispatch fast path, thread 1 fast-forward
7. Inlined GEMV in TG fast path
8. NEON vectorization of binary-ops, RMS_NORM, SSM_CONV, CONCAT

**Bug fix**: An activation quantization cache keyed on `src1->data` pointer was producing stale results due to buffer address reuse by the allocator. This caused gibberish output despite correct speed measurements. Fixed by removing the cache (2-4% throughput cost, previous numbers were inflated).

**Modified files**:
- `ggml/src/ggml-cpu/repack.h` — block structs + function declarations
- `ggml/src/ggml-cpu/repack.cpp` — repack functions, tensor_traits, forward_mul_mat
- `ggml/src/ggml-cpu/arch/arm/repack.cpp` — NEON DOTPROD GEMV/GEMM kernels

### ColBERT RAG Pipeline (`research/colbert/`)

On-device retrieval-augmented generation using **LFM2-ColBERT-350M** for late-interaction embeddings and **LFM2-1.2B** for generation. Tested end-to-end on device.

**Architecture**: ColBERT produces per-token 128D float embeddings. Documents are chunked, embedded, quantized to int8, and indexed. At query time, MaxSim scoring (max over document tokens of dot product with each query token, summed across query tokens) ranks chunks. Top-k chunks are injected into an LLM prompt for grounded generation.

**Latency budget** (measured on device):

| Step | Time |
|------|------|
| Embed query (8-15 tokens) | ~66 ms |
| MaxSim search (10 chunks) | ~15 ms |
| LLM generate (LFM2-1.2B Q4_0) | ~seconds |
| **Total** | **~2 s for short answer** |

**Files**:
- `colbert.h` — 128D int8 token embeddings, document index, MaxSim API
- `colbert.c` — C implementation: init, quantize f32->i8, add_doc, search with top-k heap
- `maxsim_neon.S` — NEON assembly: `colbert_dot_i8` (128D SDOT), `colbert_maxsim_i8` (full MaxSim)
- `test_colbert.c` — 4 correctness tests + benchmark (all pass on device)
- `moltar_rag.c` — Search tool: parses raw embedding files, runs MaxSim, outputs ranked results
- `moltar_rag.sh` — Shell orchestrator: `ingest`, `query`, `demo` commands
- `knowledge/moltar.txt` — Sample knowledge base about the Moltar project
- `Makefile` — Builds with `aarch64-linux-gnu-gcc -static`

**Models**:
- `LFM2-ColBERT-350M-Q4_0.gguf` (209 MB) — embedding model
- `LFM2-1.2B-Q4_0.gguf` (661 MB) — generation model

**Note**: The two models run sequentially (not enough RAM for both + KV cache simultaneously).

### L-Cache VDB (`research/lcvdb/`) — Semantic Memory Layer

A cache-resident HNSW vector database designed for sub-25 us associative lookup. Split storage (separate topology + vector arrays), NEON assembly distance functions, 48D int8 embeddings.

**Role**: LCVDB is not a document retrieval engine — ColBERT handles that. LCVDB is a **semantic register file**: a fast, small, dynamic index for working memory. Natural use cases:
- **Conversation memory**: embed each turn, recall related prior context during generation
- **Entity tracking**: "have we discussed this entity before?" at 436K QPS
- **Agent working memory**: index intermediate reasoning steps for self-referential retrieval
- **Semantic deduplication**: detect near-duplicate inputs in real-time

These workloads require small N (64-512), continuous inserts, instant lookups, and coarse semantic matching — exactly where LCVDB operates.

**Architecture** (split storage, M=16):
- **Topology array** (`lcvdb_topo_t`, 64 bytes/node = 1 cache line): 16 neighbor IDs (uint16), count, layer, flags, payload_id
- **Vector array** (`lcvdb_vec_t`, 64 bytes/node = 1 cache line): 48D int8 embeddings + 16 bytes padding
- **uint16 IDs**: max 65534 nodes (0xFFFF = invalid sentinel)
- **Tombstone delete**: flag in topology, skipped during search

**Performance** (A78 @ 2.2 GHz, split storage):

| N | Search (ns) | QPS | Recall@5 | Topo | Vec | Total | Fits in |
|---|------------|-----|----------|------|-----|-------|---------|
| 32 | 2,291 | 436K | 100% | 2 KB | 2 KB | 4 KB | L1D |
| 64 | 6,060 | 165K | 100% | 4 KB | 4 KB | 8 KB | L1D |
| 128 | 12,505 | 80K | 100% | 8 KB | 8 KB | 16 KB | L1D |
| 256 | 19,415 | 51K | 99.6% | 16 KB | 16 KB | 32 KB | L1D (64 KB) |
| 512 | 23,488 | 42K | 94.4% | 32 KB | 32 KB | 64 KB | L1D/L2 |
| 1024 | 24,272 | 41K | 84.8% | 64 KB | 64 KB | 128 KB | L2 (256 KB) |

**Note**: Recall numbers may be stale (pre-beam-search-fix). Re-benchmark needed. build_ref.c now uses HNSW beam search during insert (O(log N)) instead of brute-force O(N) scan.

**Files**:
- `lcvdb.h` — split storage structs and API
- `distance.S` — NEON int8 dot products (overflow-safe, per-chunk widening)
- `init_ref.c` — C reference initialization
- `build_ref.c` — C reference HNSW insert with beam search + diversity heuristic
- `search_ref.c` — C reference beam search (NEON dot products via distance.S)
- `init.S`, `build.S`, `search.S` — old assembly (pre-split-storage, not in use)

---

## Next Steps

### P0 — Memory Architecture: Three-Layer System

**Vision**: Three timescales of memory on a $99 phone.

```
┌──────────────────────────────────────────────────────────┐
│  LCVDB (Working Memory)     19 us    N=64-512   L1D     │
│  Conversation turns, entities, agent state               │
├──────────────────────────────────────────────────────────┤
│  ColBERT (Knowledge Memory) 80 ms    N=10-1000  DRAM    │
│  Document retrieval, persistent knowledge base           │
├──────────────────────────────────────────────────────────┤
│  LFM2-1.2B (Generation)    21 tok/s  661 MiB    DRAM    │
│  Prompt with merged working + knowledge context          │
└──────────────────────────────────────────────────────────┘
```

**Work**:
1. **Embedding pipeline for LCVDB**: Produce 48D int8 vectors from LFM2 hidden states during generation via random projection. The 1536D hidden states already exist; a fixed 1536x48 int8 projection matrix (73 KB) reduces them to 48D. Zero additional model loading.
2. **Conversation memory integration**: Insert each conversation turn into LCVDB after generation. Before generating, search LCVDB for top-3 related prior turns and inject into prompt alongside ColBERT knowledge context.
3. **Context merging**: Design prompt template that combines LCVDB working memory (recent related turns) with ColBERT knowledge retrieval (relevant documents).

**Acceptance criteria**:
- 48D int8 embeddings produced from LFM2 hidden states during generation
- Conversation turns inserted into LCVDB in <50 us
- Top-3 related prior turns retrieved in <25 us
- Total memory overhead: <64 KB for 256 turns (fits L1D)
- Improved coherence on multi-turn conversations vs no working memory

### P0 — LCVDB: Re-Benchmark + Quick Fixes

**Problem**: Documentation numbers may be stale (pre-beam-search-fix). Topo memory sizes were wrong in all docs (listed as 32B/node, actual is 64B/node with M=16).

**Work**:
1. Build and push `test_recall` to device, compare with documented numbers
2. Switch build_ref.c from scalar `dot_i8()` to `lcvdb_dot_i8` (NEON) — one-line change, ~3x build speedup
3. Document corrections already applied (this session)

**Acceptance criteria**:
- Fresh recall numbers at N=32..1024 with current build_ref.c (beam search)
- Build uses NEON dot product
- All doc memory numbers reflect 64B/node topo (corrected)

### P1 — RAG: Knowledge Base Expansion + Quality

**Status**: ColBERT RAG pipeline works end-to-end. Current knowledge base is a single file (`knowledge/moltar.txt`, 10 chunks).

**Work**:
- Expand knowledge base with more documents/domains
- Evaluate retrieval quality with diverse queries
- Tune chunk size and overlap for better context
- Benchmark MaxSim scaling at 100+ chunks

**Acceptance criteria**:
- Coherent, grounded answers across multiple knowledge domains
- Retrieval latency < 100 ms at 100 chunks

### P1 — LLM: LFM2.5-1.2B-Thinking Support

**Status**: Model file already on device (`LFM2.5-1.2B-Thinking-Q4_0.gguf`, 696 MiB). Not yet benchmarked or validated.

**Work**:
- Benchmark tg32 performance (expect ~21 tok/s, similar to LFM2-1.2B)
- Validate output quality
- Test thinking/reasoning traces
- Integrate with RAG pipeline as generation model

### P2 — LLM: Further Bandwidth Optimization

**Current bottleneck**: DRAM bandwidth. At 15.5 GB/s sustained, a 661 MiB model reads at ~21 tok/s. The only way to go faster is to reduce bytes read per token.

**Options**:
- **Speculative decoding**: Use LFM2-350M as draft model, LFM2-1.2B as verifier. If acceptance rate is high, effective throughput could increase 2-3x.
- **Q3_0 or Q2_0 quantization**: Lower bits per weight, but quality drops. Need to measure LFM2 quality at Q3.
- **KV cache quantization**: Reduce KV cache memory to keep more in cache.

### P2 — LCVDB: Recall Fix at Large N (if needed)

**Problem**: If re-benchmark confirms recall <95% at N=512+, two likely causes:
1. **Sparse upper layers**: P(layer>=1) = 1/8 vs standard HNSW's ~1/3. Fix: change PRNG layer assignment.
2. **Aggressive diversity**: `>=` threshold over-prunes edges. Fix: relax to `>`.

These only matter if LCVDB takes on workloads at N>512. For working memory (N=64-512), current recall is 94-100%.

### P3 — GPU Async Search/Scoring

Dispatch LCVDB search or ColBERT MaxSim to GPU via OpenCL while CPU does LLM inference. GPU dispatch overhead is ~40 us (free when overlapped with 12-50 ms token generation).

### P3 — Multi-Device Support

Moltar currently targets exactly one phone. Future devices:
- Other MediaTek Dimensity phones (similar ISA, different cache sizes)
- Snapdragon devices (Hexagon DSP, different NEON extensions)
- Raspberry Pi 5 (Cortex-A76, useful for development)

### Completed

- **P2 — Integration: LLM + RAG Pipeline** — DONE. Implemented using ColBERT late-interaction retrieval. See ColBERT RAG section above.

### Lincoln Manifold Analysis

Full exploration of LCVDB's role and architecture in `journal/scratchpad/lcvdb_{raw,nodes,reflect,synth}.md`. Key insight: LCVDB was miscast as a document retrieval engine. Its architecture (sub-25 us, L1D-resident, 48D coarse embeddings) maps to working memory, not knowledge retrieval. ColBERT and LCVDB serve different memory timescales, not competing use cases.

---

## Dead Ends (Don't Retry)

These were investigated and conclusively ruled out:

- **L3 prefetch from LITTLE cores** — no measurable benefit
- **GPU memory fabric / uncached DMA heap** — dispatch overhead too high
- **Multi-column GEMV** — register pressure kills throughput
- **Software prefetching in GEMV** — no benefit, hardware prefetch sufficient
- **2x inner loop unrolling** — no benefit, pipeline already saturated
- **Shift-add replacing SDOT** — slower than SDOT
- **SwiGLU vrecpeq optimization** — negligible impact
- **Hand-written NEON ASM for GEMV** — compiler intrinsics were better
- **DVFSRC bandwidth requests** — no effect on MT6855V
- **CM Manager disable** — no effect
- **Thermal emulation / DRAM thermal throttling hypothesis** — disproven (it was Android framework bandwidth)
- **GPU for VDB latency at small N** — 40+ us dispatch overhead exceeds CPU search time
- **Older NEON search.S** — has beam management bug, not worth fixing (rewrite from scratch)
- **NEON assembly port of LCVDB search** — `-ffixed-v0/v1/v2` constraint costs more than preloaded query saves
- **Inline asm dot product in search_neon.c** — 4% slower than C ref
- **Activation quant caching** — cache keyed on `src1->data` pointer; allocator reuses buffer addresses for different tensors, causing stale quantized activations and gibberish output. Removed entirely (2-4% throughput cost; old numbers were inflated by the bug)

---

## Build & Deploy

```bash
# Cross-compile ColBERT tools (from x86_64 host)
cd research/colbert
make clean all   # builds test_colbert, moltar_rag

# Cross-compile VDB (from x86_64 host)
cd research/lcvdb
make all   # builds test_lcvdb, test_recall, test_bench

# Cross-compile LLM (llama.cpp for Android)
cd research/llama.cpp
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r27c/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33
cmake --build build-android --target llama-cli llama-bench llama-embedding -j$(nproc)

# Deploy everything to device
adb push research/llama.cpp/build-android/bin/{llama-cli,llama-bench,llama-embedding} /data/local/tmp/
adb push research/llama.cpp/build-android/bin/lib*.so /data/local/tmp/
adb push research/colbert/{moltar_rag,test_colbert,moltar_rag.sh} /data/local/tmp/
adb push research/lcvdb/{test_lcvdb,test_recall,test_bench} /data/local/tmp/

# Setup perf mode on device
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor'"
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor'"
adb shell "su -c 'stop'"  # kill Android framework, free ~4 GB/s DRAM BW

# Run RAG demo
adb shell "su -c 'sh /data/local/tmp/moltar_rag.sh demo'"

# Run VDB tests on big cores
adb shell "su -c 'taskset c0 /data/local/tmp/test_lcvdb'"
adb shell "su -c 'taskset c0 /data/local/tmp/test_recall'"
adb shell "su -c 'taskset c0 /data/local/tmp/test_bench'"
```

## Toolchain

- **VDB cross-compiler**: `aarch64-linux-gnu-gcc` (static linking)
- **LLM / GPU cross-compiler**: NDK r27c at `/opt/android-ndk-r27c`
- **Device**: rooted via Magisk, serial `ZY22HWSKXX`, connected via ADB
- **No emulation**: all testing on physical device, no qemu
