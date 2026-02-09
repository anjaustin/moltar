# Moltar — Product Requirements Document

## What Moltar Is

Moltar is an embedded AI platform for running LLM inference on a **Motorola Moto G Power 5G (2023)** — a $99 phone with a MediaTek Dimensity 930 SoC. The goal is maximum tokens-per-second for Liquid AI's LFM2 models, plus on-device retrieval-augmented generation (RAG) using ColBERT late-interaction embeddings.

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

### L-Cache VDB (`research/lcvdb/`)

A vector database engine targeting CPU L-cache residency. HNSW graph search with NEON assembly distance functions. Split storage design (separate topology + vector arrays). **Not used by the ColBERT RAG pipeline** — ColBERT uses per-token 128D embeddings with MaxSim scoring, which is architecturally different from single-vector HNSW search.

**Architecture** (split storage, current):
- **Topology array** (`lcvdb_topo_t`, 32 bytes/node): graph edges, neighbor IDs (uint16 x 8), flags, payload_id
- **Vector array** (`lcvdb_vec_t`, 64 bytes/node): 48D int8 embeddings
- **uint16 IDs**: max 65534 nodes (up from 256 with old uint8 layout)
- **Tombstone delete**: flag in topology, skipped during search

**Performance** (A78 @ 2.2 GHz, split storage):

| N | Search (ns) | QPS | Recall@5 | Topo | Vec | Total |
|---|------------|-----|----------|------|-----|-------|
| 32 | 2,291 | 436K | 100% | 1 KB | 2 KB | 3 KB |
| 64 | 6,060 | 165K | 100% | 2 KB | 4 KB | 6 KB |
| 128 | 12,505 | 80K | 100% | 4 KB | 8 KB | 12 KB |
| 256 | 19,415 | 51K | 99.6% | 8 KB | 16 KB | 24 KB |
| 512 | 23,488 | 42K | 94.4% | 16 KB | 32 KB | 48 KB |
| 1024 | 24,272 | 41K | 84.8% | 32 KB | 64 KB | 96 KB |

**Files**:
- `lcvdb.h` — split storage structs and API
- `distance.S` — NEON int8 dot products (overflow-safe, per-chunk widening)
- `init_ref.c` — C reference initialization
- `build_ref.c` — C reference HNSW insert with diversity heuristic
- `search_ref.c` — C reference beam search (NEON dot products via distance.S)
- `init.S`, `build.S`, `search.S` — old assembly (pre-split-storage, not in use)

---

## Next Steps

### P0 — RAG: Knowledge Base Expansion + Quality

**Status**: ColBERT RAG pipeline works end-to-end. Current knowledge base is a single file (`knowledge/moltar.txt`, 10 chunks). Next steps:

**Work**:
- Expand knowledge base with more documents/domains
- Evaluate retrieval quality with diverse queries
- Tune chunk size and overlap for better context
- Benchmark MaxSim scaling at 100+ chunks (currently brute-force, O(N))
- Consider GPU-accelerated MaxSim for large indices

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

### P1 — LLM: Further Bandwidth Optimization

**Current bottleneck**: DRAM bandwidth. At 15.5 GB/s sustained, a 661 MiB model reads at ~21 tok/s. The only way to go faster is to reduce bytes read per token.

**Options**:
- **Q3_0 or Q2_0 quantization**: Lower bits per weight, but quality drops. Need to measure LFM2 quality at Q3.
- **Speculative decoding**: Use LFM2-350M as draft model, LFM2-1.2B as verifier. If acceptance rate is high, effective throughput could increase 2-3x.
- **KV cache quantization**: Reduce KV cache memory to keep more in cache.
- **Sliding window attention**: LFM2 uses SSM (state-space), not attention — already has O(1) state. This may already be optimal.

### P2 — VDB: Fix Recall at Large N

**Problem**: LCVDB recall drops from 99.6% at N=256 to 84.8% at N=512. Not currently blocking (ColBERT RAG uses brute-force MaxSim, not HNSW), but needed if LCVDB is used for other retrieval tasks.

**Fix**: Replace brute-force candidate collection with HNSW beam search during insert.

**Acceptance criteria**:
- recall@5 >= 95% at N=1024
- 100% graph reachability at all N

### P2 — VDB: GPU Async Search

**Problem**: When the CPU is running LLM inference (which saturates both A78 cores), VDB searches block. The PowerVR GPU sits idle during token generation.

**Opportunity**: Dispatch VDB search or MaxSim scoring to GPU via OpenCL while CPU does LLM inference. GPU dispatch overhead is ~40 us (too slow for standalone use), but if overlapped with a 12-50 ms token generation step, the latency is free.

**Design**:
- OpenCL kernel for int8 dot product already written and verified (`gpu_dot.cl`)
- GPU probe confirms: 128-wide SIMD, 28 KB local mem, OpenCL 3.0
- Implement: enqueue search before LLM token gen, read results after
- CPU remains primary path for standalone queries

### P3 — Multi-Device Support

Moltar currently targets exactly one phone. Future devices to consider:
- Other MediaTek Dimensity phones (similar ISA, different cache sizes)
- Snapdragon devices (Hexagon DSP available, different NEON extensions)
- Raspberry Pi 5 (Cortex-A76, similar ISA, useful for development)

### Completed

- **P2 — Integration: LLM + RAG Pipeline** — DONE. Implemented using ColBERT late-interaction retrieval (not LCVDB HNSW). See ColBERT RAG section above.

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
