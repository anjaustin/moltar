# Moltar — Product Requirements Document

## What Moltar Is

Moltar is an embedded AI platform for running LLM inference on a **Motorola Moto G Power 5G (2023)** — a $200 phone with a MediaTek Dimensity 930 SoC. The goal is maximum tokens-per-second for Liquid AI's LFM2 models, plus a vector database that fits entirely in CPU cache.

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
| LFM2-350M | Q4_0 | 190 MiB | **80.06** |
| LFM2-350M | Q8_0 | 359 MiB | **41.69** |
| LFM2-700M | Q4_0 | 423 MiB | **32.96** |
| LFM2-1.2B | Q4_0 | 661 MiB | **21.66** |

**Key discovery**: Android framework (SurfaceFlinger, SystemUI) consumes ~4 GB/s of DRAM bandwidth. Running `su -c 'stop'` frees this bandwidth and eliminates the "thermal throttling" we originally diagnosed. A Magisk boot script at `/data/adb/service.d/moltar_perf.sh` auto-stops Android on boot.

**Optimizations shipped** (in `ggml/src/ggml-cpu/`):
1. Row-Scaled Q4_0 format (`block_q4_0x4_rs`) — 64-byte cache-line-aligned blocks
2. Row-Scaled Q8_0 format (`block_q8_0x4_rs`) — 128-byte blocks
3. Pure-integer SDOT GEMV kernels — `vdotq_laneq_s32` inner loop
4. Power-of-2 shift activation quantization — IEEE754 bit manipulation
5. GEMM dispatch for prompt processing — 4 activation rows with weight reuse
6. Barrier-skip, graph dispatch fast path, thread 1 fast-forward
7. Inlined GEMV in TG fast path, activation quant caching
8. NEON vectorization of binary-ops, RMS_NORM, SSM_CONV, CONCAT

**Modified files**:
- `ggml/src/ggml-cpu/repack.h` — block structs + function declarations
- `ggml/src/ggml-cpu/repack.cpp` — repack functions, tensor_traits, forward_mul_mat
- `ggml/src/ggml-cpu/arch/arm/repack.cpp` — NEON DOTPROD GEMV/GEMM kernels

### L-Cache VDB (`research/lcvdb/`)

A vector database engine targeting CPU L-cache residency. HNSW graph search with NEON assembly distance functions. Just completed a redesign from monolithic 64-byte nodes to **split storage** (separate topology + vector arrays).

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

### P0 — VDB: Fix Recall at Large N

**Problem**: Recall drops from 99.6% at N=256 to 84.8% at N=512 and 81.2% at N=1024. Graph connectivity also degrades (1018/1024 reachable at N=1024).

**Root cause**: `build_ref.c` uses brute-force scan of ALL existing nodes to find 16 candidates (CAND_MAX=16). This is both slow (O(N) per insert, O(N^2) total build) and insufficient — at large N, 16 candidates from a global scan doesn't provide enough local diversity for the HNSW heuristic.

**Fix**: Replace brute-force candidate collection with **HNSW-style beam search during insert** (standard HNSW Algorithm 2). Use the existing graph to find neighbors, not a full scan. This:
- Reduces insert from O(N) to O(log N) per node
- Improves recall because candidates are locally relevant, not globally top-16
- Fixes connectivity because the search naturally explores connected regions
- Enables scaling to N=4096+ without quadratic build time

**Acceptance criteria**:
- recall@5 >= 95% at N=1024
- recall@5 >= 90% at N=4096
- 100% graph reachability at all N
- Build time for N=1024 under 10 ms

### P0 — VDB: Port to NEON Assembly

**Problem**: `build_ref.c` uses scalar C dot products during insert. `search_ref.c` already uses NEON (via `lcvdb_dot_i8` from `distance.S`), but the build path doesn't.

**Fix**: After the recall fix is validated in C, port `init_ref.c`, `build_ref.c`, and `search_ref.c` to AArch64 assembly using split storage offsets:

```
DB struct offsets:      Topo node offsets:     Vec slot offsets:
  +0: node_count (u32)   +0: neighbors (u16x8)   +0: vector (i8x48)
  +4: entry_point (u16)  +16: nbr_count (u8)
  +6: max_level (u8)     +17: max_layer (u8)
  +7: M (u8)             +18: flags (u16)
  +8: topo_array (ptr)   +20: payload_id (u32)
  +16: vec_array (ptr)
  +24: max_nodes (u32)   Address: topo_array + id << 5
  +28: prng_state (u32)  Address: vec_array + id << 6
```

Key changes from old assembly: `ldrh`/`strh` for uint16 neighbor IDs (was `ldrb`/`strb`), separate topo/vec array base pointers, wider ID fields in candidate pools.

**Acceptance criteria**:
- Identical recall to C reference at all N
- Search latency at N=256 <= 19 us (matching or beating old assembly)
- Insert uses NEON dot products (measured speedup vs C scalar)

### P1 — VDB: GPU Async Search

**Problem**: When the CPU is running LLM inference (which saturates both A78 cores), VDB searches block. The PowerVR GPU sits idle during token generation.

**Opportunity**: Dispatch VDB search to GPU via OpenCL while CPU does LLM inference. GPU dispatch overhead is ~40 us (too slow for standalone use), but if overlapped with a 12-50 ms token generation step, the latency is free.

**Design**:
- OpenCL kernel for int8 dot product already written and verified (`gpu_dot.cl`)
- GPU probe confirms: 128-wide SIMD, 28 KB local mem, OpenCL 3.0
- Implement: enqueue search before LLM token gen, read results after
- CPU remains primary path for standalone VDB queries

**Acceptance criteria**:
- GPU search returns correct results (matches CPU brute-force)
- GPU search completes within one LLM token generation step
- No interference with LLM inference throughput

### P1 — LLM: LFM2.5-1.2B-Thinking Support

**Status**: Model file already on device (`LFM2.5-1.2B-Thinking-Q4_0.gguf`, 696 MiB). Not yet benchmarked or validated.

**Work**:
- Benchmark tg32 performance (expect ~21 tok/s, similar to LFM2-1.2B)
- Validate output quality
- Test thinking/reasoning traces

### P2 — Integration: LLM + VDB Pipeline

**Goal**: RAG (Retrieval-Augmented Generation) pipeline running entirely on-device.

**Design**:
1. User query → quantize to int8 embedding (using LFM2-350M hidden states or a small encoder)
2. VDB search → retrieve top-k relevant chunks (payload_id maps to text)
3. Construct prompt with retrieved context
4. LFM2-1.2B generates response

**Open questions**:
- How to produce int8 48D embeddings from LFM2 hidden states (768D float → 48D int8)
- Whether to use a separate small encoder or reuse LFM2-350M
- Context window management for retrieved chunks
- Memory budget: VDB (96 KB for 1024 chunks) + LFM2-1.2B (661 MiB) + prompt = well within 6 GB

### P2 — LLM: Further Bandwidth Optimization

**Current bottleneck**: DRAM bandwidth. At 15.5 GB/s sustained, a 661 MiB model reads at ~21 tok/s. The only way to go faster is to reduce bytes read per token.

**Options**:
- **Q3_0 or Q2_0 quantization**: Lower bits per weight, but quality drops. Need to measure LFM2 quality at Q3.
- **Speculative decoding**: Use LFM2-350M as draft model, LFM2-1.2B as verifier. If acceptance rate is high, effective throughput could increase 2-3x.
- **KV cache quantization**: Reduce KV cache memory to keep more in cache.
- **Sliding window attention**: LFM2 uses SSM (state-space), not attention — already has O(1) state. This may already be optimal.

### P3 — Multi-Device Support

Moltar currently targets exactly one phone. Future devices to consider:
- Other MediaTek Dimensity phones (similar ISA, different cache sizes)
- Snapdragon devices (Hexagon DSP available, different NEON extensions)
- Raspberry Pi 5 (Cortex-A76, similar ISA, useful for development)

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

---

## Build & Deploy

```bash
# Cross-compile VDB (from x86_64 host)
cd research/lcvdb
make all

# Deploy to device
adb push test_lcvdb test_recall test_bench /data/local/tmp/

# Setup perf mode on device
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor'"
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor'"
adb shell "su -c 'stop'"  # kill Android framework, free ~4 GB/s DRAM BW

# Run on big cores
adb shell "su -c 'taskset c0 /data/local/tmp/test_lcvdb'"
adb shell "su -c 'taskset c0 /data/local/tmp/test_recall'"
adb shell "su -c 'taskset c0 /data/local/tmp/test_bench'"

# Cross-compile LLM (llama.cpp)
cd research/llama.cpp
cmake -B build -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r27c/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33
cmake --build build --target llama-cli -j$(nproc)
```

## Toolchain

- **VDB cross-compiler**: `aarch64-linux-gnu-gcc` (static linking)
- **LLM / GPU cross-compiler**: NDK r27c at `/opt/android-ndk-r27c`
- **Device**: rooted via Magisk, serial `ZY22HWSKXX`, connected via ADB
- **No emulation**: all testing on physical device, no qemu
