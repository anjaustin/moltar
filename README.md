# Moltar

Embedded AI inference on a $99 phone. Custom llama.cpp kernels, three-layer memory (LCVDB working memory + ColBERT knowledge retrieval + LFM2 generation), and persistent sessions, targeting the Motorola Moto G Power 5G (2023).

## Results

**LLM inference** — Liquid AI LFM2 models on MediaTek Dimensity 930 (2x Cortex-A78 @ 2.2 GHz):

| Model | Quant | Size | tok/s (tg32) |
|-------|-------|------|-------------|
| LFM2-350M | Q4_0 | 190 MiB | **77.1** |
| LFM2-350M | Q8_0 | 359 MiB | **40.9** |
| LFM2-700M | Q4_0 | 423 MiB | **33.0** |
| LFM2-1.2B | Q4_0 | 661 MiB | **21.3** |

**Semantic working memory** — HNSW search with NEON int8 dot products, L-cache-resident:

| N nodes | Search latency | QPS | Recall@5 | Working set |
|---------|---------------|-----|----------|-------------|
| 256 | 23.4 us | 43K | **100%** | 32 KB (L1D) |
| 512 | 27.2 us | 37K | **100%** | 64 KB (L1D/L2) |
| 1024 | 27.1 us | 37K | 93.2% | 128 KB (L2) |

**On-device RAG** — ColBERT retrieval + LFM2 generation via `moltar-agent`:

| Step | Time |
|------|------|
| ColBERT embedding (cold, model load) | ~0.9 s |
| Variance-weighted MaxSim search (10 chunks) | ~16 ms |
| LFM2-1.2B reload (mmap fault-in w/ prefetch) | ~4-5 s |
| Prompt processing + generation | ~5 s |
| **Total (RAG-enabled query)** | **~10-14 s** |

**Three-layer memory architecture**:

```
+----------------------------------------------------------+
|  LCVDB (Working Memory)     23 us    N=64-512   L1D      |
|  Conversation turns, entities, agent state                |
+----------------------------------------------------------+
|  ColBERT (Knowledge Memory) ~1 s     N=10-1000  DRAM     |
|  Document retrieval, persistent knowledge base            |
+----------------------------------------------------------+
|  LFM2-1.2B (Generation)    21 tok/s  661 MiB    DRAM     |
|  Prompt with merged working + knowledge context           |
+----------------------------------------------------------+
```

## Hardware

**Device**: Motorola Moto G Power 5G 2023 (codename `devonn`, SKU `XT2311-4`)

- **SoC**: MediaTek MT6855V / Dimensity 930
- **CPU**: 2x Cortex-A78 @ 2.2 GHz + 6x Cortex-A55
- **ISA**: ARMv8.2-a with DOTPROD, FP16, NEON (no I8MM, no SVE)
- **RAM**: 6 GB LPDDR4X @ 4266 MHz — 15.5 GB/s sustained (with Android stopped)
- **Cache**: L1D 64 KB/core, L2 256 KB/A78, L3 ~1 MB shared
- **GPU**: PowerVR BXM-8-256 — OpenCL 3.0, 128-wide SIMD

Device is rooted via Magisk. Android framework is stopped during inference to reclaim ~4 GB/s of DRAM bandwidth.

## Repository Structure

```
moltar/
├── research/
│   ├── llama.cpp/              # Custom llama.cpp fork with MT6855V kernels
│   │   └── ggml/src/ggml-cpu/
│   │       ├── repack.h        # Row-scaled Q4_0/Q8_0 block structs
│   │       ├── repack.cpp      # Repack functions, tensor traits, forward_mul_mat
│   │       └── arch/arm/
│   │           └── repack.cpp  # NEON DOTPROD GEMV/GEMM kernels
│   ├── memory/                 # Memory integration layer + agent
│   │   ├── moltar-agent.c      # Three-layer memory agent (main program)
│   │   ├── project.c/h        # Rademacher random projection (2048D -> 48D)
│   │   ├── test_project.c     # Projection + LCVDB integration tests
│   │   └── Makefile
│   ├── colbert/                # ColBERT RAG pipeline
│   │   ├── colbert.h/c        # 128D int8 token embeddings, MaxSim scoring
│   │   ├── maxsim_neon.S      # NEON SDOT assembly for MaxSim
│   │   ├── moltar_rag.c       # Variance-weighted MaxSim search binary
│   │   ├── moltar_rag.sh      # RAG orchestrator (ingest, query, demo)
│   │   ├── test_colbert.c     # Correctness tests + benchmark
│   │   └── knowledge/         # Sample knowledge base
│   └── lcvdb/                  # L-Cache VDB (semantic working memory)
│       ├── lcvdb.h             # Split storage structs + API
│       ├── distance.S          # NEON int8 dot products (overflow-safe)
│       ├── init_ref.c          # C reference init
│       ├── build_ref.c         # C reference HNSW insert with diversity heuristic
│       ├── search_ref.c        # C reference beam search (uses NEON dot products)
│       ├── test_lcvdb.c        # Correctness tests
│       ├── test_recall.c       # Recall benchmark
│       ├── test_bench.c        # Performance benchmark
│       └── Makefile
├── PRD.md                      # Product requirements + next steps
├── PERFORMANCE.md              # Detailed benchmark data
├── ARCHITECTURE.md             # System design
└── CHANGELOG.md                # What changed and when
```

## Quick Start

### Conversational Agent (Full System)

```bash
# Build the agent (NDK, links against libllama.so)
cd research/memory
make agent

# Push to device
adb push moltar-agent /data/local/tmp/

# Setup perf mode
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor'"
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor'"

# Run with working memory + session persistence
adb shell "su -c 'export LD_LIBRARY_PATH=/data/local/tmp && \
  taskset c0 /data/local/tmp/moltar-agent /data/local/tmp/LFM2-1.2B-Q4_0.gguf \
  -t 2 -c 2048 --session /data/local/tmp/session.bin'"

# Run with working memory + ColBERT RAG + session persistence
adb shell "su -c 'export LD_LIBRARY_PATH=/data/local/tmp && \
  taskset c0 /data/local/tmp/moltar-agent /data/local/tmp/LFM2-1.2B-Q4_0.gguf \
  -t 2 -c 2048 --rag --session /data/local/tmp/session.bin'"
```

### LLM Inference Only

```bash
cd research/llama.cpp
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r27c/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33
cmake --build build --target llama-cli -j$(nproc)

adb push build/bin/llama-cli /data/local/tmp/
adb shell "su -c 'stop'"
adb shell "su -c 'taskset c0 /data/local/tmp/llama-cli \
  -m /data/local/tmp/LFM2-350M-Q4_0-pure.gguf -t 2 -p \"Hello\"'"
```

### Vector Database

```bash
cd research/lcvdb
make all
adb push test_lcvdb test_recall test_bench /data/local/tmp/
adb shell "su -c 'taskset c0 /data/local/tmp/test_lcvdb'"
```

## Key Discoveries

1. **Android framework is the bandwidth bottleneck** — SurfaceFlinger, SystemUI etc. consume ~4 GB/s of DRAM bandwidth. Running `su -c 'stop'` eliminates "thermal throttling" and stabilizes inference at full speed. A Magisk boot script automates this.

2. **Row-scaled quantization** — Custom Q4_0/Q8_0 block formats aligned to 64-byte cache lines, with pure-integer SDOT accumulation and power-of-2 shift activation quantization. Eliminates all float ops from the GEMV inner loop.

3. **Split storage for VDB** — Separating topology (64 bytes/node, M=16) from vectors (64 bytes/node) keeps the graph structure in L1D cache during traversal. At N=256, total working set is 32 KB.

4. **Variance-weighted mean-centered MaxSim** — Standard ColBERT MaxSim fails on small corpora because common tokens dominate. Mean-centering per query token and weighting by cross-chunk score variance acts as learned IDF, eliminating common-token noise.

5. **Three-layer memory with session persistence** — LCVDB working memory (23 us), ColBERT knowledge retrieval (~1 s), and LFM2 generation (21 tok/s) compose into a single agent that maintains conversational state across restarts via binary session files.

6. **Activation quant cache bug** — A cache keyed on buffer pointers produced stale data due to allocator address reuse, causing gibberish output. Removed entirely; benchmarks corrected by 2-4%.

## Documentation

| Document | What it covers |
|----------|---------------|
| [PRD.md](PRD.md) | Current state, next steps, acceptance criteria |
| [PERFORMANCE.md](PERFORMANCE.md) | All benchmark data with methodology |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design: kernels, VDB, RAG, memory layers |
| [CHANGELOG.md](CHANGELOG.md) | Commit-level history of what changed |
| [HARDWARE_COMPATIBILITY.md](HARDWARE_COMPATIBILITY.md) | MT6855V specs, cache hierarchy, ISA details |
