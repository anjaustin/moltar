# Moltar

Embedded AI inference on a $99 phone. Custom llama.cpp kernels, ColBERT RAG pipeline, and an L-cache-resident vector database, targeting the Motorola Moto G Power 5G (2023).

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
| 256 | 19.4 us | 51K | 99.6% | 32 KB (L1D) |
| 512 | 23.5 us | 42K | 94.4% | 64 KB (L1D/L2) |
| 1024 | 24.3 us | 41K | 84.8% | 128 KB (L2) |

**On-device RAG** — ColBERT late-interaction retrieval + LLM generation:

| Step | Time |
|------|------|
| Embed query (8-15 tokens) | ~66 ms |
| MaxSim search (10 chunks) | ~15 ms |
| LLM generate (LFM2-1.2B Q4_0) | ~1-2 s |
| **Total (short answer)** | **~2 s** |

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
│   ├── colbert/                # ColBERT RAG pipeline
│   │   ├── colbert.h/c        # 128D int8 token embeddings, MaxSim scoring
│   │   ├── maxsim_neon.S      # NEON SDOT assembly for MaxSim
│   │   ├── moltar_rag.c       # MaxSim search binary
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

### LLM Inference

```bash
# Build llama.cpp for the device
cd research/llama.cpp
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r27c/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33
cmake --build build --target llama-cli -j$(nproc)

# Push and run
adb push build/bin/llama-cli /data/local/tmp/
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor'"
adb shell "su -c 'echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor'"
adb shell "su -c 'stop'"  # kill Android framework, free ~4 GB/s DRAM BW
adb shell "su -c 'taskset c0 /data/local/tmp/llama-cli -m /data/local/tmp/LFM2-350M-Q4_0-pure.gguf -t 2 -p \"Hello\"'"
```

### Vector Database

```bash
# Cross-compile (from x86_64 host)
cd research/lcvdb
make all   # builds test_lcvdb, test_recall, test_bench

# Push and run
adb push test_lcvdb test_recall test_bench /data/local/tmp/
adb shell "su -c 'taskset c0 /data/local/tmp/test_lcvdb'"
```

### ColBERT RAG Pipeline

```bash
# Build ColBERT tools
cd research/colbert
make clean all

# Push to device (requires llama-cli, llama-embedding, libs already deployed)
adb push moltar_rag moltar_rag.sh /data/local/tmp/

# Run the demo (3 queries with context-grounded answers)
adb shell "su -c 'sh /data/local/tmp/moltar_rag.sh demo'"
```

## Key Discoveries

1. **Android framework is the bandwidth bottleneck** — SurfaceFlinger, SystemUI etc. consume ~4 GB/s of DRAM bandwidth. Running `su -c 'stop'` eliminates "thermal throttling" and stabilizes inference at full speed. A Magisk boot script automates this.

2. **Row-scaled quantization** — Custom Q4_0/Q8_0 block formats aligned to 64-byte cache lines, with pure-integer SDOT accumulation and power-of-2 shift activation quantization.

3. **Split storage for VDB** — Separating topology (64 bytes/node, M=16) from vectors (64 bytes/node) keeps the graph structure in L1D cache during traversal. At N=256, total working set is 32 KB.

4. **ColBERT late interaction for on-device RAG** — Per-token 128D embeddings with MaxSim scoring provide better retrieval than single-vector models. Full RAG pipeline (embed + search + generate) completes in ~2 seconds.

5. **Activation quant cache bug** — A cache keyed on buffer pointers produced stale data due to allocator address reuse, causing gibberish output. Removed entirely; benchmarks corrected by 2-4%.

## Documentation

| Document | What it covers |
|----------|---------------|
| [PRD.md](PRD.md) | Current state, next steps, acceptance criteria |
| [PERFORMANCE.md](PERFORMANCE.md) | All benchmark data with methodology |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design: kernels, VDB, device setup |
| [CHANGELOG.md](CHANGELOG.md) | Commit-level history of what changed |
| [HARDWARE_COMPATIBILITY.md](HARDWARE_COMPATIBILITY.md) | MT6855V specs, cache hierarchy, ISA details |
