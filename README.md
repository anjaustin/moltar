# Moltar

Embedded AI inference on a $200 phone. Custom llama.cpp kernels + an L-cache-resident vector database, targeting the Motorola Moto G Power 5G (2023).

## Results

**LLM inference** — Liquid AI LFM2 models on MediaTek Dimensity 930 (2x Cortex-A78 @ 2.2 GHz):

| Model | Quant | Size | tok/s (tg32) |
|-------|-------|------|-------------|
| LFM2-350M | Q4_0 | 190 MiB | **80.06** |
| LFM2-350M | Q8_0 | 359 MiB | **41.69** |
| LFM2-700M | Q4_0 | 423 MiB | **32.96** |
| LFM2-1.2B | Q4_0 | 661 MiB | **21.66** |

**Vector database** — HNSW search with NEON int8 dot products, split storage layout:

| N nodes | Search latency | QPS | Recall@5 | Working set |
|---------|---------------|-----|----------|-------------|
| 256 | 19.4 us | 51K | 99.6% | 24 KB (L1D) |
| 512 | 23.5 us | 42K | 94.4% | 48 KB (L1D) |
| 1024 | 24.3 us | 41K | 84.8% | 96 KB (L2) |

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
│   └── lcvdb/                  # L-Cache Vector Database
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

## Key Discoveries

1. **Android framework is the bandwidth bottleneck** — SurfaceFlinger, SystemUI etc. consume ~4 GB/s of DRAM bandwidth. Running `su -c 'stop'` eliminates "thermal throttling" and stabilizes inference at full speed. A Magisk boot script automates this.

2. **Row-scaled quantization** — Custom Q4_0/Q8_0 block formats aligned to 64-byte cache lines, with pure-integer SDOT accumulation and power-of-2 shift activation quantization.

3. **Split storage for VDB** — Separating topology (32 bytes/node) from vectors (64 bytes/node) keeps the graph structure in L1D cache during traversal. At N=256, the full topology is 8 KB.

## Documentation

| Document | What it covers |
|----------|---------------|
| [PRD.md](PRD.md) | Current state, next steps, acceptance criteria |
| [PERFORMANCE.md](PERFORMANCE.md) | All benchmark data with methodology |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design: kernels, VDB, device setup |
| [CHANGELOG.md](CHANGELOG.md) | Commit-level history of what changed |
| [HARDWARE_COMPATIBILITY.md](HARDWARE_COMPATIBILITY.md) | MT6855V specs, cache hierarchy, ISA details |
