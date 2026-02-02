# TriX LFM2 Frozen Chip

Standalone C implementation of the LFM2-350M forward pass, decomposed into
compositions of 5 atomic primes (ADD, MUL, EXP, MAX, CONST). Zero external
dependencies. Reads GGUF directly via mmap.

**Status:** Correct logits (top-5 matches llama.cpp). Unoptimized scalar C.
NEON optimization is next (see [ROADMAP.md](ROADMAP.md)).

## Quick Start

```bash
# Build (macOS or Linux host)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run unit tests (no model file needed)
./build/test_trix_lfm2       # 22/22 pass
./build/test_spline           # 10/10 pass

# Run with model (requires LFM2-350M-Q4_0.gguf)
./build/trix_lfm2_run ../../models/LFM2-350M-Q4_0.gguf

# Compare per-layer activations to llama.cpp reference
./build/debug_layers ../../models/LFM2-350M-Q4_0.gguf
```

## Model File

The model is `LFM2-350M-Q4_0.gguf` (209 MB), a Q4_0 quantization of Liquid
AI's LFM2-350M. Download from HuggingFace:

```bash
# Place in the repo models/ directory
huggingface-cli download LiquidAI/LFM2-350M-Q4_0-GGUF \
    LFM2-350M-Q4_0.gguf --local-dir ../../models/
```

Expected size: 219,306,944 bytes. GGUF v3, mixed quantization (Q6_K embedding,
F32 conv kernels, Q4_0 everything else). Output head is weight-tied to
embedding.

## Android Cross-Compile

```bash
NDK=~/Library/Android/sdk/ndk/28.2.13676358

cmake -B build-android \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-android -j

# Deploy to device
adb push build-android/trix_lfm2_run /data/local/tmp/
adb push ../../models/LFM2-350M-Q4_0.gguf /data/local/tmp/
adb shell "/data/local/tmp/trix_lfm2_run /data/local/tmp/LFM2-350M-Q4_0.gguf"
```

## Architecture

LFM2-350M has 16 layers in a hybrid recurrent/attention pattern:

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|-------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|
| Type  | S | S | A | S | S | A | S | S | A | S | A  | S  | A  | S  | A  | S  |

**S** = ShortConv (recurrent, conv_idx 0-9), **A** = Attention (causal, attn_idx 0-5)

Key constants: D_MODEL=1024, N_HEADS=16, N_KV_HEADS=8, D_HEAD=64,
FFN_HIDDEN=4608, VOCAB=65536, L_CACHE=3, D_CONV=2.

### Layer index mapping

```
Layer  0 -> shortconv[0]     Layer  8 -> attention[2]
Layer  1 -> shortconv[1]     Layer  9 -> shortconv[6]
Layer  2 -> attention[0]     Layer 10 -> attention[3]
Layer  3 -> shortconv[2]     Layer 11 -> shortconv[7]
Layer  4 -> shortconv[3]     Layer 12 -> attention[4]
Layer  5 -> attention[1]     Layer 13 -> shortconv[8]
Layer  6 -> shortconv[4]     Layer 14 -> attention[5]
Layer  7 -> shortconv[5]     Layer 15 -> shortconv[9]
```

## File Structure

```
include/
    lfm2_trix.h         # Types, constants, full API. Start here.
    trix_spline.h        # Spline activation tables (inline eval)
src/
    lfm2_trix.c          # Forward pass: matvec, norm, rope, shortconv, attn, ffn
    gguf_loader.c        # GGUF v3 parser, mmap, zero-copy weight mapping
    trix_spline.c        # Hermite spline coefficient fitting
tests/
    test_trix_lfm2.c     # Unit tests (22 tests, standalone)
    test_spline.c        # Spline accuracy + perf tests (10 tests)
    run_lfm2.c           # Token generator with timing
    debug_layers.c       # Per-layer activation dump (compare to llama.cpp)
    debug_gguf.c         # Conv kernel + forward pass debug
    debug_inproj.c       # Focused in_proj matvec analysis
    compare_weights.c    # Byte-level GGUF weight comparison
    cross_correlate.c    # Statistical correlation analysis
    dump_gguf_meta.c     # GGUF metadata dump
    test_q4_verify.c     # Q4_0 dequant verification
    test_prompt.c        # Multi-token prompt test
    llama_layer_dump.cpp # llama.cpp eval callback (requires llama.cpp)
    verify_inproj_weights.cpp  # Definitive dequant test (requires llama.cpp)
```

## Target Device

Motorola moto g power 5G (2023):

| Component | Spec | Source |
|-----------|------|--------|
| SoC | MediaTek MT6855 (Dimensity 930) | `getprop ro.hardware` |
| Big cores | 2x Cortex-A78 (64KB L1D, 256KB L2) | `/proc/cpuinfo` part 0xd41 |
| Little cores | 6x Cortex-A55 (32KB L1D, 128KB L2) | `/proc/cpuinfo` part 0xd05 |
| SIMD | NEON + dotprod, NO i8mm | cpuinfo features |
| GPU | PowerVR BXM-8-256 (Vulkan 1.1) | `/dev/pvr_sync` |
| RAM | 3.6 GB LPDDR4X-4266 (~13 GB/s) | `/proc/meminfo` |

## Documentation

- [PRD_trix_llama_enhancement.md](PRD_trix_llama_enhancement.md) — design rationale, hardware specs, optimization strategy
- [ROADMAP.md](ROADMAP.md) — phased implementation plan with success metrics
- [journal/q4_0_nibble_bug_postmortem.md](../../journal/q4_0_nibble_bug_postmortem.md) — debugging narrative for the Q4_0 nibble mapping bug
- [journal/scratchpad/](../../journal/scratchpad/) — LMM (Lincoln Manifold Method) debug notes
