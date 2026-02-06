# Code Index: SRC-FFN and Evolving Neural Populations

A complete map of all code files in this project.

---

## Production Code

### C++ Implementations (ARM64/NEON)

| File | Purpose | Status | Performance |
|------|---------|--------|-------------|
| `six_core_llm.cpp` | Production inference with 6-core parallelism | **Working** | 74.8 tok/s |
| `dual_core_llm.cpp` | Dual-core inference | Working | 45.7 tok/s |
| `neon_ternary_llm.cpp` | Single-core NEON with ternary weights | Working | 23 tok/s |
| `src_ffn_engine.cpp` | SRC-FFN C++ implementation | Working | 35.6 tok/s |
| `src_ffn.cpp` | SRC-FFN exploration | Experimental | - |

### Failed/Abandoned C++ Approaches

| File | Purpose | Why Abandoned |
|------|---------|---------------|
| `gles_gemv.cpp` | GPU fragment shader GEMV | 0.5ms per-draw-call overhead |
| `texture_llm.cpp` | Texture-based matrix ops | Too slow |
| `compute_llm.cpp` | Compute shader approach | PowerVR limitations |
| `cfc_ternary_llm.cpp` | Combined CfC + ternary | Superseded by six_core |
| `spectral_cfc_llm.cpp` | Spectral approach | Experimental only |

---

## Python Implementations

### Original Implementations

| File | Purpose | Status |
|------|---------|--------|
| `evolving_src_ffn.py` | Original evolving populations (v1) | **Working** - proof of concept |
| `src_ffn_production.py` | SRC-FFN PyTorch reference | Working |
| `equilibrium_prop_analysis.py` | EP feasibility study | Working - gradients need debugging |
| `decay_experiment_v2.py` | Gate bias discovery | Working - key insight generator |
| `decay_experiment.py` | Earlier decay experiments | Superseded |
| `src_ffn_backward.py` | Backward pass analysis | Working |

### Scaled Implementation (evolving_v2/)

| File | Purpose | Status |
|------|---------|--------|
| `evolving_v2/vectorized_population.py` | Batched neuron computation | **Working** - 50-100x faster |
| `evolving_v2/model.py` | Full model assembly | **Working** |
| `evolving_v2/train.py` | Training script | **Working** - 97.9% accuracy achieved |

---

## Documentation

### PRDs (Product Requirements Documents)

| File | Topic | Status |
|------|-------|--------|
| `docs/PRD-001-SRC-FFN-Architecture.md` | Per-neuron CfC in FFN | Complete |
| `docs/PRD-002-Equilibrium-Propagation-Training.md` | O(1) memory training | Complete |
| `docs/PRD-003-Evolving-Neural-Populations.md` | Neural Darwinism | **Updated** with scaling results |
| `docs/DISCOVERY-SUMMARY.md` | Project overview | **Updated** with scaling results |
| `docs/CODE-INDEX.md` | This file | Current |

### Lincoln Manifold Analysis

| File | Phase | Content |
|------|-------|---------|
| `journal/scratchpad/evolving_populations_raw.md` | RAW | Unfiltered brain dump |
| `journal/scratchpad/evolving_populations_nodes.md` | NODES | 15 key insights extracted |
| `journal/scratchpad/evolving_populations_reflect.md` | REFLECT | Deep analysis, resolved tensions |
| `journal/scratchpad/evolving_populations_synth.md` | SYNTHESIZE | Concrete spec for scaling experiment |

---

## Experiment Results

### evolving_v2/experiments/

| Directory | Configuration | Result |
|-----------|--------------|--------|
| `test_001/` | 4 layers × 64 neurons, 10K tokens | 68.6% accuracy |
| `scaling_002/` | 8 layers × 256 neurons, 50K tokens | **97.9% accuracy** |
| `wikitext_001/` | WikiText-2 attempt | Partial (download timeout) |
| `scaling_final/` | 8 layers × 512 neurons | Partial (runtime timeout) |

---

## Build System

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Android NDK cross-compile for ARM64 |
| `build/` | Compiled binaries |

### Build Commands

```bash
export ANDROID_NDK=/path/to/ndk

cd build
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 ..
make six_core_llm -j4

# Deploy
adb push six_core_llm /data/local/tmp/
adb shell "./data/local/tmp/six_core_llm"
```

---

## Key Entry Points

### For Inference (Production)
```bash
# On Android device
./six_core_llm
```

### For Evolution Experiments (Research)
```bash
cd evolving_v2
python train.py --simple --max-tokens 50000 --epochs 3 --layers 8 --neurons 256
```

### For Understanding the Discovery
1. Read `docs/DISCOVERY-SUMMARY.md`
2. Read PRDs in order: 001 → 002 → 003
3. Read Lincoln Manifold analysis: raw → nodes → reflect → synth
4. Run `evolving_v2/train.py` to see evolution in action

---

## Dependency Graph

```
six_core_llm.cpp
└── NEON intrinsics (arm_neon.h)
└── pthreads (parallel execution)

evolving_v2/
├── vectorized_population.py
│   └── torch (PyTorch)
├── model.py
│   └── vectorized_population.py
└── train.py
    └── model.py
    └── datasets (optional, for WikiText)
    └── transformers (optional, for tokenizer)
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| Feb 4, 2026 | v1 | Initial discovery: 74.8 tok/s, SRC-FFN, evolving populations |
| Feb 5, 2026 | v2 | Scaled implementation: vectorized, 97.9% accuracy |

---

*Last updated: February 5, 2026*
