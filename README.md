# Moltar

A fresh start for auditable security research and embedded device development.

## LFM2-350M on MT6855V (Dimensity 930) - 60 tok/s

**Optimized llama.cpp inference for Motorola Moto G Power 5G 2023**

### Performance Results

| Metric | Performance |
|--------|-------------|
| Token Generation | **59.82 tok/s** |
| Prompt Processing | **260.65 tok/s** |
| Model | LFM2-350M Q4_0 (207 MB) |
| Context | 128K supported |

### Optimization Journey

| Configuration | Token Gen | Improvement |
|---------------|-----------|-------------|
| Without DOTPROD | 32 tok/s | baseline |
| With DOTPROD | 57.50 tok/s | +80% |
| **DOTPROD + Flash Attention** | **59.82 tok/s** | **+87%** |

### Critical Build Configuration

The MT6855V (Dimensity 930) has **DOTPROD support** (`asimddp` in `/proc/cpuinfo`). You MUST enable it:

```bash
# Build with DOTPROD enabled (CRITICAL for 60 tok/s)
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_NATIVE_API_LEVEL=24 \
  -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod+fp16" \
  -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod+fp16" \
  -DCMAKE_BUILD_TYPE=Release

make -j8 llama-bench
```

### Optimal Runtime Configuration

```bash
# Best performance: big cores + 2 threads + flash attention
taskset c0 ./llama-bench -m LFM2-350M-Q4_0.gguf -t 2 -fa 1

# c0 = 0xC0 = cores 6-7 (big Cortex-A78 cores)
# Cores 0-5 are little Cortex-A55 cores - avoid them for inference
# -fa 1 = enable flash attention (+3-4% speedup)
```

### CPU Topology

```
MT6855V (Dimensity 930):
  Cores 0-5: Cortex-A55 (little) @ 2.0 GHz
  Cores 6-7: Cortex-A78 (big) @ 2.2 GHz
  
Features: fp asimd aes pmull sha1 sha2 crc32 atomics fphp asimdhp asimddp
          ^^^^^^^^
          DOTPROD SUPPORTED - use -march=armv8.2-a+dotprod+fp16
```

### Performance Analysis (simpleperf profiling)

| Function | % Cycles | Notes |
|----------|----------|-------|
| `ggml_gemv_q4_0_4x4_q8_0` | 35.1% | Q4_0 GEMV - uses DOTPROD |
| `ggml_vec_dot_q6_K_q8_K` | 18.3% | Token embedding (Q6_K) |
| `ggml_gemm_q4_0_4x4_q8_0` | 11.2% | Batch GEMM |
| OpenMP synchronization | 7.4% | Thread sync overhead |
| Tensor repacking | 3.5% | 4x4 block format |

The model uses Q4_0 for weights (92 tensors) and Q6_K for the token embedding (1 tensor).
Q6_K provides better quality for the frequently-accessed embedding layer.

### Why This is Near-Optimal

1. **DOTPROD enabled**: Using `vdotq_laneq_s32` instruction
2. **INT accumulation**: Accumulates in int32, converts to float once per block
3. **4x4 blocking**: Optimal for NEON register utilization
4. **2 threads optimal**: Single thread = 38.84 tok/s (32% slower)
5. **Flash attention**: Reduces memory bandwidth pressure

Further gains would require:
- Hardware with i8mm support (INT8 matrix multiply)
- Re-quantizing model to pure Q4_0 (quality tradeoff)
- Smaller model variant

### Performance Without DOTPROD

Without the `+dotprod` flag, performance drops to ~32 tok/s (87% slower than optimal).

---

## About

This repository provides a clean foundation for conducting rigorous, reproducible security research with proper methodological controls. It includes essential tools for device connection and research environment setup.

## Repository Structure

```
moltar/
├── moltar                    # Command launcher (./moltar setup)
├── moltar_setup.sh          # One-click setup wizard
├── install.sh               # Global installation script
├── tools/                    # Research tools and utilities
│   └── android/             # Android platform tools (ADB, Fastboot)
├── scripts/                 # Research automation scripts
│   └── device/              # Device connection and setup scripts
├── docs/                    # Research documentation and methodology
│   └── methodology/         # Research method frameworks
├── research/                # Active research data and analysis
│   ├── data/               # Research data collections
│   └── analysis/           # Research analysis results
├── logs/                    # Setup and debug logs
├── archive/                # Historical falsified research (ignored)
├── CHANGELOG.md            # Complete change history
└── README.md               # This file
```

## Command Launcher

Use the `moltar` command for easy access to all functionality:

```bash
# One-click setup
./moltar setup

# Quick connect
./moltar setup --quick

# Device operations
./moltar connect         # Connect to device
./moltar research        # Setup research environment
./moltar device info     # Show device information
./moltar device test     # Test connectivity

# Help
./moltar help
```

### Global Installation (Optional)

Make moltar available system-wide:

```bash
# Add to your shell profile
echo 'export PATH="$PATH:/path/to/moltar"' >> ~/.zshrc

# Or create a symlink
sudo ln -s /path/to/moltar/moltar /usr/local/bin/moltar

# Then use from anywhere
moltar setup
```

## 🚀 One-Click Setup (Recommended for Newcomers)

### First-Time Setup
```bash
# One-click automated setup - handles everything!
./moltar_setup.sh
```

This wizard will:
- ✅ Detect your Motorola device
- ✅ Guide you through USB debugging setup
- ✅ Handle ADB authorization
- ✅ Configure root access (optional)
- ✅ Set up complete research environment
- ✅ Generate quick-start documentation

### Daily Research Sessions
```bash
# Quick connect for established setups
./moltar_setup.sh --quick

# Or use the command launcher
./moltar setup --quick
```

## Manual Setup (Advanced Users)

### 1. Connect Your Device
```bash
# Connect and verify Motorola device
./scripts/device/connect_device.sh

# Set up research environment
./scripts/device/setup_research_device.sh
```

### 2. Verify Setup
```bash
# Check connection status
./scripts/device/connect_device.sh check

# Verify research environment
./scripts/device/setup_research_device.sh verify
```

## Research Methodology

This repository follows rigorous scientific methodology for security research:

- **Pre-registered protocols** before data collection
- **Falsification-first approach** to hypothesis testing
- **Independent validation** of all claims
- **Complete audit trails** for reproducibility
- **Statistical rigor** in experimental design

See `docs/methodology/RESEARCH_METHODOLOGY.md` for detailed guidelines.

## Essential Tools

### Device Connection
- **ADB/Fastboot**: Platform tools for device communication
- **Connection Scripts**: Automated device detection and setup
- **Root Access Tools**: Enhanced device capabilities (when available)

### Research Infrastructure
- **Logging Systems**: Automated data collection and monitoring
- **Performance Profiling**: Device and system performance analysis
- **Environment Configuration**: Consistent research environments

## Archive Warning

The `archive/` directory contains materials from falsified entropy-based security research. These are preserved for educational reference only.

**⚠️ WARNING:** All approaches in the archive have been systematically proven non-viable. Do not attempt to implement or deploy any archived code.

## Getting Started with Research

1. **Define Your Problem**: Start with real security challenges, not solution invention
2. **Formulate Hypotheses**: Create falsifiable, testable claims
3. **Design Experiments**: Use proper statistical methods and controls
4. **Execute Rigorously**: Follow pre-registered protocols exactly
5. **Validate Independently**: External verification of all results

## Contribution Guidelines

- Follow the research methodology framework
- Pre-register experimental protocols
- Maintain complete audit trails
- Practice open science principles
- Submit negative results alongside positive findings

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md) for complete version history, breaking changes, and repository evolution.

---

## 📚 Documentation

### Core Documentation
- **[docs/INDEX.md](docs/INDEX.md)** - Start here (docs entry point)
- **[INSTALL.md](INSTALL.md)** - Complete installation and setup guide
- **[API.md](API.md)** - API reference for all tools and interfaces
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **[docs/TROUBLESHOOTING_GUIDE.md](docs/TROUBLESHOOTING_GUIDE.md)** - Comprehensive deployment troubleshooting
- **[PERFORMANCE.md](PERFORMANCE.md)** - Performance benchmarks and optimization
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development workflow and guidelines
- **[ROADMAP.md](ROADMAP.md)** - Project vision and future plans
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design
- **[FAQ.md](FAQ.md)** - Frequently asked questions
- **[GLOSSARY.md](GLOSSARY.md)** - Technical terms and definitions
- **[CHANGELOG.md](CHANGELOG.md)** - Complete change history and version releases
- **[MAINTAINERS.md](MAINTAINERS.md)** - Maintenance and governance guidelines
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines and research standards
- **[SECURITY.md](SECURITY.md)** - Security policy and responsible disclosure
- **[LICENSE](LICENSE)** - Research license terms and conditions
- **[docs/README.md](docs/README.md)** - Documentation overview and guides
- **[docs/methodology/RESEARCH_METHODOLOGY.md](docs/methodology/RESEARCH_METHODOLOGY.md)** - Research standards and validation frameworks
- **[docs/SPACEGHOST_BRACK_INTEGRATION.md](docs/SPACEGHOST_BRACK_INTEGRATION.md)** - SpaceGhost + Brack integration guide
- **[docs/LFN_DEPLOYMENT_GUIDE.md](docs/LFN_DEPLOYMENT_GUIDE.md)** - Complete LFN model deployment with SpaceGhost optimizations
- **[docs/SPACEGHOST_ARCHITECTURE.md](docs/SPACEGHOST_ARCHITECTURE.md)** - Detailed SpaceGhost optimization architecture
- **[docs/SPACEGHOST_EXAMPLES.md](docs/SPACEGHOST_EXAMPLES.md)** - Practical code examples for SpaceGhost usage
- **[docs/SPACEGHOST_MIGRATION_GUIDE.md](docs/SPACEGHOST_MIGRATION_GUIDE.md)** - Migration guide for adopting SpaceGhost optimizations
- **[scripts/README.md](scripts/README.md)** - Device connection and automation guides

### Generated Documentation
- **QUICK_START.md** - Auto-generated during setup with device-specific instructions
- **deployment_report.md** - Generated during device setup with performance metrics

---

## 🔬 Research Standards

This repository adheres to rigorous scientific methodology:

- **Pre-registered protocols** before data collection
- **Falsification-first approach** to hypothesis testing
- **Independent validation** of all claims
- **Complete audit trails** for reproducibility
- **Statistical rigor** in experimental design

---

**Repository:** moltar
**Started:** January 26, 2026
**Methodology:** Systematic falsification and rigorous validation
**Focus:** Auditable security research on embedded devices
**Version:** 1.0.0