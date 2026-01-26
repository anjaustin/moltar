# Moltar

A fresh start for auditable security research and embedded device development.

## About

This repository provides a clean foundation for conducting rigorous, reproducible security research with proper methodological controls. It includes essential tools for device connection and research environment setup.

## Repository Structure

```
moltar/
├── moltar                    # Command launcher (./moltar setup)
├── moltar_setup.sh          # One-click setup wizard
├── tools/                    # Research tools and utilities
│   └── android/             # Android platform tools (ADB, Fastboot)
├── scripts/                 # Research automation scripts
│   └── device/              # Device connection and setup scripts
├── docs/                    # Research documentation and methodology
│   └── methodology/         # Research method frameworks
├── research/                # Active research data and analysis
│   ├── data/               # Research data collections
│   └── analysis/           # Research analysis results
└── archive/                # Historical falsified research (ignored)
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

---

**Repository:** moltar
**Started:** January 26, 2026
**Methodology:** Systematic falsification and rigorous validation
**Focus:** Auditable security research on embedded devices