# Frequently Asked Questions

## General Questions

### What is Moltar?

Moltar is a research platform for conducting rigorous, reproducible security research with embedded systems. It provides tools and methodologies for scientific validation of security claims, with a focus on mobile and embedded device security.

### Why "Moltar"?

Moltar is named after the Motorola devices used in the initial research, combining "Motorola" with "altar" to signify a platform for serious security research.

### Who can use Moltar?

Moltar is designed for:
- **Security Researchers**: Academic and industry researchers
- **Security Engineers**: Professionals developing security solutions
- **Students**: Learning security research methodologies
- **Open Source Contributors**: Anyone interested in security research

## Getting Started

### How do I get started with Moltar?

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anjaustin/moltar.git
   cd moltar
   ```

2. **Run the setup script**:
   ```bash
   ./moltar_setup.sh
   ```

3. **Follow the installation guide** in `INSTALL.md`
4. **Use the docs index** for the right path:
   - `docs/INDEX.md`

### What are the system requirements?

- **OS**: macOS 11+, Linux, or Windows with WSL2
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Network**: Stable internet connection

### Do I need Android development experience?

Not necessarily. Moltar provides automated setup and deployment scripts. However, basic Android concepts are helpful for advanced usage.

## Research Methodology

### What makes Moltar different from other security tools?

Moltar emphasizes **scientific rigor** and **reproducibility**:
- Pre-registered experimental protocols
- Systematic falsification testing
- Statistical validation of claims
- Complete audit trails

### How does falsification testing work?

Falsification testing systematically attempts to disprove performance and security claims:

1. **State the claim** (e.g., "latency <200ms")
2. **Design falsification tests** to disprove the claim
3. **Collect evidence** through experimentation
4. **Draw conclusions** based on empirical results

### Can I use Moltar for commercial research?

Yes, but with restrictions. See `LICENSE` for commercial use terms. Contact the maintainers for commercial licensing inquiries.

## Technical Questions

### What devices does Moltar support?

Currently validated for:
- **Motorola moto g power 5G (2023)**  
  - **SoC**: MediaTek Dimensity 930 (MT6855V)  
  - **GPU**: PowerVR BXM-8-256
- **Android 12+ (API 31+)** devices (arm64)
- USB debugging enabled devices

Notes:
- Some documentation and optimization workstreams mention Snapdragon/DSP paths; treat those as **device-specific** and not automatically applicable to the Dimensity/PowerVR device class.
- For the authoritative compatibility matrix, see `HARDWARE_COMPATIBILITY.md`.

### What AI models does Moltar support?

Currently supports:
- **Liquid.ai LFM2-350M**: Optimized for mobile deployment
- Framework extensible to other ExecuTorch-compatible models

### How does ExecuTorch integration work?

ExecuTorch provides on-device AI inference:
- **Hardware acceleration** on Snapdragon DSP/GPU
- **Low latency** inference (<200ms target)
- **Memory efficient** operation (<256MB)
- **Cross-platform** Android deployment

### Can I add my own research modules?

Yes! Moltar has a modular architecture. See `DEVELOPMENT.md` for adding custom research modules.

## Device-Specific Questions

### Why Motorola devices specifically?

Motorola devices were chosen for initial research because:
- **Unlocked bootloader** options available
- **Mainstream Android** implementation
- **Consistent hardware** specifications
- **Research accessibility** for the team

### Do I need root access?

Root access is **recommended but not required**:
- **With root**: Full system access, advanced research capabilities
- **Without root**: Limited to user-space research, still valuable

### Can I use Moltar without physical hardware?

Yes, for development and testing:
- **Simulation mode**: Test deployment workflows
- **Emulator support**: Android Studio emulator
- **Cloud testing**: Future remote device access

## Performance & Optimization

### What are the performance targets?

Current targets for LFM2-350M on mobile-class Android devices:
- **Latency**: <200ms response time
- **Memory**: <256MB RAM usage
- **Storage**: ~500MB model size
- **Battery**: <5% additional drain

### How does Moltar optimize for mobile?

- **Model quantization**: 4-bit precision for smaller size
- **DSP acceleration**: Hardware-accelerated inference
- **Memory management**: Efficient resource allocation
- **Power optimization**: Battery-aware operation

### What affects performance?

Performance depends on:
- **Device hardware** (Snapdragon 480+ recommended)
- **Android version** (12+ recommended)
- **Available RAM** (4GB+ recommended)
- **Background processes** (minimize for testing)
- **Model size** (smaller models = better performance)

## Development & Contribution

### How do I contribute to Moltar?

1. **Read the guidelines**: `CONTRIBUTING.md`
2. **Fork the repository**
3. **Create a feature branch**
4. **Make your changes**
5. **Add tests**
6. **Submit a pull request**

### What programming languages are used?

- **Python**: Research scripts, automation, analysis
- **Kotlin**: Android application development
- **Shell/Bash**: Build scripts, deployment automation
- **C++**: Native performance-critical components

### How do I report bugs?

1. **Check existing issues** on GitHub
2. **Create a new issue** with detailed information:
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Log files
3. **Use appropriate labels** (bug, enhancement, etc.)

## Research & Science

### How does Moltar ensure research reproducibility?

- **Version control**: All code and data tracked
- **Environment specification**: Exact dependency versions
- **Protocol documentation**: Pre-registered experimental designs
- **Audit trails**: Complete activity logging

### What statistical methods does Moltar use?

- **Hypothesis testing**: Null/alternative hypothesis formulation
- **Effect size calculation**: Meaningful result quantification
- **Confidence intervals**: Uncertainty quantification
- **Power analysis**: Appropriate sample size determination

### How do I validate my research results?

Use the built-in falsification framework:

```bash
# Test your performance claims
./research/brack/scripts/falsify_performance_claims.sh

# Generate validation reports
./scripts/generate_research_report.sh
```

## Troubleshooting

### Why can't I connect to my device?

Common solutions:
- **Enable USB debugging** in Developer Options
- **Accept authorization dialog** on device
- **Try different USB ports/cables**
- **Restart ADB**: `adb kill-server && adb start-server`
- **Check device screen** for prompts

### Why are builds failing?

Common causes:
- **Missing dependencies**: Run `./moltar_setup.sh`
- **Java version mismatch**: Ensure JDK 11+
- **Android SDK issues**: Verify `$ANDROID_HOME` path
- **Gradle cache problems**: Run `./gradlew clean`

### Why is performance worse than expected?

Possible factors:
- **Background processes**: Close other apps
- **Device heating**: Allow cool-down period
- **Memory pressure**: Close memory-intensive apps
- **Android optimization**: Device may optimize performance
- **Model size**: Larger models = slower performance

## Future & Roadmap

### What's planned for future versions?

See `ROADMAP.md` for detailed plans:
- **Multi-device support** (iOS, wearables, IoT)
- **Advanced AI models** (vision, audio, multi-modal)
- **Research automation** (automated testing, benchmarking)
- **Enterprise features** (compliance, deployment)

### How can I stay updated?

- **GitHub**: Watch the repository for releases
- **Issues**: Subscribe to relevant issues
- **Discussions**: Participate in community discussions
- **Changelog**: Review `CHANGELOG.md` for updates

### Can I suggest new features?

Absolutely! Open a GitHub issue with the "enhancement" label and provide:
- **Use case**: What problem does it solve?
- **Implementation**: How it could work
- **Benefits**: Why it's valuable
- **Requirements**: Any prerequisites

## Support

### Where can I get help?

- **Documentation**: Start with `README.md` and `TROUBLESHOOTING.md`
- **GitHub Issues**: Search existing issues or create new ones
- **GitHub Discussions**: Ask questions in community discussions
- **Contributing Guide**: `CONTRIBUTING.md` for development help

### Is there a community?

Yes! Moltar encourages community participation:
- **Open source development**
- **Research collaboration**
- **Knowledge sharing**
- **Peer review and validation**

### How do I contact the maintainers?

- **GitHub Issues**: For bugs, features, and general questions
- **Email**: iam@anjaustin.com for security or sensitive matters
- **Repository**: https://github.com/anjaustin/moltar

---

*This FAQ is regularly updated. If your question isn't answered here, please check the documentation or create a GitHub issue.*