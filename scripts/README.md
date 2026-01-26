# Research Scripts

This directory contains essential scripts for device connection, setup, and research operations.

## Device Scripts (`device/`)

### `connect_device.sh`
**Purpose**: Establish connection to Motorola research devices and verify functionality

**Usage**:
```bash
# Full connection and setup
./connect_device.sh

# Just check prerequisites
./connect_device.sh check

# Collect device information
./connect_device.sh info

# Test device responsiveness
./connect_device.sh test
```

**Prerequisites**:
- Android device with USB debugging enabled
- Authorized ADB connection
- Android platform tools in `../tools/android/`

### `setup_research_device.sh`
**Purpose**: Configure device for comprehensive research operations

**Usage**:
```bash
# Full research setup
./setup_research_device.sh

# Verify existing setup
./setup_research_device.sh verify
```

**Features**:
- Research directory structure creation
- Essential tool installation (BusyBox, etc.)
- Logging infrastructure setup
- Performance monitoring configuration
- Environment configuration

## Directory Structure Created on Device

```
/data/local/tmp/moltar-research/
├── data/           # Research data collection
├── logs/           # System and research logs
├── scripts/        # Research automation scripts
├── tools/          # Research utilities
├── research_env.sh # Environment configuration
└── bin/            # Additional binaries
```

## Research Environment

After setup, the research environment provides:
- Enhanced shell capabilities
- Automated logging and monitoring
- Performance profiling tools
- Data collection infrastructure

## Troubleshooting

### Connection Issues
- Ensure USB debugging is enabled
- Accept USB debugging authorization dialog
- Try different USB cables/ports
- Restart ADB server if needed

### Root Access
- Some features require root access
- Magisk or similar rooting solutions recommended
- Root status checked automatically

### Performance Issues
- Monitor battery levels during extended research
- Use performance monitoring scripts
- Adjust collection frequencies as needed

## Maintenance

Regular maintenance tasks:
- Log rotation (automated)
- Performance monitoring (automated)
- Data backup and archival
- Environment updates

## Security Considerations

- Research scripts operate with minimal privileges
- Root access only when necessary for specific research
- Data collection follows privacy guidelines
- Audit trails maintained for all operations