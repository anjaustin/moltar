#!/bin/bash

# Moltar One-Click Device Setup
# Automated device detection, preparation, and connection for newcomers

set -e

# Cleanup trap for signal handling
cleanup() {
    local exit_code=$?
    log_info "Cleaning up..."
    # Add cleanup logic here if needed
    exit $exit_code
}

trap cleanup EXIT ERR INT TERM

# Timeout wrapper for ADB commands
adb_command() {
    local timeout_seconds=30
    local cmd="$*"

    # Run command with timeout
    timeout $timeout_seconds bash -c "$cmd" 2>&1 || {
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log_error "ADB command timed out after ${timeout_seconds}s: $cmd"
        else
            log_error "ADB command failed: $cmd"
        fi
        return $exit_code
    }
}

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Bold and styles
BOLD='\033[1m'
DIM='\033[2m'
UNDERLINE='\033[4m'

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$REPO_ROOT/tools/android"
SCRIPTS_DIR="$REPO_ROOT/scripts/device"
LOG_FILE="$REPO_ROOT/setup_$(date +%Y%m%d_%H%M%S).log"

# Initialize logging
init_logging() {
    echo "==========================================" > "$LOG_FILE"
    echo "Moltar Device Setup Log" >> "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "==========================================" >> "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1" | tee -a "$LOG_FILE"
}

# ASCII Art Header
show_header() {
    echo -e "${CYAN}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                    🔬  M O L T A R   🔬                      ║
║              One-Click Device Research Setup                 ║
║                                                              ║
║  Automated Motorola Device Detection, Preparation & Setup    ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo ""
}

# System requirements check
check_system_requirements() {
    log_step "🔍 Checking system requirements..."

    # Check OS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_error "This setup script is designed for macOS"
        log_info "For other platforms, please adapt the script manually"
        exit 1
    fi
    log_success "macOS detected ✓"

    # Check for required tools
    local missing_tools=()

    if ! command -v brew &> /dev/null; then
        missing_tools+=("Homebrew")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_warning "Missing recommended tools: ${missing_tools[*]}"
        echo ""
        if [[ "$(prompt_user "Install missing tools automatically? (y/N): " "N")" =~ ^[Yy]$ ]]; then
            install_missing_tools "${missing_tools[@]}"
        fi
    fi

    log_success "System requirements verified"
}

# Install missing tools
install_missing_tools() {
    local tools=("$@")

    for tool in "${tools[@]}"; do
        case $tool in
            "Homebrew")
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                ;;
        esac
    done
}

# Device detection wizard
device_detection_wizard() {
    log_step "📱 Device Detection & Connection Wizard"

    echo ""
    echo -e "${BOLD}Device Connection Setup${NC}"
    echo "========================"
    echo ""
    echo "This wizard will help you connect your Motorola device."
    echo "Make sure your device is:"
    echo "  • Powered on"
    echo "  • USB cable connected to your Mac"
    echo ""

    # Check if device is already connected
    if adb_command "$TOOLS_DIR/adb devices 2>/dev/null" | grep -q "device$"; then
        log_success "Device already connected!"
        show_device_info
        return 0
    fi

    echo -e "${YELLOW}No device detected. Let's troubleshoot:${NC}"
    echo ""

    # Step-by-step troubleshooting
    echo "1. 📱 On your Motorola device:"
    echo "   • Go to Settings → About Phone"
    echo "   • Tap 'Build number' 7 times to enable Developer Options"
    echo "   • Go back to Settings → Developer Options"
    echo "   • Enable 'USB Debugging'"
    echo ""

    echo "2. 🔌 USB Connection:"
    echo "   • Try different USB ports on your Mac"
    echo "   • Try a different USB cable"
    echo "   • Ensure the cable supports data transfer (not charge-only)"
    echo ""

    echo "3. 🔓 Authorization:"
    echo "   • When you connect, your device may show 'Allow USB debugging?'"
    echo "   • Check 'Always allow from this computer'"
    echo "   • Tap 'Allow' or 'OK'"
    echo ""

    echo -n "Press Enter when you've completed the above steps..."
    read -r

    # Try to detect device again
    log_info "Retrying device detection..."
    if "$TOOLS_DIR/adb" devices 2>/dev/null | grep -q "device$"; then
        log_success "Device connected successfully!"
        show_device_info
    else
        log_error "Device still not detected"
        echo ""
        echo -e "${YELLOW}Additional troubleshooting:${NC}"
        echo "• Restart your Motorola device"
        echo "• Restart ADB: $TOOLS_DIR/adb kill-server && $TOOLS_DIR/adb start-server"
        echo "• Check device screen for authorization prompts"
        echo "• Try on a different computer"
        echo ""
        exit 1
    fi
}

# Show device information
show_device_info() {
    local device_info=$(adb_command "$TOOLS_DIR/adb shell getprop 2>/dev/null" | grep -E "(ro.product.model|ro.build.version.release|ro.product.manufacturer)" | sort)

    echo ""
    echo -e "${BOLD}Connected Device Information:${NC}"
    echo "$device_info" | while IFS=':' read -r key value; do
        echo "  ${key#*[}: ${value%]*}"
    done
    echo ""
}

# Root access setup wizard
root_setup_wizard() {
    log_step "🔑 Root Access Setup (Optional)"

    echo ""
    echo -e "${BOLD}Root Access Setup${NC}"
    echo "=================="
    echo ""

    # Check current root status
    if adb_command "$TOOLS_DIR/adb shell su -c \"whoami\" 2>/dev/null" | grep -q "root"; then
        log_success "Root access already available!"
        return 0
    fi

    echo "Root access provides enhanced research capabilities but is optional."
    echo "Some advanced features may be limited without root access."
    echo ""

    if [[ ! "$(prompt_user "Would you like to set up root access? (y/N): " "N")" =~ ^[Yy]$ ]]; then
        log_info "Skipping root setup. You can set it up later if needed."
        return 0
    fi

    echo ""
    echo -e "${YELLOW}Root Access Setup Instructions:${NC}"
    echo ""
    echo "1. 📥 Download Magisk Manager:"
    echo "   • Go to https://github.com/topjohnwu/Magisk/releases"
    echo "   • Download Magisk-vXX.X.apk (latest stable)"
    echo "   • Save to your Downloads folder"
    echo ""

    echo "2. 📱 Transfer to Device:"
    echo "   • The script will transfer Magisk to your device"
    echo ""

    echo "3. 🛠️  Install & Configure:"
    echo "   • Open the APK on your device to install Magisk Manager"
    echo "   • Follow Magisk's patching instructions"
    echo ""

    prompt_user "Press Enter when you have Magisk APK ready in Downloads..."

    # Look for Magisk APK
    local magisk_apk=""
    for apk in ~/Downloads/Magisk-*.apk; do
        if [[ -f "$apk" ]]; then
            magisk_apk="$apk"
            break
        fi
    done

    if [[ -z "$magisk_apk" ]]; then
        log_error "Magisk APK not found in Downloads folder"
        log_info "Please download Magisk manually and place it in Downloads"
        return 1
    fi

    log_info "Found Magisk APK: $(basename "$magisk_apk")"

    # Transfer to device
    log_info "Transferring Magisk to device..."
    "$TOOLS_DIR/adb" push "$magisk_apk" /sdcard/Download/ || {
        log_error "Failed to transfer Magisk APK"
        return 1
    }

    log_success "Magisk transferred successfully!"

    echo ""
    echo -e "${GREEN}Next steps on your device:${NC}"
    echo "1. Open your file manager"
    echo "2. Navigate to Downloads folder"
    echo "3. Tap on Magisk APK to install"
    echo "4. Open Magisk Manager"
    echo "5. Follow the patching instructions"
    echo ""

    prompt_user "Press Enter after completing Magisk setup on device..."

    # Verify root access
    if "$TOOLS_DIR/adb" shell su -c "whoami" 2>/dev/null | grep -q "root"; then
        log_success "Root access successfully configured!"
    else
        log_warning "Root access not yet detected"
        log_info "Root access may take effect after device reboot"
        log_info "Run this setup again after rebooting your device"
    fi
}

# Run automated setup
run_automated_setup() {
    log_step "🤖 Running Automated Research Setup"

    # Run connection script
    log_info "Establishing device connection..."
    if ! "$SCRIPTS_DIR/connect_device.sh"; then
        log_error "Device connection failed"
        return 1
    fi

    # Run research setup
    log_info "Configuring research environment..."
    if ! "$SCRIPTS_DIR/setup_research_device.sh"; then
        log_error "Research environment setup failed"
        return 1
    fi

    log_success "Automated setup completed!"
}

# Generate quick start guide
generate_quick_start() {
    local guide_file="$REPO_ROOT/QUICK_START.md"

    cat > "$guide_file" << EOF
# Moltar Quick Start Guide

## Your Device is Ready! 🎉

Your Motorola device has been successfully configured for research. Here's how to get started:

## Daily Research Workflow

### Start Your Research Session
\`\`\`bash
# Activate research environment on device
./scripts/device/connect_device.sh

# Or use the one-click setup for new sessions
./moltar_setup.sh --quick
\`\`\`

### Access Research Environment
\`\`\`bash
# Connect to device shell
./tools/android/adb shell

# Source research environment
source /data/local/tmp/moltar-research/research_env.sh

# Check available research tools
research-logs    # Go to logs directory
research-data    # Go to data directory
\`\`\`

### Monitor Performance
\`\`\`bash
# Run performance monitoring
/data/local/tmp/moltar-research/scripts/perf_monitor.sh
\`\`\`

## Research Directories

- \`/data/local/tmp/moltar-research/data/\` - Research data collection
- \`/data/local/tmp/moltar-research/logs/\` - System logs and monitoring
- \`/data/local/tmp/moltar-research/scripts/\` - Automation scripts

## Troubleshooting

### Device Not Detected
\`\`\`bash
# Restart ADB
./tools/android/adb kill-server && ./tools/android/adb start-server

# Full setup
./moltar_setup.sh
\`\`\`

### Permission Issues
- Ensure USB debugging is enabled
- Accept USB debugging authorization
- Check device screen for prompts

### Root Access Issues
- Verify Magisk is properly installed
- Check if device needs reboot after root setup
- Run root verification: \`./tools/android/adb shell su -c "whoami"\`

## Next Steps

1. **Explore Device Capabilities**: Use the research environment to investigate device features
2. **Collect Baseline Data**: Run performance monitoring to establish normal behavior
3. **Develop Research Tools**: Create custom scripts in the research environment
4. **Follow Methodology**: Refer to \`docs/methodology/RESEARCH_METHODOLOGY.md\`

## Support

- Check device logs: \`./tools/android/adb logcat\`
- Research environment logs: \`/data/local/tmp/moltar-research/logs/\`
- Setup logs: Check files matching \`setup_*.log\` in repository root

Happy researching! 🔬
EOF

    log_success "Quick start guide generated: $guide_file"
}

# Show completion summary
show_completion_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    🎉  S E T U P   C O M P L E T E  🎉      ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo -e "${BOLD}Your Motorola device is now ready for research!${NC}"
    echo ""
    echo "📋 What's been configured:"
    echo "  ✅ Device connection established"
    echo "  ✅ Research environment created"
    echo "  ✅ Logging and monitoring enabled"
    if "$TOOLS_DIR/adb" shell su -c "whoami" 2>/dev/null | grep -q "root"; then
        echo "  ✅ Root access configured"
    else
        echo "  ⚠️  Root access not available (optional)"
    fi
    echo "  ✅ Quick start guide generated"
    echo ""

    echo "🚀 Next steps:"
    echo "  1. Read QUICK_START.md for daily workflow"
    echo "  2. Start your research in the configured environment"
    echo "  3. Follow the methodology in docs/methodology/"
    echo ""

    echo "📁 Research directories on device:"
    echo "  /data/local/tmp/moltar-research/data/"
    echo "  /data/local/tmp/moltar-research/logs/"
    echo "  /data/local/tmp/moltar-research/scripts/"
    echo ""

    echo -e "${CYAN}Happy researching with Moltar! 🔬${NC}"
    echo ""
    echo "Setup log saved to: $LOG_FILE"
}

# Main setup function
main_setup() {
    init_logging
    show_header

    # Welcome message
    echo -e "${BOLD}Welcome to Moltar - One-Click Research Setup${NC}"
    echo ""
    echo "This setup will:"
    echo "  • Detect and connect your Motorola device"
    echo "  • Guide you through any necessary configuration"
    echo "  • Set up a complete research environment"
    echo "  • Generate documentation for future use"
    echo ""

    # Prerequisites check
    check_system_requirements

    # Device detection and connection
    device_detection_wizard

    # Optional root setup
    root_setup_wizard

    # Automated setup
    run_automated_setup

    # Generate documentation
    generate_quick_start

    # Completion summary
    show_completion_summary
}

# Quick mode for established setups
quick_mode() {
    log_info "Running quick setup mode..."

    if ! adb_command "$TOOLS_DIR/adb devices 2>/dev/null" | grep -q "device$"; then
        log_error "No device connected. Run full setup first."
        exit 1
    fi

    "$SCRIPTS_DIR/connect_device.sh" check
    log_success "Device ready for research!"
}

# Interactive prompt wrapper
prompt_user() {
    local prompt="$1"
    local default="${2:-N}"
    local response

    # Check if running non-interactively
    if [[ ! -t 0 ]] || [[ "${NON_INTERACTIVE:-false}" == "true" ]]; then
        log_info "Non-interactive mode: defaulting to '$default' for: $prompt"
        response="$default"
    else
        read -p "$prompt" -n 1 -r response
        echo ""  # New line after single char input
        response="${response:-$default}"
    fi

    # Return the response via echo for capture
    echo "$response"
}

# Parse command line arguments
case "${1:-}" in
    "--quick"|"-q")
        quick_mode
        ;;
    "--non-interactive"|"-ni")
        export NON_INTERACTIVE=true
        log_info "Running in non-interactive mode"
        main_setup
        ;;
    "--help"|"-h")
        echo "Moltar Device Setup"
        echo ""
        echo "Usage:"
        echo "  ./moltar_setup.sh                    # Full setup wizard (interactive)"
        echo "  ./moltar_setup.sh --quick           # Quick connect for established setups"
        echo "  ./moltar_setup.sh --non-interactive # Full setup without prompts"
        echo "  ./moltar_setup.sh --help            # Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  NON_INTERACTIVE=true                # Force non-interactive mode"
        ;;
    *)
        main_setup
        ;;
esac