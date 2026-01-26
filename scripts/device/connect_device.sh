#!/bin/bash

# Device Connection and Setup Script
# Essential device connection utilities for Motorola research devices

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR/../../tools/android"
PROJECT_ROOT="$SCRIPT_DIR/../.."

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Android platform tools
    if [ ! -f "$TOOLS_DIR/adb" ]; then
        log_error "ADB not found in $TOOLS_DIR/"
        log_error "Please ensure Android platform tools are available"
        exit 1
    fi
    log_success "ADB found at $TOOLS_DIR/adb"

    if [ ! -f "$TOOLS_DIR/fastboot" ]; then
        log_error "Fastboot not found in $TOOLS_DIR/"
        exit 1
    fi
    log_success "Fastboot found at $TOOLS_DIR/fastboot"
}

# Check device connection
check_device_connection() {
    log_info "Checking device connection..."

    local device_count=$("$TOOLS_DIR/adb" devices | grep -c "device$")
    if [ "$device_count" -eq 0 ]; then
        log_error "No Android device connected or authorized"
        log_info "Troubleshooting steps:"
        echo "1. Enable USB debugging in Developer Options"
        echo "2. Accept USB debugging authorization on device"
        echo "3. Try different USB cable/port"
        echo "4. Restart ADB: $TOOLS_DIR/adb kill-server && $TOOLS_DIR/adb start-server"
        exit 1
    fi

    local device_info=$("$TOOLS_DIR/adb" devices -l | grep "device$" | head -1)
    log_success "Device connected: $device_info"

    # Get device details
    local model=$("$TOOLS_DIR/adb" shell getprop ro.product.model 2>/dev/null || echo "Unknown")
    local android_version=$("$TOOLS_DIR/adb" shell getprop ro.build.version.release 2>/dev/null || echo "Unknown")
    local api_level=$("$TOOLS_DIR/adb" shell getprop ro.build.version.sdk 2>/dev/null || echo "Unknown")

    log_info "Device Info:"
    log_info "  Model: $model"
    log_info "  Android: $android_version (API $api_level)"
}

# Check root access
check_root_access() {
    log_info "Checking root access..."

    local root_check=$("$TOOLS_DIR/adb" shell su -c "whoami" 2>/dev/null)
    if [[ "$root_check" == "root" ]]; then
        log_success "Root access available"
        return 0
    else
        log_warning "Root access not available"
        log_info "Some research functionality may be limited without root"
        return 1
    fi
}

# Setup device for research
setup_device() {
    log_info "Setting up device for research..."

    # Create research directory on device
    "$TOOLS_DIR/adb" shell mkdir -p /data/local/tmp/moltar-research 2>/dev/null || true

    # Set permissions
    "$TOOLS_DIR/adb" shell chmod 755 /data/local/tmp/moltar-research 2>/dev/null || true

    log_success "Research directory created: /data/local/tmp/moltar-research"
}

# Collect device information
collect_device_info() {
    log_info "Collecting device information..."

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local info_dir="$PROJECT_ROOT/research/data/device_info_$timestamp"

    mkdir -p "$info_dir"

    # Collect various device information
    "$TOOLS_DIR/adb" shell getprop > "$info_dir/device_properties.txt"
    "$TOOLS_DIR/adb" shell cat /proc/cpuinfo > "$info_dir/cpuinfo.txt" 2>/dev/null || true
    "$TOOLS_DIR/adb" shell cat /proc/meminfo > "$info_dir/meminfo.txt" 2>/dev/null || true
    "$TOOLS_DIR/adb" shell df > "$info_dir/disk_usage.txt" 2>/dev/null || true

    # Root-specific information (if available)
    if "$TOOLS_DIR/adb" shell su -c "whoami" 2>/dev/null | grep -q "root"; then
        "$TOOLS_DIR/adb" shell su -c "cat /proc/version" > "$info_dir/kernel_version.txt" 2>/dev/null || true
        "$TOOLS_DIR/adb" shell su -c "ls -la /system/bin" > "$info_dir/system_binaries.txt" 2>/dev/null || true
    fi

    log_success "Device information collected in: $info_dir"
    echo "$info_dir"
}

# Test device responsiveness
test_device_responsiveness() {
    log_info "Testing device responsiveness..."

    # Test basic ADB commands
    if "$TOOLS_DIR/adb" shell echo "test" > /dev/null; then
        log_success "ADB shell access working"
    else
        log_error "ADB shell access failed"
        return 1
    fi

    # Test file operations
    if "$TOOLS_DIR/adb" shell "touch /data/local/tmp/test_file && rm /data/local/tmp/test_file" 2>/dev/null; then
        log_success "File operations working"
    else
        log_warning "File operations may be limited"
    fi

    # Test performance
    local start_time=$(date +%s%N)
    "$TOOLS_DIR/adb" shell "for i in {1..100}; do echo test > /dev/null; done" > /dev/null 2>&1
    local end_time=$(date +%s%N)
    local latency=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds

    log_info "Average command latency: ~${latency}ms"
}

# Main connection function
connect_device() {
    echo "🔌 Motorola Device Connection Setup"
    echo "=================================="

    check_prerequisites
    check_device_connection

    local has_root=0
    if check_root_access; then
        has_root=1
    fi

    setup_device
    test_device_responsiveness

    local info_dir=$(collect_device_info)

    echo ""
    echo "📱 Device Connection Summary:"
    echo "  ✅ Prerequisites verified"
    echo "  ✅ Device connected and responsive"
    if [ $has_root -eq 1 ]; then
        echo "  ✅ Root access available"
    else
        echo "  ⚠️  Root access not available"
    fi
    echo "  ✅ Research directory created"
    echo "  ✅ Device information collected"
    echo ""
    echo "📂 Device info saved to: $info_dir"
    echo ""
    echo "🚀 Device ready for research operations!"

    return 0
}

# Handle command line arguments
case "${1:-}" in
    "check")
        check_prerequisites
        check_device_connection
        check_root_access
        ;;
    "info")
        check_device_connection
        collect_device_info > /dev/null
        ;;
    "test")
        check_device_connection
        test_device_responsiveness
        ;;
    *)
        connect_device
        ;;
esac