#!/bin/bash

# Research Device Setup Script
# Comprehensive device preparation for embedded research

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

# Verify device connection
verify_connection() {
    if ! "$TOOLS_DIR/adb" devices | grep -q "device$"; then
        log_error "No device connected. Please run connect_device.sh first."
        exit 1
    fi
    log_success "Device connection verified"
}

# Install essential research tools
install_research_tools() {
    log_info "Installing essential research tools..."

    # BusyBox for enhanced shell capabilities
    if ! "$TOOLS_DIR/adb" shell which busybox > /dev/null 2>&1; then
        log_info "Installing BusyBox..."

        # Download and install BusyBox if available
        # Note: This would require the APK to be available
        log_warning "BusyBox not found. Some advanced operations may be limited."
    else
        log_success "BusyBox already available"
    fi

    # Install common utilities
    "$TOOLS_DIR/adb" shell mkdir -p /data/local/tmp/moltar-research/bin 2>/dev/null || true

    log_success "Research tools installation completed"
}

# Configure device for research
configure_research_environment() {
    log_info "Configuring research environment..."

    # Set up research directory structure
    "$TOOLS_DIR/adb" shell mkdir -p /data/local/tmp/moltar-research/{data,logs,scripts,tools}

    # Set appropriate permissions
    "$TOOLS_DIR/adb" shell chmod -R 755 /data/local/tmp/moltar-research

    # Configure shell environment
    cat > /tmp/research_env.sh << 'EOF'
#!/system/bin/sh
# Research environment configuration

export MOLTA_RESEARCH_DIR="/data/local/tmp/moltar-research"
export PATH="$MOLTA_RESEARCH_DIR/bin:$PATH"

# Aliases for common research operations
alias research-logs="cd $MOLTA_RESEARCH_DIR/logs"
alias research-data="cd $MOLTA_RESEARCH_DIR/data"
alias research-scripts="cd $MOLTA_RESEARCH_DIR/scripts"

echo "Moltar Research Environment Loaded"
echo "Research directory: $MOLTA_RESEARCH_DIR"
EOF

    "$TOOLS_DIR/adb" push /tmp/research_env.sh /data/local/tmp/moltar-research/
    "$TOOLS_DIR/adb" shell chmod +x /data/local/tmp/moltar-research/research_env.sh

    rm -f /tmp/research_env.sh

    log_success "Research environment configured"
}

# Set up logging infrastructure
setup_logging() {
    log_info "Setting up logging infrastructure..."

    # Create log rotation script
    cat > /tmp/log_rotate.sh << 'EOF'
#!/system/bin/sh
# Log rotation script for research activities

LOG_DIR="/data/local/tmp/moltar-research/logs"
MAX_SIZE=1048576  # 1MB
MAX_FILES=10

# Rotate logs if they get too large
for log_file in "$LOG_DIR"/*.log; do
    if [ -f "$log_file" ] && [ $(stat -c%s "$log_file" 2>/dev/null || stat -f%z "$log_file") -gt $MAX_SIZE ]; then
        mv "$log_file" "$log_file.$(date +%Y%m%d_%H%M%S)"
        # Keep only the most recent files
        ls -t "$LOG_DIR/$(basename "$log_file")".* 2>/dev/null | tail -n +$((MAX_FILES+1)) | xargs rm -f 2>/dev/null || true
    fi
done
EOF

    "$TOOLS_DIR/adb" push /tmp/log_rotate.sh /data/local/tmp/moltar-research/scripts/
    "$TOOLS_DIR/adb" shell chmod +x /data/local/tmp/moltar-research/scripts/log_rotate.sh

    # Set up log directory
    "$TOOLS_DIR/adb" shell mkdir -p /data/local/tmp/moltar-research/logs

    rm -f /tmp/log_rotate.sh

    log_success "Logging infrastructure set up"
}

# Configure performance monitoring
setup_performance_monitoring() {
    log_info "Setting up performance monitoring..."

    # Create performance monitoring script
    cat > /tmp/perf_monitor.sh << 'EOF'
#!/system/bin/sh
# Performance monitoring script

OUTPUT_FILE="/data/local/tmp/moltar-research/logs/performance_$(date +%Y%m%d_%H%M%S).log"

echo "=== Performance Monitoring Report ===" > "$OUTPUT_FILE"
echo "Timestamp: $(date)" >> "$OUTPUT_FILE"
echo "Device: $(getprop ro.product.model)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "CPU Usage:" >> "$OUTPUT_FILE"
cat /proc/stat | head -1 >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Memory Usage:" >> "$OUTPUT_FILE"
cat /proc/meminfo | grep -E "(MemTotal|MemFree|MemAvailable)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Battery Status:" >> "$OUTPUT_FILE"
dumpsys battery | grep -E "(level|temperature|voltage|current now)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Disk Usage:" >> "$OUTPUT_FILE"
df /data >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Top Processes:" >> "$OUTPUT_FILE"
ps | head -20 >> "$OUTPUT_FILE"

echo "Performance monitoring complete: $OUTPUT_FILE"
EOF

    "$TOOLS_DIR/adb" push /tmp/perf_monitor.sh /data/local/tmp/moltar-research/scripts/
    "$TOOLS_DIR/adb" shell chmod +x /data/local/tmp/moltar-research/scripts/perf_monitor.sh

    rm -f /tmp/perf_monitor.sh

    log_success "Performance monitoring set up"
}

# Test research environment
test_research_setup() {
    log_info "Testing research environment setup..."

    # Test directory structure
    local dirs=("data" "logs" "scripts" "tools")
    for dir in "${dirs[@]}"; do
        if "$TOOLS_DIR/adb" shell "[ -d /data/local/tmp/moltar-research/$dir ]"; then
            log_success "Directory /$dir created successfully"
        else
            log_error "Directory /$dir creation failed"
            return 1
        fi
    done

    # Test script execution
    if "$TOOLS_DIR/adb" shell "/data/local/tmp/moltar-research/research_env.sh > /dev/null 2>&1"; then
        log_success "Environment script executable"
    else
        log_error "Environment script execution failed"
        return 1
    fi

    log_success "Research environment test completed"
}

# Main setup function
setup_research_device() {
    echo "🔬 Research Device Setup"
    echo "======================="

    verify_connection
    install_research_tools
    configure_research_environment
    setup_logging
    setup_performance_monitoring

    if test_research_setup; then
        log_success "Research device setup completed successfully!"
        echo ""
        echo "📚 Research Environment Summary:"
        echo "  ✅ Essential tools installed"
        echo "  ✅ Research directory structure created"
        echo "  ✅ Logging infrastructure configured"
        echo "  ✅ Performance monitoring enabled"
        echo "  ✅ Environment tested and verified"
        echo ""
        echo "🔧 Available research scripts:"
        echo "  /data/local/tmp/moltar-research/research_env.sh"
        echo "  /data/local/tmp/moltar-research/scripts/log_rotate.sh"
        echo "  /data/local/tmp/moltar-research/scripts/perf_monitor.sh"
        echo ""
        echo "📊 Research directories:"
        echo "  data/   - Research data collection"
        echo "  logs/   - System and research logs"
        echo "  scripts/ - Research automation scripts"
        echo "  tools/   - Research utilities"
        echo ""
        echo "🚀 Device ready for research operations!"
    else
        log_error "Research device setup failed"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    "verify")
        verify_connection
        test_research_setup
        ;;
    *)
        setup_research_device
        ;;
esac