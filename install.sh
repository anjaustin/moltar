#!/bin/bash

# Moltar Installation Script
# Sets up global access and shortcuts

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running on macOS
check_system() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_warning "This installation script is optimized for macOS"
        log_info "Continuing anyway..."
    fi
}

# Add to PATH
setup_path() {
    local shell_profile=""

    # Detect shell profile
    if [[ -n "$ZSH_VERSION" ]]; then
        shell_profile="$HOME/.zshrc"
    elif [[ -n "$BASH_VERSION" ]]; then
        shell_profile="$HOME/.bashrc"
    else
        shell_profile="$HOME/.profile"
    fi

    if [[ -f "$shell_profile" ]]; then
        if ! grep -q "moltar" "$shell_profile"; then
            echo "export PATH=\"\$PATH:$REPO_ROOT\"" >> "$shell_profile"
            log_success "Added moltar to PATH in $shell_profile"
            log_info "Run 'source $shell_profile' or restart your terminal"
        else
            log_info "Moltar already in PATH"
        fi
    else
        log_warning "Could not find shell profile to modify"
        log_info "You can manually add: export PATH=\"\$PATH:$REPO_ROOT\""
    fi
}

# Create desktop shortcut (macOS)
create_desktop_shortcut() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        return
    fi

    log_info "Creating desktop shortcut..."

    local desktop_dir="$HOME/Desktop"
    local shortcut_path="$desktop_dir/Moltar Setup.app"

    # Create a simple AppleScript application
    cat > /tmp/moltar_shortcut.scpt << EOF
tell application "Terminal"
    do script "cd '$REPO_ROOT' && ./moltar setup"
    activate
end tell
EOF

    # Convert to app bundle (requires osacompile)
    if command -v osacompile &> /dev/null; then
        osacompile -o "$shortcut_path" /tmp/moltar_shortcut.scpt 2>/dev/null && \
        log_success "Desktop shortcut created: $shortcut_path" || \
        log_warning "Could not create desktop shortcut"
    else
        log_warning "osacompile not found - skipping desktop shortcut"
    fi

    rm -f /tmp/moltar_shortcut.scpt
}

# Create symlink in /usr/local/bin (requires sudo)
create_symlink() {
    if [[ -w "/usr/local/bin" ]]; then
        ln -sf "$REPO_ROOT/moltar" "/usr/local/bin/moltar"
        log_success "Created symlink in /usr/local/bin"
    else
        log_info "Creating symlink requires sudo access"
        echo "Run: sudo ln -sf '$REPO_ROOT/moltar' /usr/local/bin/moltar"
        sudo ln -sf "$REPO_ROOT/moltar" "/usr/local/bin/moltar" && \
        log_success "Created symlink in /usr/local/bin" || \
        log_warning "Symlink creation failed"
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    # Check if moltar is accessible
    if command -v moltar &> /dev/null; then
        log_success "Moltar command is accessible"
        echo "Try: moltar help"
    else
        log_warning "Moltar command not found in PATH"
        log_info "You may need to restart your terminal or run: source ~/.zshrc"
    fi

    # Check if scripts are executable
    if [[ -x "$REPO_ROOT/moltar_setup.sh" ]]; then
        log_success "Setup script is executable"
    else
        log_warning "Setup script is not executable"
    fi
}

# Main installation
main() {
    echo -e "${GREEN}Moltar Installation${NC}"
    echo "=================="
    echo ""

    check_system

    echo "This will set up global access to moltar commands."
    echo ""
    read -p "Continue with installation? (y/N): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi

    setup_path
    create_symlink
    create_desktop_shortcut
    verify_installation

    echo ""
    log_success "Installation complete!"
    echo ""
    echo "You can now use:"
    echo "  moltar setup          # One-click device setup"
    echo "  moltar connect         # Connect to device"
    echo "  moltar help           # Show all commands"
    echo ""
    echo "Or run the desktop shortcut if created."
}

# Run installation
main