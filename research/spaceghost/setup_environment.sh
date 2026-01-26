#!/bin/bash

# SpaceGhost Environment Setup
# Configure development environment for ExecuTorch research and improvements

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
MOLTAR_ROOT="$PROJECT_ROOT/../.."

log_info() {
    echo -e "${BLUE}[SPACEGHOST]${NC} $(date '+%H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') - $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Python
    if ! command -v python3 >/dev/null 2>&1; then
        log_error "Python3 not found. Please install Python 3.8+"
        exit 1
    fi

    # Check Git
    if ! command -v git >/dev/null 2>&1; then
        log_error "Git not found. Please install Git"
        exit 1
    fi

    # Check CMake
    if ! command -v cmake >/dev/null 2>&1; then
        log_warning "CMake not found. Installing..."
        if command -v brew >/dev/null 2>&1; then
            brew install cmake
        else
            log_error "Please install CMake manually"
            exit 1
        fi
    fi

    log_success "Prerequisites verified"
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python environment..."

    cd "$SCRIPT_DIR"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Created Python virtual environment"
    fi

    # Activate and install dependencies
    source venv/bin/activate

    # Install basic development tools
    pip install --upgrade pip
    pip install wheel setuptools

    # Install ExecuTorch development dependencies
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Install additional development tools
    pip install numpy scipy matplotlib jupyterlab
    pip install black flake8 mypy pytest
    pip install huggingface_hub

    log_success "Python environment configured"
}

# Setup ExecuTorch development
setup_executorch_dev() {
    log_info "Setting up ExecuTorch development environment..."

    cd "$SCRIPT_DIR/executorch"

    # Initialize submodules
    if [ -f ".gitmodules" ]; then
        log_info "Initializing Git submodules..."
        git submodule update --init --recursive
    fi

    # Install Python dependencies for ExecuTorch
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "ExecuTorch Python dependencies installed"
    fi

    # Check for Android NDK (for Android builds)
    if [ -z "$ANDROID_NDK_HOME" ]; then
        log_warning "Android NDK not found. Android builds will be limited."
        log_info "To enable Android builds, install Android Studio and set ANDROID_NDK_HOME"
    else
        log_success "Android NDK found: $ANDROID_NDK_HOME"
    fi

    log_success "ExecuTorch development environment ready"
}

# Create research directories
setup_research_structure() {
    log_info "Setting up research structure..."

    cd "$SCRIPT_DIR"

    # Create research directories
    mkdir -p research/{hypotheses,experiments,results}
    mkdir -p patches/{dsp,memory,lfm,power}
    mkdir -p benchmarks/{baseline,optimized,comparison}
    mkdir -p integration/{brack,moltar}

    # Create initial research files
    touch research/hypotheses/dsp_optimization.md
    touch research/hypotheses/memory_efficiency.md
    touch research/hypotheses/lfm_integration.md
    touch research/hypotheses/power_management.md

    log_success "Research structure created"
}

# Configure development tools
setup_development_tools() {
    log_info "Setting up development tools..."

    cd "$SCRIPT_DIR"

    # Create .vscode settings if VS Code is available
    if command -v code >/dev/null 2>&1; then
        mkdir -p .vscode
        cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "cmake.configureOnOpen": true,
    "cmake.buildDirectory": "\${workspaceFolder}/executorch/cmake-out",
    "C_Cpp.default.compilerPath": "clang",
    "clangd.arguments": [
        "--compile-commands-dir=\${workspaceFolder}/executorch/cmake-out"
    ]
}
EOF
        log_success "VS Code configuration created"
    fi

    # Setup git hooks for ExecuTorch
    cd executorch
    if [ -d ".githooks" ]; then
        git config core.hooksPath .githooks
        log_success "Git hooks configured"
    fi

    log_success "Development tools configured"
}

# Test setup
test_setup() {
    log_info "Testing environment setup..."

    # Test Python environment
    source venv/bin/activate
    if python3 -c "import torch; print('PyTorch version:', torch.__version__)" >/dev/null 2>&1; then
        log_success "PyTorch installation working"
    else
        log_error "PyTorch installation failed"
    fi

    # Test ExecuTorch import
    if python3 -c "import executorch" >/dev/null 2>&1 2>/dev/null; then
        log_success "ExecuTorch import working"
    else
        log_warning "ExecuTorch import not available (expected for development builds)"
    fi

    # Test basic build capability
    if command -v cmake >/dev/null 2>&1 && command -v make >/dev/null 2>&1; then
        log_success "Build tools available"
    else
        log_warning "Build tools incomplete - some features may not work"
    fi

    log_success "Environment testing completed"
}

# Main setup function
main() {
    echo -e "${BLUE}🚀 SPACEGHOST ENVIRONMENT SETUP${NC}"
    echo "=================================="
    echo ""

    log_info "Setting up SpaceGhost research environment for ExecuTorch improvements..."

    check_prerequisites
    setup_python_env
    setup_executorch_dev
    setup_research_structure
    setup_development_tools
    test_setup

    echo ""
    echo -e "${GREEN}🎉 SPACEGHOST ENVIRONMENT SETUP COMPLETE!${NC}"
    echo ""
    echo "📁 Environment configured:"
    echo "  • Python virtual environment with PyTorch"
    echo "  • ExecuTorch development repository"
    echo "  • Research structure and documentation"
    echo "  • Development tools and VS Code integration"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Review research/INITIAL_ASSESSMENT.md"
    echo "  2. Begin with DSP optimization research"
    echo "  3. Build and test ExecuTorch: cd executorch && ./install_executorch.sh"
    echo "  4. Start developing improvements!"
    echo ""
    echo -e "${BLUE}Ready to optimize ExecuTorch for Liquid AI! 🔬${NC}"
}

main