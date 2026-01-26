#!/bin/bash
# Download Liquid.ai LFM model from HuggingFace

MODEL_NAME="${1:-liquid-ai/LFM-2B-Chat}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
MODELS_DIR="$PROJECT_ROOT/models"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%H:%M:%S') - $1"
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

    # Check Python and huggingface_hub
    if ! command -v python3 >/dev/null 2>&1; then
        log_error "Python3 not found. Please install Python 3.8+"
        exit 1
    fi

    # Check if huggingface_hub is available
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        log_info "Installing huggingface_hub..."
        pip install huggingface_hub
    fi

    # Check git lfs for large files
    if ! command -v git-lfs >/dev/null 2>&1; then
        log_warning "Git LFS not found. Installing..."
        if command -v brew >/dev/null 2>&1; then
            brew install git-lfs
            git lfs install
        else
            log_error "Please install Git LFS manually: https://git-lfs.github.io/"
            exit 1
        fi
    fi

    log_success "Prerequisites verified"
}

# Download model from HuggingFace
download_model() {
    local model_name="$1"
    local local_dir="$MODELS_DIR/$(basename "$model_name")"

    log_info "Downloading LFN model: $model_name"
    log_info "Target directory: $local_dir"

    # Create models directory
    mkdir -p "$MODELS_DIR"

    # Download using huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
import os

model_name = '$model_name'
local_dir = '$local_dir'

print(f'Downloading {model_name} to {local_dir}...')
try:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print('Download complete!')
except Exception as e:
    print(f'Download failed: {e}')
    exit(1)
"

    if [ $? -eq 0 ]; then
        log_success "Model downloaded successfully"

        # List downloaded files
        echo ""
        echo "Downloaded files:"
        find "$local_dir" -type f \( -name "*.pte" -o -name "*.json" -o -name "*.bin" \) 2>/dev/null | head -10

        # Update config if config file exists
        if [ -f "$PROJECT_ROOT/config/lfm_config.json" ]; then
            log_info "Updating model configuration..."
            # Note: Would need more sophisticated JSON editing here
            log_info "Please verify config/lfm_config.json points to: $local_dir"
        fi

        return 0
    else
        log_error "Model download failed"
        return 1
    fi
}

# Main download function
main() {
    echo -e "${BLUE}🤖 BRACK LFN MODEL DOWNLOAD${NC}"
    echo "============================"
    echo ""

    if [ $# -eq 0 ]; then
        echo "Available LFN models on HuggingFace:"
        echo ""
        echo "2B Parameter Models:"
        echo "  liquid-ai/LFM-2B-Chat      # Conversational model"
        echo ""
        echo "7B Parameter Models:"
        echo "  liquid-ai/LFM-7B-Chat      # Advanced conversational"
        echo ""
        echo "40B Parameter Models:"
        echo "  liquid-ai/LFM-40B-Chat     # High-capability model"
        echo ""
        echo "Usage: $0 <model-name>"
        echo "Example: $0 liquid-ai/LFM-2B-Chat"
        echo ""
        echo "Note: Models are large (2-40GB). Ensure sufficient disk space."
        exit 0
    fi

    local model_name="$1"

    log_info "Starting LFN model download process..."
    log_info "Model: $model_name"

    check_prerequisites

    if download_model "$model_name"; then
        echo ""
        echo -e "${GREEN}🎉 MODEL DOWNLOAD COMPLETE!${NC}"
        echo ""
        echo "📁 Model location: $MODELS_DIR/$(basename "$model_name")"
        echo ""
        echo "🚀 Next steps:"
        echo "  1. Verify model files are present"
        echo "  2. Update config/lfm_config.json if needed"
        echo "  3. Build and deploy: ./scripts/build_debug.sh && ./scripts/deploy_device.sh"
        echo ""
        echo -e "${BLUE}Ready to deploy Liquid AI on Motorola! 🔬${NC}"
    else
        log_error "Model download failed. Please check your internet connection and try again."
        exit 1
    fi
}

main "$@"