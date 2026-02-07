#!/bin/bash
# run_lfm_mt6855v.sh - Optimal LFM2-350M runner for Motorola MT6855V
#
# Uses Cortex-A78 big cores (cores 6-7) with optimal thread count
#
# CPU Topology on MT6855V (Dimensity 930):
#   Core 0-5: Cortex-A55 (little) - slower, used for background tasks
#   Core 6-7: Cortex-A78 (big) - faster, optimal for inference
#
# Optimal: 2 threads on big cores = 58+ tok/s (vs 36 tok/s on all 8 cores)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_BIN="${SCRIPT_DIR}/research/llama.cpp/build-android/bin/llama-bench"
LLAMA_CLI="${SCRIPT_DIR}/research/llama.cpp/build-android/bin/llama-cli"
MODEL="${SCRIPT_DIR}/data/LFM2-350M-Q4_0.gguf"

# Use big cores (0xC0 = cores 6 and 7)
BIG_CORES="c0"
THREADS=2

echo "=== MT6855V LFM2-350M Optimal Config ==="
echo "Affinity: Cortex-A78 big cores (6-7)"
echo "Threads:  ${THREADS}"
echo "Expected: ~58 tok/s"
echo ""

if [ ! -f "$LLAMA_BIN" ]; then
    echo "Error: llama-bench not found at $LLAMA_BIN"
    echo "Build with: ./research/llama.cpp/build-android.sh"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    exit 1
fi

if [ "$1" = "bench" ]; then
    echo "=== Benchmark ==="
    taskset ${BIG_CORES} "$LLAMA_BIN" -m "$MODEL" -p 64 -n 10 -t ${THREADS}
elif [ "$1" = "cli" ] || [ "$1" = "chat" ]; then
    shift
    echo "=== Interactive Chat ==="
    taskset ${BIG_CORES} "$LLAMA_CLI" -m "$MODEL" -t ${THREADS} "$@"
else
    echo "Usage: $0 {bench|cli|chat} [options]"
    echo ""
    echo "Examples:"
    echo "  $0 bench              # Run benchmark"
    echo "  $0 chat              # Start interactive chat"
    echo "  $0 chat -p \"Hello\"   # Single prompt"
    echo ""
    echo "Options passed to llama:"
    echo "  -p, --prompt PROMPT   Prompt string"
    echo "  -n, --tokens N        Number of tokens to generate"
    echo "  -t, --threads N       Number of threads (default: 2)"
fi
