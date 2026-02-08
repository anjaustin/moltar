#!/bin/bash
# run_lfm_mt6855v.sh - Optimal LFM2-350M runner for Motorola MT6855V
#
# Runs llama-bench or llama-cli on the attached Motorola device via ADB
#
# CPU Topology on MT6855V (Dimensity 930):
#   Core 0-5: Cortex-A55 (little) - slower, used for background tasks
#   Core 6-7: Cortex-A78 (big) - faster, optimal for inference
#
# Optimal config: 2 threads on big cores + flash attention = ~59.82 tok/s

set -e

DEVICE_DIR="/data/local/tmp"

# Verify device connection
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected"
    echo "Check USB connection and USB debugging is enabled"
    exit 1
fi

# Use big cores (0xC0 = cores 6 and 7)
BIG_CORES="c0"
THREADS=2

echo "=== MT6855V LFM2-350M Optimal Config ==="
echo "Device:   $(adb shell getprop ro.product.model 2>/dev/null || echo 'unknown')"
echo "Affinity: Cortex-A78 big cores (6-7)"
echo "Threads:  ${THREADS}"
echo "Flash attn: ON"
echo "Expected: ~59.82 tok/s (Q4_0), varies by quant"
echo ""

MODEL_Q4="${DEVICE_DIR}/LFM2-350M-Q4_0.gguf"
MODEL_Q8="${DEVICE_DIR}/LFM2-350M-Q8_0.gguf"

if [ "$1" = "bench" ]; then
    MODEL="${2:-q4}"
    if [ "$MODEL" = "q8" ]; then
        GGUF="$MODEL_Q8"
    else
        GGUF="$MODEL_Q4"
    fi
    echo "=== Benchmark ($(basename $GGUF)) ==="
    adb shell "cd ${DEVICE_DIR} && LD_LIBRARY_PATH=. taskset ${BIG_CORES} ./llama-bench -m $(basename $GGUF) -t ${THREADS} -fa 1"

elif [ "$1" = "bench-all" ]; then
    echo "=== Benchmark All Quants ==="
    for gguf in "$MODEL_Q4" "$MODEL_Q8"; do
        name=$(basename "$gguf")
        if adb shell "test -f ${DEVICE_DIR}/${name}" 2>/dev/null; then
            echo ""
            echo "--- ${name} ---"
            adb shell "cd ${DEVICE_DIR} && LD_LIBRARY_PATH=. taskset ${BIG_CORES} ./llama-bench -m ${name} -t ${THREADS} -fa 1"
        else
            echo "Skipping ${name} (not on device)"
        fi
    done

elif [ "$1" = "cli" ] || [ "$1" = "chat" ]; then
    shift
    MODEL="${1:-q4}"
    if [ "$MODEL" = "q8" ]; then
        GGUF="$MODEL_Q8"
        shift 2>/dev/null || true
    else
        GGUF="$MODEL_Q4"
        # Only shift if it was a model selector, not a real argument
        [ "$MODEL" = "q4" ] && { shift 2>/dev/null || true; }
    fi
    echo "=== Interactive Chat ($(basename $GGUF)) ==="
    adb shell "cd ${DEVICE_DIR} && LD_LIBRARY_PATH=. taskset ${BIG_CORES} ./llama-cli -m $(basename $GGUF) -t ${THREADS} -fa 1 $*"

else
    echo "Usage: $0 {bench|bench-all|cli|chat} [q4|q8] [options]"
    echo ""
    echo "Examples:"
    echo "  $0 bench              # Benchmark Q4_0"
    echo "  $0 bench q8           # Benchmark Q8_0"
    echo "  $0 bench-all          # Benchmark all quants on device"
    echo "  $0 chat               # Interactive chat (Q4_0)"
    echo "  $0 chat q8            # Interactive chat (Q8_0)"
    echo "  $0 cli q4 -p 'Hello'  # Single prompt"
fi
