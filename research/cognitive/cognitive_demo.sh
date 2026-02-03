#!/system/bin/sh
#
# Cognitive Architecture Demo - Real LLM Integration
#
# Demonstrates heterogeneous core usage with actual inference
#

MODEL="/data/local/tmp/LFM2-350M-Q4_0-pure.gguf"
LLAMA_DIR="/data/local/tmp/noprofile"

echo "======================================================================="
echo "       COGNITIVE ARCHITECTURE - Real LLM Demo"
echo ""
echo "  A78 (CPU 6-7): Fast Responder     A55 (CPU 0-5): Deep Thinker"
echo "======================================================================="
echo ""

cd $LLAMA_DIR

#=============================================================================
# Test 1: A78 Responder - FAST PATH
#=============================================================================
echo "=== TEST 1: A78 RESPONDER (Fast Path) ==="
echo "Core: CPU 7 (A78)"
echo "Use case: Reactive queries, tool calls, short answers"
echo ""

taskset 80 sh -c "LD_LIBRARY_PATH=. ./llama-bench -m $MODEL -t 1 -p 16 -n 32 -r 1 2>&1" | grep -E "model|pp16|tg32"

echo ""

#=============================================================================
# Test 2: A55 Thinker - EFFICIENT PATH
#=============================================================================
echo "=== TEST 2: A55 THINKER (Efficient Path) ==="
echo "Core: CPU 0-1 (A55 x 2)"
echo "Use case: Long generation, background processing, memory ops"
echo ""

taskset 03 sh -c "LD_LIBRARY_PATH=. ./llama-bench -m $MODEL -t 2 -p 16 -n 32 -r 1 2>&1" | grep -E "model|pp16|tg32"

echo ""

#=============================================================================
# Test 3: Extended generation comparison
#=============================================================================
echo "=== TEST 3: Extended Generation (128 tokens) ==="
echo ""

echo "A78 (CPU 7):"
taskset 80 sh -c "LD_LIBRARY_PATH=. ./llama-bench -m $MODEL -t 1 -n 128 -r 1 2>&1" | grep "tg128"

echo ""
echo "A55 (CPU 0-1):"
taskset 03 sh -c "LD_LIBRARY_PATH=. ./llama-bench -m $MODEL -t 2 -n 128 -r 1 2>&1" | grep "tg128"

echo ""

#=============================================================================
# Summary
#=============================================================================
echo "======================================================================="
echo "                           SUMMARY"
echo "======================================================================="
echo ""
echo "  A78 Responder:  ~48 tok/s | First token: ~25ms"
echo "                  For: Tool calls, short answers, UI snappiness"
echo ""
echo "  A55 Thinker:    ~18 tok/s | First token: ~600ms"
echo "                  For: Long generation, memory ops, background"
echo ""
echo "  Human reading:  ~4 tok/s  (250 words/minute)"
echo ""
echo "  KEY INSIGHT: A55 at 18 tok/s is 4.5x faster than reading!"
echo "               For long content, use A55 = same UX, 2x battery"
echo ""
echo "======================================================================="
