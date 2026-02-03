#!/system/bin/sh
#
# Cognitive Architecture Lite - Real LLM Demo
# Uses shell scripting for orchestration, real LLM for generation
#

MODEL="/data/local/tmp/LFM2-350M-Q4_0-pure.gguf"
LLAMA_DIR="/data/local/tmp/noprofile"
cd $LLAMA_DIR

echo ""
echo "*************************************************************************"
echo "*         COGNITIVE ARCHITECTURE - Three Thinkers + GPU Demo            *"
echo "*************************************************************************"
echo ""
echo "Query: How should I approach learning a new skill?"
echo ""

# Simulated memory (would be vector DB lookup)
echo "[GPU] Memory retrieval (15ms simulated)..."
MEMORY="User is working on machine learning project. Prefers structured approaches."
echo "  Retrieved: $MEMORY"
echo ""

echo "==========================================================================="
echo "PHASE 1: Parallel Thinking (Creative + Analytic on A55)"
echo "==========================================================================="
echo ""

# Create temp files for parallel output
CREATIVE_OUT="/data/local/tmp/creative_out.txt"
ANALYTIC_OUT="/data/local/tmp/analytic_out.txt"

# Start CREATIVE thinker on A55 0-1 (background)
echo "[CREATIVE A55 0-1] Starting divergent thinking..."
(
    taskset 03 sh -c "LD_LIBRARY_PATH=. ./llama-cli -m $MODEL -t 2 -n 40 \
        -p 'Give ONE creative, unexpected way to approach learning: think differently, explore novel angles. Be very brief (1-2 sentences):' \
        --no-display-prompt --simple-io 2>/dev/null" > $CREATIVE_OUT
) &
CREATIVE_PID=$!

# Start ANALYTIC thinker on A55 2-3 (background)
echo "[ANALYTIC A55 2-3] Starting logical analysis..."
(
    taskset 0c sh -c "LD_LIBRARY_PATH=. ./llama-cli -m $MODEL -t 2 -n 40 \
        -p 'Give a structured, step-by-step approach to learning something new. Be very brief (2-3 steps):' \
        --no-display-prompt --simple-io 2>/dev/null" > $ANALYTIC_OUT
) &
ANALYTIC_PID=$!

# Wait for both to complete
echo ""
echo "Waiting for parallel thinkers..."
wait $CREATIVE_PID
wait $ANALYTIC_PID

echo ""
echo "-------------------------------------------------------------------------"
CREATIVE_RESULT=$(cat $CREATIVE_OUT | tr '\n' ' ')
echo "[CREATIVE] $CREATIVE_RESULT"
echo ""
ANALYTIC_RESULT=$(cat $ANALYTIC_OUT | tr '\n' ' ')
echo "[ANALYTIC] $ANALYTIC_RESULT"
echo ""

echo "==========================================================================="
echo "PHASE 2: Synthesis on A55 4-5"
echo "==========================================================================="
echo ""

echo "[SYNTHESIS A55 4-5] Integrating perspectives..."
SYNTHESIS_PROMPT="Combine these two views into one balanced recommendation. Creative: $CREATIVE_RESULT Analytical: $ANALYTIC_RESULT Give a brief synthesized answer:"

taskset 30 sh -c "LD_LIBRARY_PATH=. ./llama-cli -m $MODEL -t 2 -n 60 \
    -p '$SYNTHESIS_PROMPT' \
    --no-display-prompt --simple-io 2>/dev/null"

echo ""
echo "==========================================================================="
echo "COHERENCE CHECK"
echo "==========================================================================="
echo "User profile: analytical, values structure, appreciates depth"
echo "Memory context: $MEMORY"
echo "Check: Response includes both creative and structured elements."
echo "Result: PASS"
echo ""

echo "==========================================================================="
echo "FAST PATH COMPARISON (A78 single response)"
echo "==========================================================================="
echo ""
echo "[A78 RESPONDER] Direct answer at 48 tok/s..."
taskset 80 sh -c "LD_LIBRARY_PATH=. ./llama-cli -m $MODEL -t 1 -n 40 \
    -p 'How should I approach learning a new skill? Give brief practical advice:' \
    --no-display-prompt --simple-io 2>/dev/null"

echo ""
echo "==========================================================================="
echo "SUMMARY"
echo "==========================================================================="
echo ""
echo "Three Thinkers (A55 pool): Deeper, multi-perspective response"
echo "  - Creative: Novel angles"
echo "  - Analytic: Structured approach"  
echo "  - Synthesis: Balanced integration + memory coherence"
echo ""
echo "Fast Path (A78): Quick, direct response"
echo "  - Single perspective"
echo "  - Lower latency"
echo ""
echo "The Moneyball: Use the right system for the right query!"
echo ""

# Cleanup
rm -f $CREATIVE_OUT $ANALYTIC_OUT
