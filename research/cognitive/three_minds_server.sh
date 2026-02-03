#!/system/bin/sh
#
# THREE MINDS V3 - Using llama-server for zero model load overhead
# Server must be running: llama-server on port 8080
#

SERVER="127.0.0.1:8080"

# Function to query llama-server
query_server() {
    local prompt="$1"
    local n_predict="${2:-30}"
    local outfile="/data/local/tmp/resp_$$.txt"
    
    # Build JSON request - escape quotes and newlines
    local escaped_prompt=$(echo "$prompt" | tr '\n' ' ' | sed 's/"/\\"/g' | tr -d '\r')
    local body="{\"prompt\":\"$escaped_prompt\",\"n_predict\":$n_predict,\"temperature\":0.7,\"stream\":false}"
    local len=$(echo -n "$body" | wc -c)
    
    # Make request to file (nc reads until connection closes)
    (echo -e "POST /completion HTTP/1.1\r\nHost: $SERVER\r\nContent-Type: application/json\r\nContent-Length: $len\r\nConnection: close\r\n\r\n$body"; sleep 3) | nc 127.0.0.1 8080 > "$outfile" 2>/dev/null
    
    # Extract content field - handle escaped newlines in response
    cat "$outfile" | tr '\r\n' ' ' | sed 's/.*"content":"\([^"]*\)".*/\1/' | sed 's/\\n/ /g' | head -c 500
    rm -f "$outfile"
}

echo ""
echo "========================================================================"
echo "    THREE MINDS V3 - Server Architecture (Zero Model Load)"
echo "========================================================================"
echo ""
echo "  Server: llama-server on A78-1 (CPU 7) @ 48 tok/s"
echo "  Model:  LFM2-350M persistent in DRAM"
echo "  Client: A55 thinkers query via HTTP"
echo ""
echo "========================================================================"
echo ""

# Check server health
echo "[Checking server...]"
HEALTH=$(echo -e "GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n" | nc -w 5 127.0.0.1 8080 2>/dev/null | tail -1)
if echo "$HEALTH" | grep -q "ok"; then
    echo "  Server: ONLINE"
else
    echo "  Server: OFFLINE - start with:"
    echo "  taskset 80 ./llama-server -m model.gguf -t 1 --port 8080 -np 1 --no-mmap"
    exit 1
fi
echo ""

QUERY="How should I approach a difficult conversation with my manager?"
echo "QUERY: $QUERY"
echo ""

# Timing
TIME_START=$(date +%s%N | cut -c1-13)

echo "========================================================================"
echo "  PARALLEL PHASE: Three Thinkers (via HTTP)"
echo "========================================================================"
echo ""

# Temp files for parallel output
C_OUT="/data/local/tmp/c3.txt"
A_OUT="/data/local/tmp/a3.txt"

# Launch Creative in background
echo "[CREATIVE thinker starting...]"
(
    query_server "Give ONE creative unexpected perspective on difficult workplace conversations. Be brief:" 35
) > $C_OUT 2>&1 &
C_PID=$!

# Launch Analytic in background  
echo "[ANALYTIC thinker starting...]"
(
    query_server "Give a 3-step structured approach to difficult conversations at work. Be brief:" 40
) > $A_OUT 2>&1 &
A_PID=$!

echo "[Waiting for parallel responses...]"
echo ""

wait $C_PID
wait $A_PID

echo "------------------------------------------------------------------------"
echo "CREATIVE:"
cat $C_OUT
echo ""
echo ""
echo "------------------------------------------------------------------------"
echo "ANALYTIC:"
cat $A_OUT
echo ""

echo ""
echo "========================================================================"
echo "  SYNTHESIS PHASE"  
echo "========================================================================"
echo ""

C_TEXT=$(cat $C_OUT | tr '\n' ' ' | head -c 150)
A_TEXT=$(cat $A_OUT | tr '\n' ' ' | head -c 150)

echo "[Synthesizing...]"
echo ""
SYNTH=$(query_server "Combine these briefly: Creative: $C_TEXT Analytic: $A_TEXT Synthesis:" 45)
echo "$SYNTH"
echo ""

TIME_END=$(date +%s%N | cut -c1-13)

echo "========================================================================"
echo "  PERFORMANCE"
echo "========================================================================"
echo ""

if [ -n "$TIME_START" ] && [ -n "$TIME_END" ]; then
    ELAPSED=$((TIME_END - TIME_START))
    echo "  Total time: ${ELAPSED}ms"
    echo "  (vs ~15-20 sec with model reload per query)"
else
    echo "  (timing not available)"
fi
echo ""
echo "  Key insight: Model stays in DRAM, only HTTP overhead (~1ms)"
echo ""

# Cleanup
rm -f $C_OUT $A_OUT
