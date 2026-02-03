#!/system/bin/sh
#
# Simple HTTP client for llama-server on Android
# Uses /dev/tcp if available, falls back to nc
#

SERVER_HOST="127.0.0.1"
SERVER_PORT="8080"

# Function to make HTTP request
http_post() {
    local endpoint="$1"
    local body="$2"
    local len=$(echo -n "$body" | wc -c)
    
    # Create request
    local request="POST $endpoint HTTP/1.1\r\nHost: $SERVER_HOST\r\nContent-Type: application/json\r\nContent-Length: $len\r\nConnection: close\r\n\r\n$body"
    
    # Send request and capture response body (skip headers)
    response=$(echo -e "$request" | nc -w 10 $SERVER_HOST $SERVER_PORT 2>/dev/null)
    
    # Extract JSON body (after blank line)
    echo "$response" | sed '1,/^[[:space:]]*$/d'
}

# Health check
echo "=== Health Check ==="
http_post "/health" "{}"
echo ""

# Completion request
echo "=== Completion Test ==="
echo "Prompt: What is 2+2?"
echo ""

START=$(date +%s%3N 2>/dev/null || date +%s)

RESULT=$(http_post "/completion" '{"prompt":"What is 2+2? Answer:","n_predict":20,"temperature":0.7}')

END=$(date +%s%3N 2>/dev/null || date +%s)

echo "Response:"
echo "$RESULT" | grep -o '"content":"[^"]*"' | sed 's/"content":"//;s/"$//'
echo ""

# Try to calculate time
if [ "$END" != "$START" ]; then
    ELAPSED=$((END - START))
    echo "Time: ${ELAPSED}ms"
fi
