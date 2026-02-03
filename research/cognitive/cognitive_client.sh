#!/system/bin/sh
#
# COGNITIVE CLIENT - Shell-based API for llama-server and Qdrant
# For use by cognitive architecture components
#

LLAMA_HOST="127.0.0.1"
LLAMA_PORT="8080"
QDRANT_HOST="127.0.0.1"
QDRANT_PORT="6333"

# HTTP helper - makes request and returns body
# Usage: http_request METHOD HOST PORT PATH [BODY]
http_request() {
    local method="$1"
    local host="$2"
    local port="$3"
    local path="$4"
    local body="$5"
    local response
    
    if [ -n "$body" ]; then
        local len=$(echo -n "$body" | wc -c)
        response=$((printf "%s %s HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s" "$method" "$path" "$host" "$len" "$body"; sleep 2) | nc $host $port 2>/dev/null)
    else
        response=$((printf "%s %s HTTP/1.0\r\nHost: %s\r\n\r\n" "$method" "$path" "$host"; sleep 1) | nc $host $port 2>/dev/null)
    fi
    
    # Return body (after headers - skip until empty line)
    echo "$response" | awk '/^[[:space:]]*$/{found=1;next}found'
}

#
# LLAMA-SERVER FUNCTIONS
#

# Generate completion from prompt (uses chat endpoint for quality)
# Usage: llm_complete "prompt" [max_tokens] [temperature]
llm_complete() {
    local prompt="$1"
    local max_tokens="${2:-50}"
    local temperature="${3:-0.7}"
    
    # Use chat endpoint for better output quality
    # Wrap prompt as a user message
    llm_chat "" "$prompt" "$max_tokens" "$temperature"
}

# Chat completion (with system prompt) - uses /v1/chat/completions
# Usage: llm_chat "system_prompt" "user_message" [max_tokens] [temperature]
llm_chat() {
    local system="$1"
    local user="$2"
    local max_tokens="${3:-100}"
    local temperature="${4:-0.7}"
    
    # Escape for JSON
    local escaped_user=$(echo "$user" | tr '\n' ' ' | sed 's/"/\\"/g' | tr -d '\r')
    local escaped_system=$(echo "$system" | tr '\n' ' ' | sed 's/"/\\"/g' | tr -d '\r')
    
    # Build messages array
    local body
    if [ -n "$system" ]; then
        body="{\"messages\":[{\"role\":\"system\",\"content\":\"$escaped_system\"},{\"role\":\"user\",\"content\":\"$escaped_user\"}],\"max_tokens\":$max_tokens,\"temperature\":$temperature}"
    else
        body="{\"messages\":[{\"role\":\"user\",\"content\":\"$escaped_user\"}],\"max_tokens\":$max_tokens,\"temperature\":$temperature}"
    fi
    
    local response=$(http_request "POST" "$LLAMA_HOST" "$LLAMA_PORT" "/v1/chat/completions" "$body")
    
    # Extract content from chat response
    # Format: "choices":[{"message":{"content":"..."}}]
    echo "$response" | awk -F'"content":"' '{print $2}' | awk -F'"}' '{print $1}' | sed 's/\\n/\n/g'
}

# Check llama-server health
llm_health() {
    http_request "GET" "$LLAMA_HOST" "$LLAMA_PORT" "/health"
}

#
# QDRANT FUNCTIONS
#

# Create a collection
# Usage: qdrant_create_collection "name" vector_size
qdrant_create_collection() {
    local name="$1"
    local size="${2:-512}"
    
    local body="{\"vectors\":{\"size\":$size,\"distance\":\"Cosine\"},\"on_disk_payload\":true}"
    http_request "PUT" "$QDRANT_HOST" "$QDRANT_PORT" "/collections/$name" "$body"
}

# Add point to collection
# Usage: qdrant_upsert "collection" id "vector_json" "payload_json"
qdrant_upsert() {
    local collection="$1"
    local id="$2"
    local vector="$3"
    local payload="$4"
    
    local body="{\"points\":[{\"id\":$id,\"vector\":$vector,\"payload\":$payload}]}"
    http_request "PUT" "$QDRANT_HOST" "$QDRANT_PORT" "/collections/$collection/points" "$body"
}

# Search collection
# Usage: qdrant_search "collection" "vector_json" [limit]
qdrant_search() {
    local collection="$1"
    local vector="$2"
    local limit="${3:-5}"
    
    local body="{\"vector\":$vector,\"limit\":$limit,\"with_payload\":true}"
    http_request "POST" "$QDRANT_HOST" "$QDRANT_PORT" "/collections/$collection/points/search" "$body"
}

# Get collection info
qdrant_info() {
    local collection="$1"
    http_request "GET" "$QDRANT_HOST" "$QDRANT_PORT" "/collections/$collection"
}

# Check Qdrant health
qdrant_health() {
    http_request "GET" "$QDRANT_HOST" "$QDRANT_PORT" "/"
}

#
# MAIN - Run as command-line tool
#

case "$1" in
    complete)
        shift
        llm_complete "$@"
        ;;
    chat)
        shift
        llm_chat "$@"
        ;;
    health)
        echo "=== llama-server ==="
        llm_health
        echo ""
        echo "=== qdrant ==="
        qdrant_health
        ;;
    qdrant-create)
        shift
        qdrant_create_collection "$@"
        ;;
    qdrant-search)
        shift
        qdrant_search "$@"
        ;;
    qdrant-info)
        shift
        qdrant_info "$@"
        ;;
    test)
        echo "=== Testing llm_complete ==="
        echo "Prompt: What is the capital of France?"
        echo ""
        echo "Response:"
        llm_complete "What is the capital of France?" 30 0.3
        echo ""
        echo ""
        echo "=== Testing llm_chat ==="
        echo "System: You are a helpful assistant. Be brief."
        echo "User: What is 2+2?"
        echo ""
        echo "Response:"
        llm_chat "You are a helpful assistant. Be brief." "What is 2+2?" 20 0.3
        echo ""
        echo ""
        echo "=== Testing with longer response ==="
        echo "User: Explain why the sky is blue in one sentence."
        echo ""
        echo "Response:"
        llm_complete "Explain why the sky is blue in one sentence." 50 0.5
        ;;
    *)
        echo "Usage: $0 {complete|chat|health|qdrant-create|qdrant-search|qdrant-info|test}"
        echo ""
        echo "Commands:"
        echo "  complete \"prompt\" [max_tokens] [temp]  - Generate completion"
        echo "  chat \"system\" \"user\" [max_tokens]     - Chat completion"
        echo "  health                                  - Check all services"
        echo "  qdrant-create \"name\" [vector_size]     - Create collection"
        echo "  qdrant-search \"coll\" \"vector\" [limit]  - Search vectors"
        echo "  qdrant-info \"collection\"               - Get collection info"
        echo "  test                                    - Run quick test"
        ;;
esac
