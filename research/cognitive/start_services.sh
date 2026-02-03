#!/system/bin/sh
#
# COGNITIVE ARCHITECTURE - Service Startup Script
# Moto G Power 5G (Dimensity 7020)
#
# Services:
#   1. llama-server (A78-1, port 8080) - LLM inference
#   2. qdrant (port 6333) - Vector database
#

LLAMA_DIR="/data/local/tmp/noprofile"
MODEL="/data/local/tmp/LFM2-350M-Q4_0-pure.gguf"
QDRANT_DIR="/data/local/tmp"
LOG_DIR="/data/local/tmp/logs"

# Create log directory
mkdir -p $LOG_DIR

echo "========================================"
echo "  COGNITIVE ARCHITECTURE - Services"
echo "========================================"
echo ""

# Function to check if a service is running
check_service() {
    ps -A 2>/dev/null | grep -q "$1" && echo "RUNNING" || echo "STOPPED"
}

# Function to wait for port
wait_for_port() {
    local port=$1
    local timeout=$2
    local count=0
    while [ $count -lt $timeout ]; do
        nc -z 127.0.0.1 $port 2>/dev/null && return 0
        sleep 1
        count=$((count + 1))
    done
    return 1
}

case "$1" in
    start)
        echo "[1/2] Starting llama-server..."
        
        # Check if already running
        if ps -A | grep -q "llama-server"; then
            echo "  Already running"
        else
            cd $LLAMA_DIR
            # Pin to A78-1 (CPU 7), single thread for consistent performance
            nohup taskset 80 sh -c "LD_LIBRARY_PATH=. ./llama-server \
                -m $MODEL \
                -t 1 \
                --host 0.0.0.0 \
                --port 8080 \
                -c 2048 \
                -np 1 \
                --no-mmap \
                --log-disable" > $LOG_DIR/llama-server.log 2>&1 &
            
            echo "  Waiting for model load..."
            if wait_for_port 8080 30; then
                echo "  Started on port 8080"
            else
                echo "  FAILED - check $LOG_DIR/llama-server.log"
            fi
        fi
        
        echo ""
        echo "[2/2] Starting Qdrant..."
        
        if ps -A | grep -q "qdrant"; then
            echo "  Already running"
        else
            cd $QDRANT_DIR
            mkdir -p qdrant_storage
            nohup ./qdrant --disable-telemetry > $LOG_DIR/qdrant.log 2>&1 &
            
            if wait_for_port 6333 10; then
                echo "  Started on port 6333"
            else
                echo "  FAILED - check $LOG_DIR/qdrant.log"
            fi
        fi
        
        echo ""
        echo "========================================"
        echo "  Services Status"
        echo "========================================"
        $0 status
        ;;
        
    stop)
        echo "Stopping services..."
        pkill -9 llama-server 2>/dev/null && echo "  llama-server stopped" || echo "  llama-server not running"
        pkill -9 qdrant 2>/dev/null && echo "  qdrant stopped" || echo "  qdrant not running"
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        echo ""
        echo "llama-server: $(check_service llama-server)"
        if nc -z 127.0.0.1 8080 2>/dev/null; then
            echo "  Port 8080: OPEN"
            # Quick health check
            HEALTH=$(echo -e "GET /health HTTP/1.0\r\n\r\n" | nc -w 2 127.0.0.1 8080 2>/dev/null | tail -1)
            echo "  Health: $HEALTH"
        else
            echo "  Port 8080: CLOSED"
        fi
        
        echo ""
        echo "qdrant: $(check_service qdrant)"
        if nc -z 127.0.0.1 6333 2>/dev/null; then
            echo "  Port 6333: OPEN"
            VERSION=$(echo -e "GET / HTTP/1.0\r\n\r\n" | nc -w 2 127.0.0.1 6333 2>/dev/null | tail -1)
            echo "  Version: $(echo $VERSION | grep -o '"version":"[^"]*"' | cut -d'"' -f4)"
        else
            echo "  Port 6333: CLOSED"
        fi
        
        echo ""
        echo "Memory usage:"
        ps -A -o pid,rss,comm 2>/dev/null | grep -E "llama-server|qdrant" | while read pid rss name; do
            echo "  $name: $((rss/1024)) MB"
        done
        ;;
        
    logs)
        echo "=== llama-server ===" 
        tail -20 $LOG_DIR/llama-server.log 2>/dev/null
        echo ""
        echo "=== qdrant ==="
        tail -20 $LOG_DIR/qdrant.log 2>/dev/null
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
