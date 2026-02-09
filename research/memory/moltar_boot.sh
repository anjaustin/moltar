#!/data/data/com.termux/files/usr/bin/sh
# Moltar Server — Termux boot script
# Place in ~/.termux/boot/moltar_boot.sh (requires Termux:Boot add-on)
#
# Starts moltar-server on the big cores with performance governors,
# RAG enabled, and session persistence.

LOG=/data/local/tmp/moltar-server.log

# Set big cores (A78) to performance mode
su -c 'echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor' 2>/dev/null
su -c 'echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor' 2>/dev/null

# Kill any existing server
su -c 'pkill -f moltar-server' 2>/dev/null
sleep 1

# Start moltar-server on big cores
export LD_LIBRARY_PATH=/data/local/tmp
su -c "export LD_LIBRARY_PATH=/data/local/tmp && \
  taskset c0 /data/local/tmp/moltar-server \
    /data/local/tmp/LFM2-1.2B-Q4_0.gguf \
    -t 2 -c 2048 -p 8080 \
    --rag --session /data/local/tmp/session.bin \
    --temp 0.7 \
    >> $LOG 2>&1 &"

echo "Moltar server starting on port 8080 (log: $LOG)"
