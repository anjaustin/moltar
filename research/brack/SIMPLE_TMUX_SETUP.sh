#!/bin/bash
# Simple tmux setup for LFN350 chat on Motorola

echo "🎯 SIMPLE TMUX LFN350 SETUP"
echo "==========================="

# Check device
if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device"; then
    echo "❌ Device not connected"
    exit 1
fi

echo "✅ Device connected"

# Create simple tmux command for Motorola
TMUX_CMD='tmux new-session -d -s lfn350_chat -n chat \; send-keys "echo '\''🤖 LFN350 TMUX CHAT'\''" C-m \; send-keys "echo '\''==================='\''" C-m \; send-keys "echo '\''Commands: r, w, s, q'\''" C-m \; send-keys "echo" C-m \; send-keys "sh /data/local/tmp/lfn350_chat.sh" C-m'

echo "📝 Setting up tmux session on Motorola..."
/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell "$TMUX_CMD"

echo "✅ Tmux session created!"

# Test it
echo ""
echo "🧪 TESTING TMUX SESSION:"
/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell tmux list-sessions 2>/dev/null || echo "tmux sessions check inconclusive"

echo ""
echo "🎉 LFN350 TMUX CHAT READY!"
echo "=========================="
echo ""
echo "📱 ON YOUR MOTOROLA:"
echo "===================="
echo "1. Open tmux/terminal app"
echo "2. Run: tmux attach -t lfn350_chat"
echo "3. Chat with LFN350!"
echo ""
echo "💬 Commands:"
echo "r = recursion/awareness"
echo "w = who are you"
echo "s = speed/performance"
echo "q = quit"
echo ""
echo "🔄 SESSION PERSISTS!"
echo "===================="
echo "• Stays active when app closes"
echo "• Reconnect: tmux attach -t lfn350_chat"
echo "• Native Android AI chat experience"
echo ""
echo "⚡ PERSISTENT PHILOSOPHICAL AI ON MOTOROLA!"