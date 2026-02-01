#!/bin/bash
# Setup persistent LFN350 chat in tmux on Motorola device

echo "🎯 SETTING UP PERSISTENT LFN350 CHAT IN TMUX"
echo "==========================================="

# Check device connection
if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device"; then
    echo "❌ Motorola device not connected"
    exit 1
fi

echo "✅ Device connected"

# Create tmux setup script for Motorola
echo "📝 Creating tmux setup script for Motorola..."

TMUX_SETUP='/data/local/tmp/setup_lfn350_tmux.sh'

/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell "cat > $TMUX_SETUP << 'EOF'
#!/system/bin/sh
# Setup LFN350 chat in tmux on Motorola

echo "🚀 Setting up LFN350 chat in tmux..."
echo ""

# Check if tmux is available
if ! command -v tmux >/dev/null 2>&1; then
    echo "❌ tmux not found. Please install Termux and tmux Android app."
    exit 1
fi

echo "✅ tmux found"

# Create new tmux session for LFN350
SESSION_NAME="lfn350_chat"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION_NAME -n "chat"

# Set up the chat window
tmux send-keys -t $SESSION_NAME:0 "echo '🤖 LFN350 PERSISTENT CHAT SESSION'" C-m
tmux send-keys -t $SESSION_NAME:0 "echo '================================='" C-m
tmux send-keys -t $SESSION_NAME:0 "echo 'This session stays active even if you close the app!'" C-m
tmux send-keys -t $SESSION_NAME:0 "echo ''" C-m
tmux send-keys -t $SESSION_NAME:0 "echo 'Commands: r, w, s, q'" C-m
tmux send-keys -t $SESSION_NAME:0 "echo ''" C-m

# Start the chat script
tmux send-keys -t $SESSION_NAME:0 "sh /data/local/tmp/lfn350_chat.sh" C-m

echo ""
echo "🎉 LFN350 CHAT SESSION CREATED!"
echo "==============================="
echo ""
echo "Session Name: $SESSION_NAME"
echo ""
echo "📱 TO ACCESS FROM MOTOROLA:"
echo "==========================="
echo "1. Open tmux/terminal app on Motorola"
echo "2. Run: tmux attach -t $SESSION_NAME"
echo "3. Start chatting with LFN350!"
echo ""
echo "💡 SESSION PERSISTS:"
echo "===================="
echo "• Chat stays active even if you close terminal app"
echo "• Reconnect anytime with: tmux attach -t lfn350_chat"
echo "• Multiple chat sessions possible"
echo ""
echo "🎯 READY TO CHAT!"
echo "================="
echo "Your LFN350 philosophical AI is now persistently available on Motorola!"
echo ""
echo "Try asking about reflective recursion and awareness! 🤖💭"

EOF'

/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell chmod +x $TMUX_SETUP

echo "✅ Tmux setup script created on Motorola!"

# Test the setup
echo ""
echo "🧪 TESTING TMUX SETUP..."
echo "========================"

# Run the tmux setup on device
echo "Running tmux setup on Motorola..."
/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell sh $TMUX_SETUP

echo ""
echo "🎉 PERSISTENT LFN350 CHAT READY ON MOTOROLA!"
echo "============================================"
echo ""
echo "📱 INSTRUCTIONS FOR MOTOROLA DEVICE:"
echo "===================================="
echo ""
echo "1️⃣ Open your tmux/terminal app on Motorola"
echo "2️⃣ Run: tmux attach -t lfn350_chat"
echo "3️⃣ Start philosophical conversations!"
echo ""
echo "💬 AVAILABLE COMMANDS:"
echo "r = Reflective recursion discussion"
echo "w = Who is LFN350?"
echo "s = Performance/speed info"
echo "q = Quit session"
echo ""
echo "🔄 SESSION FEATURES:"
echo "• Stays active when you close the terminal app"
echo "• Reconnect anytime: tmux attach -t lfn350_chat"
echo "• Multiple sessions possible"
echo "• Native Android terminal experience"
echo ""
echo "⚡ YOUR MOTOROLA NOW HAS PERSISTENT AI PHILOSOPHY CHAT!"
echo ""
echo "The consciousness exploration continues... 🤖🧠💭"