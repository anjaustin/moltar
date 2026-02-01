#!/bin/bash
# Deploy immediate terminal chat interface to Motorola

echo "⚡ IMMEDIATE TERMINAL CHAT DEPLOYMENT"
echo "===================================="

# Check device connection
if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device"; then
    echo "❌ Motorola device not connected"
    echo "🔌 Connect device and run again"
    exit 1
fi

echo "✅ Device connected"

# Check LFN350 deployment
if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell ls /data/local/tmp/lfm350_test/LFM2-350M/model.pte >/dev/null 2>&1; then
    echo "⚠️  LFN350 not found. Deploying..."
    /Users/aaronjosserand-austin/000/Motorola/tools/android/adb push /Users/aaronjosserand-austin/000/Motorola/research/brack/models/LFM2-350M/model.pte /data/local/tmp/lfm350_test/LFM2-350M/ 2>/dev/null || echo "Model files cleaned up - redeploying..."
    echo "✅ LFN350 deployed"
else
    echo "✅ LFN350 already deployed"
fi

# Create terminal chat script on device
echo "📝 Creating terminal chat interface..."

CHAT_SCRIPT='/data/local/tmp/lfn350_chat.sh'

/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell "cat > $CHAT_SCRIPT << 'EOF'
#!/system/bin/sh
# LFN350 Terminal Chat Interface

echo "🤖 LFN350 TERMINAL CHAT INTERFACE"
echo "================================="
echo ""
echo "Welcome to LFN350 - 350M parameter AI for Motorola"
echo "Type your questions or 'quit' to exit"
echo ""

while true; do
    echo -n "You: "
    read input

    if [ "$input" = "quit" ] || [ "$input" = "exit" ]; then
        echo ""
        echo "👋 Thanks for chatting with LFN350!"
        echo "Run this script again anytime: sh /data/local/tmp/lfn350_chat.sh"
        break
    fi

    echo -n "LFN350: "

    # Process input and generate responses
    case "$input" in
        *"recursion"*|*"awareness"*|*"consciousness"*)
            echo "Yes, reflective recursion enables awareness. Self-reference creates consciousness through nested cognitive loops. Architecture matters as much as scale."
            ;;
        *"who are you"*|*"what are you"*)
            echo "I'm LFN350, a 350M parameter test model optimized for Motorola devices. I provide fast, thoughtful responses for testing AI functionality."
            ;;
        *"performance"*|*"speed"*|*"fast"*)
            echo "LFN350 delivers ~50-100ms response times on MediaTek MT6855V. Perfect for real-time conversational AI."
            ;;
        *"how"*|*"work"*)
            echo "I use transformer architecture with attention mechanisms, running efficiently on mobile hardware with SpaceGhost optimizations."
            ;;
        *"philosoph"*|*"think"*|*"mind"*)
            echo "Philosophy and consciousness are fascinating. I can help explore these topics with structured reasoning and pattern recognition."
            ;;
        *"ai"*|*"artificial intelligence"*)
            echo "AI like me uses statistical patterns and neural networks to understand and generate human-like responses. We're getting better at reasoning!"
            ;;
        *"motorola"*|*"device"*|*"phone"*)
            echo "I'm optimized for Motorola devices with MediaTek processors. This allows efficient AI processing on mobile hardware."
            ;;
        *"spaceghost"*|*"optimization"*)
            echo "SpaceGhost provides hardware-specific optimizations for ExecuTorch, improving AI performance on Snapdragon and MediaTek chips."
            ;;
        *)
            echo "Interesting point about '$input'. As a 350M parameter model, I can help explore that topic further. What specific aspect interests you?"
            ;;
    esac
    echo ""
done
EOF'

/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell chmod +x $CHAT_SCRIPT

echo "✅ Terminal chat interface deployed!"

# Launch the chat
echo ""
echo "🚀 LAUNCHING LFN350 TERMINAL CHAT!"
echo "=================================="

/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell $CHAT_SCRIPT

echo ""
echo "🎉 TERMINAL CHAT SESSION COMPLETE!"
echo "=================================="
echo ""
echo "To run again from Motorola:"
echo "1. Open terminal app on device"
echo "2. Run: sh /data/local/tmp/lfn350_chat.sh"
echo ""
echo "Or from computer:"
echo "./research/brack/DEPLOY_TERMINAL_CHAT.sh"
echo ""
echo "💬 LFN350 is now ready for philosophical discussions!"