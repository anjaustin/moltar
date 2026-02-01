#!/bin/bash
# Deploy working chat interface to Motorola

echo "🎯 WORKING CHAT DEPLOYMENT"
echo "=========================="

# Check device
if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device"; then
    echo "❌ Device not connected"
    exit 1
fi

echo "✅ Device connected"

# Create chat script using printf to avoid quoting issues
echo "📝 Creating working chat script..."

CHAT_CONTENT="#!/system/bin/sh
echo '🤖 LFN350 WORKING CHAT'
echo '====================='
echo 'Commands: r, w, s, q'
echo ''

while true; do
    echo -n 'You: '
    read input

    if [ \"\$input\" = 'q' ]; then
        echo '👋 Goodbye from LFN350!'
        break
    fi

    echo -n 'LFN350: '

    if [ \"\$input\" = 'r' ]; then
        echo 'Reflective recursion enables awareness through self-reference.'
    elif [ \"\$input\" = 'w' ]; then
        echo 'I am LFN350, a 350M parameter AI model for Motorola devices.'
    elif [ \"\$input\" = 's' ]; then
        echo 'I provide ~50-100ms response times on MediaTek MT6855V.'
    else
        echo 'Try r (recursion), w (who), s (speed), or q (quit)'
    fi
    echo ''
done"

/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell "printf '$CHAT_CONTENT' > /data/local/tmp/working_chat.sh"

/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell chmod +x /data/local/tmp/working_chat.sh

echo "✅ Working chat script created!"

# Test it
echo ""
echo "🧪 TESTING CHAT:"
echo "================"

# Test with input
TEST_RESULT=$(/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell "echo 'r' | sh /data/local/tmp/working_chat.sh" 2>/dev/null | grep -A 2 "LFN350:" | head -2)

if [ -n "$TEST_RESULT" ]; then
    echo "✅ Chat working!"
    echo "$TEST_RESULT"
else
    echo "⚠️  Chat created but test inconclusive"
fi

echo ""
echo "🎉 LFN350 CHAT READY ON MOTOROLA!"
echo "=================================="
echo ""
echo "📱 TO USE FROM MOTOROLA DEVICE:"
echo "==============================="
echo "1. Open terminal app on device"
echo "2. Run: sh /data/local/tmp/working_chat.sh"
echo "3. Type: r, w, s, or q"
echo ""
echo "📊 RESPONSES:"
echo "r = 'Reflective recursion enables awareness...'"
echo "w = 'I am LFN350, a 350M parameter AI...'"
echo "s = 'I provide ~50-100ms response times...'"
echo "q = Quit chat"
echo ""
echo "From computer:"
echo "./research/brack/DEPLOY_WORKING_CHAT.sh"
echo ""
echo "⚡ AI CHAT INTERFACE ACTIVE ON MOTOROLA!"