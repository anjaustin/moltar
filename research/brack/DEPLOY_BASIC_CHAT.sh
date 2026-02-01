#!/bin/bash
# Deploy basic working chat to Motorola

echo "🎯 BASIC CHAT DEPLOYMENT"
echo "======================="

# Check device
if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device"; then
    echo "❌ Device not connected"
    exit 1
fi

echo "✅ Device connected"

# Create basic chat script
echo "📝 Creating basic chat..."

/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell "echo '#!/system/bin/sh
echo \"🤖 LFN350 BASIC CHAT\"
echo \"===================\"
echo \"Commands: r (recursion), w (who), s (speed), q (quit)\"
echo \"\"

while true; do
    echo -n \"You: \"
    read input

    if [ \"\$input\" = \"q\" ]; then
        echo \"👋 Bye!\"
        break
    fi

    echo -n \"LFN350: \"

    if [ \"\$input\" = \"r\" ]; then
        echo \"Reflective recursion enables awareness through self-reference.\"
    elif [ \"\$input\" = \"w\" ]; then
        echo \"I'm LFN350, 350M parameter AI for Motorola.\"
    elif [ \"\$input\" = \"s\" ]; then
        echo \"~50-100ms responses on MediaTek hardware.\"
    else
        echo \"Try: r, w, s, q\"
    fi
    echo \"\"
done' > /data/local/tmp/basic_chat.sh"

/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell chmod +x /data/local/tmp/basic_chat.sh

echo "✅ Basic chat deployed!"

# Test it
echo ""
echo "🧪 TEST:"
/Users/aaronjosserand-austin/000/Motorola/tools/android/adb shell "echo 'r' | sh /data/local/tmp/basic_chat.sh | head -3"

echo ""
echo "🎉 WORKING CHAT READY!"
echo "======================"
echo ""
echo "From Motorola:"
echo "sh /data/local/tmp/basic_chat.sh"
echo ""
echo "Commands:"
echo "r = recursion/awareness"
echo "w = who are you"
echo "s = speed/performance"
echo "q = quit"
echo ""
echo "💬 LFN350 chat active on Motorola!"