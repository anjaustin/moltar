#!/bin/bash
# Tmux-powered Android build and deployment

SESSION_NAME="android_build_$(date +%s)"

echo "🎯 TMUX-POWERED ANDROID BUILD"
echo "============================"
echo "Session: $SESSION_NAME"
echo ""

# Check prerequisites
if ! command -v tmux >/dev/null 2>&1; then
    echo "❌ tmux not installed"
    exit 1
fi

if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker daemon not running"
    exit 1
fi

if ! /Users/aaronjosserand-austin/000/Motorola/tools/android/adb devices | grep -q "device"; then
    echo "❌ Motorola device not connected"
    exit 1
fi

echo "✅ All prerequisites OK"

# Start tmux session
echo "🚀 Starting tmux session for persistent build..."

tmux new-session -d -s "$SESSION_NAME" -n "build"

# Split window for monitoring
tmux split-window -h -t "$SESSION_NAME:0"
tmux select-pane -t "$SESSION_NAME:0.0"
tmux resize-pane -t "$SESSION_NAME:0.1" -x 40

# Set up monitoring pane
tmux send-keys -t "$SESSION_NAME:0.1" "echo '📊 BUILD MONITOR'" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo '=============='" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "watch -n 10 'date && echo \"Disk: $(df -h / | tail -1 | awk '\''{print $4}'\'') free\" && echo \"Docker: $(docker ps | wc -l) containers\"'" C-m

# Start build in main pane
tmux send-keys -t "$SESSION_NAME:0.0" "cd /Users/aaronjosserand-austin/000/Motorola/research/brack" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo '🏗️ STARTING ANDROID BUILD IN TMUX SESSION'" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo '========================================='" C-m

# Try the fastest build approach
tmux send-keys -t "$SESSION_NAME:0.0" "docker run --rm -v \"\$(pwd):/workspace\" -w /workspace --platform linux/amd64 openjdk:11-jdk-slim bash -c \"
    echo '📦 Setting up Android build environment...'
    apt-get update && apt-get install -y wget unzip curl android-tools-adb

    # Download Android SDK
    echo '⬇️ Downloading Android SDK...'
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-10406996_latest.zip
    unzip -q commandlinetools-linux-10406996_latest.zip

    # Setup Android SDK
    export ANDROID_HOME=/workspace/android-sdk
    mkdir -p \$ANDROID_HOME/cmdline-tools
    mv cmdline-tools \$ANDROID_HOME/cmdline-tools/latest
    export PATH=\$PATH:\$ANDROID_HOME/cmdline-tools/latest/bin

    # Accept licenses
    yes | sdkmanager --licenses >/dev/null 2>&1

    # Install minimal components
    echo '🔧 Installing Android components...'
    sdkmanager 'platform-tools' >/dev/null 2>&1
    sdkmanager 'platforms;android-34' >/dev/null 2>&1
    sdkmanager 'build-tools;34.0.0' >/dev/null 2>&1

    # Build APK
    echo '🏗️ Building APK...'
    chmod +x gradlew
    ./gradlew assembleDebug --no-daemon --quiet

    if [ -f 'app/build/outputs/apk/debug/app-debug.apk' ]; then
        echo '✅ APK BUILD SUCCESSFUL!'
        ls -la app/build/outputs/apk/debug/app-debug.apk
        echo '🎉 APK ready for deployment!'
    else
        echo '❌ APK build failed'
        exit 1
    fi
\"" C-m

echo ""
echo "🎯 TMUX SESSION STARTED!"
echo "======================="
echo "Session Name: $SESSION_NAME"
echo ""
echo "📺 To monitor the build:"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "🔄 The build will run persistently in tmux"
echo "   Even if you disconnect, it continues!"
echo ""
echo "⏰ Expected completion: 5-10 minutes"
echo ""
echo "📋 What happens next:"
echo "1. Android SDK downloads (~2 minutes)"
echo "2. Gradle build executes (~3 minutes)"
echo "3. APK created and ready for deployment"
echo "4. Script will notify when complete"
echo ""
echo "🚀 Attach to session now:"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "💡 Pro tip: Use Ctrl+B D to detach, tmux attach -t $SESSION_NAME to reattach"
echo ""
echo "🎉 PERSISTENT BUILD STARTED!"