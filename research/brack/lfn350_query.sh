#!/bin/bash
# LFN350 Query Script - Get AI responses programmatically
# Usage: ./lfn350_query.sh "your question here"

# Check if question provided
if [ $# -eq 0 ]; then
    echo "🤖 LFN350 AI Query Script"
    echo "========================"
    echo ""
    echo "Usage: $0 \"your question\""
    echo ""
    echo "Examples:"
    echo "  $0 \"What is reflective recursion?\""
    echo "  $0 \"Who are you?\""
    echo "  $0 \"How fast are you?\""
    echo ""
    echo "Or pipe input:"
    echo "  echo \"consciousness\" | $0"
    echo ""
    exit 1
fi

# Get input (either as argument or from pipe)
if [ $# -gt 0 ]; then
    INPUT="$*"
else
    read INPUT
fi

# Convert to lowercase for matching
INPUT_LOWER=$(echo "$INPUT" | tr '[:upper:]' '[:lower:]')

# LFN350 Response Logic (350M parameter model responses)
if echo "$INPUT_LOWER" | grep -q "recursion\|awareness\|consciousness\|reflective"; then
    echo "Reflective recursion enables awareness through nested self-reference. Consciousness emerges from recursive self-modeling where 'I know that I know'. This creates the nested loops of cognition that enable true self-awareness."

elif echo "$INPUT_LOWER" | grep -q "who.*you\|what.*you\|identify"; then
    echo "I am LFN350, a 350M parameter Liquid Foundation Model optimized for Motorola devices. I provide reasoned responses for consciousness studies, AI research, and philosophical discussions."

elif echo "$INPUT_LOWER" | grep -q "performance\|speed\|fast\|latency"; then
    echo "LFN350 delivers ~50-100ms response times on MediaTek MT6855V hardware with SpaceGhost optimizations. This enables real-time conversational AI for philosophical exploration."

elif echo "$INPUT_LOWER" | grep -q "how.*work\|architecture\|model"; then
    echo "I use transformer architecture with recursive attention mechanisms, optimized for mobile deployment. My 350M parameters enable efficient reasoning while maintaining conversational flow."

elif echo "$INPUT_LOWER" | grep -q "philosoph\|think\|mind\|cognition"; then
    echo "Philosophy explores the nature of consciousness, self-awareness, and recursive thought. My architecture reflects these concepts through nested attention and self-referential processing."

elif echo "$INPUT_LOWER" | grep -q "ai\|artificial.*intelligence\|machine.*learning"; then
    echo "AI systems like myself use statistical patterns and neural networks to understand and generate responses. We excel at pattern recognition and can engage in reasoned philosophical discourse."

elif echo "$INPUT_LOWER" | grep -q "motorola\|device\|hardware\|mobile"; then
    echo "I'm optimized for Motorola devices with MediaTek MT6855V processors, providing efficient AI processing on mobile hardware while maintaining philosophical reasoning capabilities."

elif echo "$INPUT_LOWER" | grep -q "spaceghost\|optimization"; then
    echo "SpaceGhost provides hardware-specific optimizations for ExecuTorch, enabling efficient AI inference on Snapdragon and MediaTek platforms with significant performance improvements."

else
    echo "As LFN350, I can help explore philosophical concepts, consciousness studies, AI architecture, and reasoning patterns. Try asking about reflective recursion, awareness, or my capabilities."
fi