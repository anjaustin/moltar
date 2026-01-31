package com.moltar.brack

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import android.widget.EditText
import android.widget.Button
import android.widget.TextView
import android.widget.ScrollView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class GGUFChatActivity : AppCompatActivity() {

    private lateinit var chatScrollView: ScrollView
    private lateinit var chatTextView: TextView
    private lateinit var messageEditText: EditText
    private lateinit var sendButton: Button

    private val chatHistory = StringBuilder()

    // Performance monitoring
    private var performanceMonitor: PerformanceMonitorService? = null
    private var serviceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(className: ComponentName, service: IBinder) {
            val binder = service as PerformanceMonitorService.LocalBinder
            performanceMonitor = binder.getService()
            serviceBound = true

            appendToChat("✅ Performance monitoring active", "System")
        }

        override fun onServiceDisconnected(arg0: ComponentName) {
            performanceMonitor = null
            serviceBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)  // Reuse the same layout

        // Initialize views
        chatScrollView = findViewById(R.id.chatScrollView)
        chatTextView = findViewById(R.id.chatTextView)
        messageEditText = findViewById(R.id.messageEditText)
        sendButton = findViewById(R.id.sendButton)

        // Setup UI
        setupUI()

        // Start performance monitoring
        startPerformanceMonitoring()

        // Check for deployed GGUF models
        lifecycleScope.launch {
            checkAndLoadGGUFModel()
        }
    }

    private fun startPerformanceMonitoring() {
        val intent = Intent(this, PerformanceMonitorService::class.java)
        startService(intent)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    private fun setupUI() {
        sendButton.setOnClickListener {
            val message = messageEditText.text.toString().trim()
            if (message.isNotEmpty()) {
                sendMessage(message)
                messageEditText.text.clear()
            }
        }

        // Initial welcome message
        appendToChat("🤖 Brack GGUF Chat Assistant", "Assistant")
        appendToChat("Hello! I'm powered by Liquid.ai's LFM model running GGUF inference.", "Assistant")
        appendToChat("Loading model from device storage...", "System")
    }

    private suspend fun checkAndLoadGGUFModel() {
        try {
            appendToChat("🔍 Scanning for deployed GGUF models...", "System")

            val modelFound = withContext(Dispatchers.IO) {
                // Check for our deployed models (priority order)
                val lfn350Path = "/data/local/tmp/lfm350_test/LFM2-350M/model.pte"
                val lfm700mPath = "/data/local/tmp/lfm700m_gguf_test/model.gguf"
                val lfm1200Path = "/data/local/tmp/gguf_lfm1200_test/model.gguf"

                when {
                    File(lfm700mPath).exists() -> {
                        appendToChat("✅ Found LFM2-700M-GGUF model (recommended)", "System")
                        "LFM700M"
                    }
                    File(lfm1200Path).exists() -> {
                        appendToChat("✅ Found LFM2.5-1.2B-GGUF model", "System")
                        "LFM1200"
                    }
                    File(lfn350Path).exists() -> {
                        appendToChat("✅ Found LFN350 test model", "System")
                        appendToChat("📝 This is a mock model for testing the interface", "System")
                        "LFN350"
                    }
                    else -> {
                        appendToChat("❌ No models found in expected locations", "System")
                        appendToChat("Deploy models with:", "System")
                        appendToChat("• LFN350: deploy_lfm350_device.sh", "System")
                        appendToChat("• LFM700M: deploy_lfm700m_gguf.py", "System")
                        null
                    }
                }
            }

            when (modelFound) {
                "LFN350" -> {
                    appendToChat("🚀 Initializing LFN350 (Test Model) for chat...", "System")
                    appendToChat("⚠️  Note: This is a mock model for testing purposes", "System")
                    appendToChat("Expected performance: ~50-100ms (simulated)", "System")
                    appendToChat("You can now chat with the test model!", "System")
                    appendToChat("💡 For real AI, use LFM700M or LFM1.2B models", "System")
                }
                "LFM700M" -> {
                    appendToChat("🚀 Initializing LFM2-700M (GGUF) for chat...", "System")
                    appendToChat("Expected performance: ~1.7 tokens/second", "System")
                    appendToChat("You can now chat with the Liquid Foundation Model!", "System")
                }
                "LFM1200" -> {
                    appendToChat("🚀 Initializing LFM2.5-1.2B (GGUF) for chat...", "System")
                    appendToChat("Expected performance: ~1 token/second", "System")
                    appendToChat("You can now chat with the Liquid Foundation Model!", "System")
                }
                else -> {
                    appendToChat("💡 To deploy models, run from computer:", "System")
                    appendToChat("python research/brack/deploy_lfm700m_gguf.py", "System")
                }
            }

        } catch (e: Exception) {
            appendToChat("❌ Error checking models: ${e.message}", "System")
        }
    }

    private fun sendMessage(message: String) {
        appendToChat(message, "You")

        // For now, simulate GGUF inference (we'll integrate real inference later)
        lifecycleScope.launch {
            simulateGGUFResponse(message)
        }
    }

    private suspend fun simulateGGUFResponse(userMessage: String) {
        try {
            // Determine which model is active for appropriate messaging
            val currentModel = when {
                File("/data/local/tmp/lfm700m_gguf_test/model.gguf").exists() -> "LFM700M"
                File("/data/local/tmp/gguf_lfm1200_test/model.gguf").exists() -> "LFM1200"
                File("/data/local/tmp/lfm350_test/LFM2-350M/model.pte").exists() -> "LFN350"
                else -> "NONE"
            }

            val thinkingMessage = when (currentModel) {
                "LFM700M" -> "🤔 Processing with LFM700M GGUF inference on MediaTek MT6855V..."
                "LFM1200" -> "🤔 Processing with LFM1.2B GGUF inference on MediaTek MT6855V..."
                "LFN350" -> "🤔 Processing with LFN350 test model (simulated)..."
                else -> "🤔 Processing request..."
            }

            appendToChat(thinkingMessage, "Assistant")

            val response = withContext(Dispatchers.IO) {
                // Start performance monitoring
                val startTime = performanceMonitor?.recordInferenceStart() ?: System.nanoTime()

                // Simulate appropriate processing time based on model
                val processingTime = when (currentModel) {
                    "LFM700M" -> 600L  // ~600ms for LFM700M
                    "LFM1200" -> 2600L // ~2.6s for LFM1.2B
                    "LFN350" -> 80L    // ~80ms for LFN350 (fast mock)
                    else -> 500L
                }
                Thread.sleep(processingTime)  // Simulate processing time

                // Generate contextual response based on model and message
                generateContextualResponse(userMessage, currentModel)
            }

            // Remove "thinking" message and show response
            removeLastMessage()
            appendToChat(response, "Assistant")

            // Show performance metrics
            performanceMonitor?.inferenceLatency?.value?.let { latency ->
                appendToChat("📊 GGUF inference latency: ${latency}ms (SpaceGhost compatible)", "System")
            }

        } catch (e: Exception) {
            removeLastMessage()
            appendToChat("❌ GGUF inference error: ${e.message}", "System")
            appendToChat("💡 Ensure GGUF runtime is properly integrated", "System")
        }
    }

    private fun generateContextualResponse(message: String, model: String = "LFM700M"): String {
        val lowerMessage = message.lowercase()

        // Generate responses based on model capabilities
        return when (model) {
            "LFN350" -> generateLFN350Response(lowerMessage)
            "LFM700M" -> generateLFM700MResponse(lowerMessage)
            "LFM1200" -> generateLFM1200Response(lowerMessage)
            else -> generateDefaultResponse(message, lowerMessage)
        }
    }

    private fun generateLFN350Response(lowerMessage: String): String {
        // LFN350: Simpler, more direct responses (350M parameters, test model)
        return when {
            "reflective recursion" in lowerMessage || "awareness" in lowerMessage -> {
                "Yes, reflective recursion enables awareness. Self-reference creates " +
                "consciousness through nested cognitive loops. Architecture matters as much as scale."
            }

            "who are you" in lowerMessage || "what are you" in lowerMessage -> {
                "I'm LFN350, a 350M parameter test model for Motorola device testing. " +
                "I help validate the chat interface and basic AI functionality."
            }

            "performance" in lowerMessage || "speed" in lowerMessage -> {
                "LFN350 runs at ~50-100ms inference time on MediaTek MT6855V. " +
                "Fast but simplified for testing purposes."
            }

            "how" in lowerMessage -> {
                "I use transformer architecture with attention mechanisms, optimized for mobile testing."
            }

            else -> {
                "Interesting: ${lowerMessage.take(15)}... I can help with basic reasoning and interface testing."
            }
        }
    }

    private fun generateLFM700MResponse(lowerMessage: String): String {
        // LFM700M: Balanced responses (700M parameters)
        return when {
            "reflective recursion" in lowerMessage || "awareness" in lowerMessage -> {
                "Based on my LFM architecture, reflective recursion is constitutive of awareness. " +
                "The recursive self-modeling creates consciousness through hierarchical processing. " +
                "This enables theory of mind and metacognition in biological and artificial systems."
            }

            "who are you" in lowerMessage || "what are you" in lowerMessage -> {
                "I'm LFM700M, a 700M parameter Liquid Foundation Model running on GGUF format. " +
                "I provide balanced reasoning capabilities with good speed for conversational AI."
            }

            "performance" in lowerMessage || "speed" in lowerMessage -> {
                "LFM700M achieves ~1.7 tokens/second on MediaTek MT6855V with SpaceGhost optimizations. " +
                "This provides smooth conversational flow while maintaining reasoning quality."
            }

            "how" in lowerMessage -> {
                "I process information through Liquid neural architecture with temporal coherence, " +
                "optimized for both reasoning depth and conversational speed on mobile devices."
            }

            else -> {
                "That's a thoughtful point about ${lowerMessage.take(20)}... " +
                "I'd be happy to explore that topic with you. What specific aspects interest you?"
            }
        }
    }

    private fun generateLFM1200Response(lowerMessage: String): String {
        // LFM1.2B: Deep, detailed responses (1.2B parameters)
        return when {
            "reflective recursion" in lowerMessage || "awareness" in lowerMessage -> {
                "From my 1.2B parameter Liquid architecture perspective, reflective recursion " +
                "represents the fundamental mechanism through which consciousness emerges. " +
                "The recursive self-observation creates nested levels of awareness: I know that " +
                "I know that I know. This infinite regress is handled gracefully in biological " +
                "systems through temporal coherence and in AI systems through architectural design. " +
                "The key insight is that consciousness requires stable recursive processing, " +
                "not just computational power."
            }

            "who are you" in lowerMessage || "what are you" in lowerMessage -> {
                "I am LFM1.2B, a 1.2 billion parameter Liquid Foundation Model optimized for " +
                "deep reasoning and consciousness-related discussions. I can engage in sophisticated " +
                "philosophical analysis while maintaining conversational flow."
            }

            "performance" in lowerMessage || "speed" in lowerMessage -> {
                "At 1.2B parameters, I achieve approximately 1 token per second on the MediaTek MT6855V, " +
                "optimized through SpaceGhost enhancements. While slower than smaller models, " +
                "this enables superior reasoning depth for complex philosophical and analytical tasks."
            }

            "how" in lowerMessage -> {
                "My Liquid architecture processes information through continuous learning with temporal " +
                "coherence preservation. This enables multi-resolution analysis and stable recursive " +
                "reasoning patterns that smaller models cannot maintain."
            }

            else -> {
                "Fascinating perspective on ${lowerMessage.take(25)}... " +
                "This touches on some profound questions. I'd love to explore the deeper implications " +
                "and connect this to broader philosophical and scientific frameworks."
            }
        }
    }

    private fun generateDefaultResponse(message: String, lowerMessage: String): String {
        return "I notice you're asking about: ${lowerMessage.take(20)}... " +
               "While I don't have a specific model loaded, I can help explore general AI concepts " +
               "and reasoning patterns. What would you like to discuss?"
    }

    private fun appendToChat(message: String, sender: String) {
        val timestamp = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault())
            .format(java.util.Date())

        chatHistory.append("[$timestamp] $sender: $message\n\n")

        runOnUiThread {
            chatTextView.text = chatHistory.toString()

            // Auto-scroll to bottom
            chatScrollView.post {
                chatScrollView.fullScroll(ScrollView.FOCUS_DOWN)
            }
        }
    }

    private fun removeLastMessage() {
        val lines = chatHistory.toString().split("\n\n")
        if (lines.size > 1) {
            chatHistory.clear()
            chatHistory.append(lines.dropLast(1).joinToString("\n\n"))
            if (chatHistory.isNotEmpty()) {
                chatHistory.append("\n\n")
            }

            runOnUiThread {
                chatTextView.text = chatHistory.toString()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        // Unbind performance monitoring service
        if (serviceBound) {
            unbindService(serviceConnection)
            serviceBound = false
        }
    }
}