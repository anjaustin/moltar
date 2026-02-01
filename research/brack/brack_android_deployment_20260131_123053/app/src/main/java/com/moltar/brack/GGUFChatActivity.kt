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

    // Load native library
    companion object {
        init {
            System.loadLibrary("brack_jni")
        }
    }

    // Native methods
    private external fun nativeGetModelResponse(prompt: String): String
    private external fun nativeInitVulkan(): Boolean

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

        // Initialize Native Backend
        try {
            val vulkanInit = nativeInitVulkan()
            if (vulkanInit) {
                appendToChat("✅ Native Vulkan Backend Initialized", "System")
            } else {
                appendToChat("❌ Native Backend Failed to Initialize", "System")
            }
        } catch (e: Exception) {
            appendToChat("❌ Native Library Load Error: ${e.message}", "System")
        }

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

        // Call Native Inference
        lifecycleScope.launch {
            callNativeInference(message)
        }
    }

    private suspend fun callNativeInference(userMessage: String) {
        try {
            appendToChat("🤔 Processing with Native Vulkan Backend...", "Assistant")

            val response = withContext(Dispatchers.IO) {
                // Record start time
                val startTime = performanceMonitor?.recordInferenceStart() ?: System.nanoTime()
                
                // CALL NATIVE C++ CODE
                val result = nativeGetModelResponse(userMessage)
                
                result
            }

            // Remove "thinking" message and show response
            removeLastMessage()
            appendToChat(response, "Assistant")

            // Show performance metrics
            performanceMonitor?.inferenceLatency?.value?.let { latency ->
                appendToChat("📊 Native inference latency: ${latency}ms", "System")
            }

        } catch (e: Exception) {
            removeLastMessage()
            appendToChat("❌ Native inference error: ${e.message}", "System")
        }
    }

    // REMOVED: simulateGGUFResponse and fake generation logic


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