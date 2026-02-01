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
import java.util.Locale

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
    private external fun nativeLoadGGUFModel(path: String): String

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

            val modelPath = withContext(Dispatchers.IO) {
                val baseDir = getExternalFilesDir(null)
                val candidates = listOf(
                    File(baseDir, "lfm2-700m.gguf"),
                    File(baseDir, "LFM2-700M.gguf"),
                    File(baseDir, "model.gguf"),
                )
                candidates.firstOrNull { it.exists() }?.absolutePath
            }

            if (modelPath == null) {
                val baseDir = getExternalFilesDir(null)?.absolutePath ?: "(unavailable)"
                appendToChat("❌ No GGUF model found in app storage.", "System")
                appendToChat("Expected one of: lfm2-700m.gguf / LFM2-700M.gguf / model.gguf", "System")
                appendToChat("Model directory: $baseDir", "System")
                appendToChat("💡 Deploy from computer with adb push into that directory.", "System")
                return
            }

            appendToChat("✅ Found GGUF model: $modelPath", "System")
            appendToChat("🚀 Loading LFM2-700M (GGUF) via native loader...", "System")

            val loadResult = withContext(Dispatchers.IO) { nativeLoadGGUFModel(modelPath) }
            if (loadResult.startsWith("OK")) {
                appendToChat("✅ $loadResult", "System")
                appendToChat("You can now chat. (Inference integration is next.)", "System")
            } else {
                appendToChat("❌ Model load failed: $loadResult", "System")
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
                performanceMonitor?.recordInferenceStart()
                
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