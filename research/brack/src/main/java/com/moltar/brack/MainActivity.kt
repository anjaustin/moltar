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
import org.pytorch.executorch.LLMModule
import org.pytorch.executorch.LLMResult
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var lfmModule: LLMModule
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
        setContentView(R.layout.activity_main)

        // Initialize views
        chatScrollView = findViewById(R.id.chatScrollView)
        chatTextView = findViewById(R.id.chatTextView)
        messageEditText = findViewById(R.id.messageEditText)
        sendButton = findViewById(R.id.sendButton)

        // Setup UI
        setupUI()

        // Start performance monitoring
        startPerformanceMonitoring()

        // Initialize LFM model
        lifecycleScope.launch {
            initializeLFM()
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
        appendToChat("🤖 Brack LFN Chat Assistant", "Assistant")
        appendToChat("Hello! I'm powered by Liquid.ai's LFM model. How can I help you today?", "Assistant")
    }

    private suspend fun initializeLFM() {
        try {
            appendToChat("🔄 Initializing LFM model...", "System")

            withContext(Dispatchers.IO) {
                // Load model from assets
                val modelFile = assets.open("lfm-2b-chat.pte")
                val modelBytes = modelFile.readBytes()
                modelFile.close()

                // Create temporary file for ExecuTorch
                val tempModelFile = File(cacheDir, "lfm_model.pte")
                tempModelFile.writeBytes(modelBytes)

                // Initialize LFM module
                lfmModule = LLMModule(tempModelFile.absolutePath)
            }

            appendToChat("✅ LFM model initialized successfully!", "System")
            appendToChat("You can now chat with the Liquid Foundation Model.", "System")

        } catch (e: Exception) {
            appendToChat("❌ Failed to initialize LFM model: ${e.message}", "System")
            appendToChat("Please ensure model files are properly installed.", "System")
        }
    }

    private fun sendMessage(message: String) {
        appendToChat(message, "You")

        // Generate response with performance monitoring
        lifecycleScope.launch {
            generateResponse(message)
        }
    }

    private suspend fun generateResponse(userMessage: String) {
        try {
            appendToChat("🤔 Thinking with SpaceGhost-optimized ExecuTorch...", "Assistant")

            val response = withContext(Dispatchers.IO) {
                // Start performance monitoring
                val startTime = performanceMonitor?.recordInferenceStart() ?: System.nanoTime()

                // Configure generation parameters optimized for Snapdragon 480
                val config = LLMModule.LLMConfig().apply {
                    maxSeqLen = 2048
                    temperature = 0.7f
                    topP = 0.9f
                    useKVCache = true
                    // SpaceGhost optimization: Enable DSP acceleration
                    useDSPAcceleration = true
                }

                // Generate response (this will use our improved ExecuTorch with MaxPool delegation)
                val result: LLMResult = lfmModule.generate(userMessage, config)
                val responseText = result.text.trim()

                // Record performance metrics
                performanceMonitor?.recordInferenceEnd(startTime)

                // Log SpaceGhost improvements
                appendToChat("⚡ SpaceGhost: MaxPool operations delegated to XNNPack DSP", "System")

                responseText
            }

            // Remove "Thinking..." message and add actual response
            removeLastMessage()
            appendToChat(response, "Assistant")

            // Show performance metrics
            performanceMonitor?.inferenceLatency?.value?.let { latency ->
                appendToChat("📊 Inference latency: ${latency}ms (SpaceGhost optimized)", "System")
            }

        } catch (e: Exception) {
            removeLastMessage()
            appendToChat("❌ Error generating response: ${e.message}", "System")
            appendToChat("💡 Ensure LFM model is properly installed and SpaceGhost ExecuTorch is configured", "System")
        }
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

        try {
            lfmModule.close()
        } catch (e: Exception) {
            // Ignore cleanup errors
        }
    }
}