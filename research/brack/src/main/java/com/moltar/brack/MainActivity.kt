package com.moltar.brack

import android.os.Bundle
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

        // Initialize LFM model
        lifecycleScope.launch {
            initializeLFM()
        }
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

        // Generate response
        lifecycleScope.launch {
            generateResponse(message)
        }
    }

    private suspend fun generateResponse(userMessage: String) {
        try {
            appendToChat("🤔 Thinking...", "Assistant")

            val response = withContext(Dispatchers.IO) {
                // Configure generation parameters
                val config = LLMModule.LLMConfig().apply {
                    maxSeqLen = 2048
                    temperature = 0.7f
                    topP = 0.9f
                    useKVCache = true
                }

                // Generate response
                val result: LLMResult = lfmModule.generate(userMessage, config)
                result.text.trim()
            }

            // Remove "Thinking..." message and add actual response
            removeLastMessage()
            appendToChat(response, "Assistant")

        } catch (e: Exception) {
            removeLastMessage()
            appendToChat("❌ Error generating response: ${e.message}", "System")
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
        try {
            lfmModule.close()
        } catch (e: Exception) {
            // Ignore cleanup errors
        }
    }
}