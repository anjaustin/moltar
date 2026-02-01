package com.moltar.brack

import android.app.Service
import android.content.Intent
import android.os.*
import android.util.Log
import androidx.lifecycle.MutableLiveData
import java.io.File

class PerformanceMonitorService : Service() {

    private val TAG = "PerformanceMonitor"

    // Performance metrics
    val inferenceLatency = MutableLiveData<Long>()
    val memoryUsage = MutableLiveData<Long>()
    val batteryLevel = MutableLiveData<Int>()

    private var handler: Handler? = null
    private var monitoringActive = false

    private val binder = LocalBinder()

    inner class LocalBinder : Binder() {
        fun getService(): PerformanceMonitorService = this@PerformanceMonitorService
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "PerformanceMonitorService created")

        handler = Handler(Looper.getMainLooper())
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "PerformanceMonitorService started")
        return START_STICKY
    }

    fun startMonitoring() {
        if (monitoringActive) return

        monitoringActive = true
        Log.d(TAG, "Starting performance monitoring")

        // Start periodic monitoring
        handler?.post(monitoringRunnable)
    }

    fun stopMonitoring() {
        monitoringActive = false
        Log.d(TAG, "Stopping performance monitoring")
    }

    private val monitoringRunnable = object : Runnable {
        override fun run() {
            if (!monitoringActive) return

            try {
                // Monitor memory usage
                val runtime = Runtime.getRuntime()
                val usedMemory = runtime.totalMemory() - runtime.freeMemory()
                memoryUsage.postValue(usedMemory)

                // Monitor battery (simplified)
                // In real implementation, use BatteryManager
                batteryLevel.postValue(85) // Placeholder

                Log.d(TAG, "Memory usage: ${usedMemory / 1024 / 1024}MB")

            } catch (e: Exception) {
                Log.e(TAG, "Error monitoring performance", e)
            }

            // Schedule next monitoring in 1 second
            handler?.postDelayed(this, 1000)
        }
    }

    fun recordInferenceStart(): Long {
        return System.nanoTime()
    }

    fun recordInferenceEnd(startTime: Long) {
        val latencyNs = System.nanoTime() - startTime
        val latencyMs = latencyNs / 1_000_000
        inferenceLatency.postValue(latencyMs)

        Log.d(TAG, "Inference latency: ${latencyMs}ms")

        // Log to file for analysis
        logPerformanceData(latencyMs)
    }

    private fun logPerformanceData(latencyMs: Long) {
        try {
            val logFile = File(getExternalFilesDir(null), "performance_log.txt")
            val timestamp = System.currentTimeMillis()
            val logEntry = "$timestamp,latency=$latencyMs\n"

            logFile.appendText(logEntry)
        } catch (e: Exception) {
            Log.e(TAG, "Error logging performance data", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopMonitoring()
        Log.d(TAG, "PerformanceMonitorService destroyed")
    }
}