#include <jni.h>
#include <string>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include <vector>

#define TAG "BrackJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" JNIEXPORT jstring JNICALL
Java_com_moltar_brack_GGUFChatActivity_nativeGetModelResponse(
        JNIEnv* env,
        jobject /* this */,
        jstring prompt) {

    // TODO: Connect to actual ExecuTorch Runtime
    // For now, return a string proving JNI is working
    const char* promptCStr = env->GetStringUTFChars(prompt, nullptr);
    LOGI("Received prompt: %s", promptCStr);
    
    std::string response = "Native: Vulkan Backend Initialized (Staging)";
    
    env->ReleaseStringUTFChars(prompt, promptCStr);
    return env->NewStringUTF(response.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_moltar_brack_GGUFChatActivity_nativeInitVulkan(
        JNIEnv* env,
        jobject /* this */) {
    
    LOGI("Initializing Vulkan Backend...");
    
    // Check Vulkan support
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Brack";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "ExecuTorch";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

    if (result == VK_SUCCESS) {
        LOGI("✅ Vulkan Instance Created Successfully!");
        
        // Enumerate physical devices to find the Mali GPU
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        if (deviceCount > 0) {
            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
            
            for (const auto& device : devices) {
                VkPhysicalDeviceProperties deviceProperties;
                vkGetPhysicalDeviceProperties(device, &deviceProperties);
                LOGI("Found GPU: %s", deviceProperties.deviceName);
            }
        } else {
             LOGE("❌ No Vulkan devices found");
        }

        vkDestroyInstance(instance, nullptr);
        return JNI_TRUE;
    } else {
        LOGE("❌ Failed to create Vulkan instance: %d", result);
        return JNI_FALSE;
    }
}
