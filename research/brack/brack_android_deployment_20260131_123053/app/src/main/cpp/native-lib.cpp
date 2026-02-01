#include <jni.h>
#include <string>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

#define TAG "BrackJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {
    int g_model_fd = -1;
    size_t g_model_size = 0;
    std::string g_model_path;
    bool g_model_loaded = false;

    void close_model() {
        if (g_model_fd >= 0) {
            close(g_model_fd);
            g_model_fd = -1;
        }
        g_model_size = 0;
        g_model_path.clear();
        g_model_loaded = false;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_moltar_brack_GGUFChatActivity_nativeLoadGGUFModel(
        JNIEnv* env,
        jobject /* this */,
        jstring path) {
    const char* pathCStr = env->GetStringUTFChars(path, nullptr);
    std::string p = pathCStr ? pathCStr : "";
    env->ReleaseStringUTFChars(path, pathCStr);

    if (p.empty()) {
        return env->NewStringUTF("ERR: empty path");
    }

    close_model();

    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) {
        LOGE("Failed to open model: %s (errno=%d)", p.c_str(), errno);
        return env->NewStringUTF(("ERR: open failed errno=" + std::to_string(errno)).c_str());
    }

    struct stat st{};
    if (fstat(fd, &st) != 0) {
        int e = errno;
        close(fd);
        LOGE("Failed to stat model: %s (errno=%d)", p.c_str(), e);
        return env->NewStringUTF(("ERR: stat failed errno=" + std::to_string(e)).c_str());
    }

    if (st.st_size < 8) {
        close(fd);
        LOGE("Model file too small: %s (%lld bytes)", p.c_str(), (long long)st.st_size);
        return env->NewStringUTF("ERR: file too small");
    }

    // Validate GGUF magic
    char magic[4] = {0,0,0,0};
    ssize_t n = pread(fd, magic, 4, 0);
    if (n != 4) {
        int e = errno;
        close(fd);
        return env->NewStringUTF(("ERR: read failed errno=" + std::to_string(e)).c_str());
    }
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        close(fd);
        LOGE("Not a GGUF file (magic %.4s): %s", magic, p.c_str());
        return env->NewStringUTF("ERR: not a GGUF file (missing GGUF magic)");
    }

    // Read GGUF version (uint32 little endian at offset 4)
    uint32_t version = 0;
    n = pread(fd, &version, sizeof(version), 4);
    if (n != (ssize_t)sizeof(version)) {
        int e = errno;
        close(fd);
        return env->NewStringUTF(("ERR: version read failed errno=" + std::to_string(e)).c_str());
    }

    g_model_fd = fd;
    g_model_size = (size_t)st.st_size;
    g_model_path = p;
    g_model_loaded = true;

    LOGI("✅ GGUF model opened: %s (%zu bytes), version=%u", g_model_path.c_str(), g_model_size, version);
    std::string ok = "OK: GGUF loaded (" + std::to_string(g_model_size) + " bytes, v" + std::to_string(version) + ")";
    return env->NewStringUTF(ok.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_moltar_brack_GGUFChatActivity_nativeGetModelResponse(
        JNIEnv* env,
        jobject /* this */,
        jstring prompt) {

    // TODO: Connect to actual ExecuTorch Runtime
    // For now, return a string proving JNI is working
    const char* promptCStr = env->GetStringUTFChars(prompt, nullptr);
    LOGI("Received prompt: %s", promptCStr);
    
    std::string response;
    if (!g_model_loaded) {
        response = "Native: No model loaded yet. Place LFM2-700M GGUF in app external files dir and relaunch.";
    } else {
        response = "Native: LFM2-700M GGUF loaded (staging). Inference integration next.";
    }
    
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
