// Texture-Based GEMV using OpenGL ES 3.1
// 
// Strategy: Each fragment (pixel) computes one complete output element
// - Weights stored as 2D texture (M rows x K cols)
// - Activations stored as 1D texture (K elements)  
// - Render to framebuffer (M x 1 pixels)
// - Fragment shader loops over K, accumulating the dot product

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES3/gl3ext.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

#ifndef EGL_OPENGL_ES3_BIT_KHR
#define EGL_OPENGL_ES3_BIT_KHR 0x0040
#endif

// Globals
EGLDisplay display;
EGLContext context;
EGLSurface surface;

// Simple vertex shader - fullscreen triangle
const char* vertex_shader_src = R"(#version 310 es
precision highp float;
out vec2 v_texCoord;
void main() {
    // Fullscreen triangle covering clip space
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
    v_texCoord = vec2((x + 1.0) * 0.5, (y + 1.0) * 0.5);
}
)";

// Fragment shader - each fragment computes one output row
// Simple FP32 version first
const char* fragment_shader_fp32_src = R"(#version 310 es
precision highp float;
precision highp sampler2D;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_weights;  // M rows x K cols
uniform sampler2D u_input;    // 1 row x K cols
uniform int u_K;
uniform int u_M;

void main() {
    // Which row are we computing?
    int row = int(gl_FragCoord.y - 0.5);
    
    if (row >= u_M) {
        fragColor = vec4(0.0);
        return;
    }
    
    float sum = 0.0;
    
    // Loop over all K elements
    for (int k = 0; k < u_K; k++) {
        float w = texelFetch(u_weights, ivec2(k, row), 0).r;
        float a = texelFetch(u_input, ivec2(k, 0), 0).r;
        sum += w * a;
    }
    
    fragColor = vec4(sum, 0.0, 0.0, 1.0);
}
)";

// Q4_0 version
const char* fragment_shader_q4_src = R"(#version 310 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

in vec2 v_texCoord;
out vec4 fragColor;

uniform usampler2D u_weights;  // Q4_0 packed data
uniform sampler2D u_input;     // FP32 activations
uniform int u_K;
uniform int u_M;
uniform int u_blocks_per_row;

// Unpack fp16 to float
float unpack_fp16(uint bits) {
    uint sign = (bits & 0x8000u) << 16u;
    uint exp = (bits >> 10u) & 0x1Fu;
    uint mant = bits & 0x3FFu;
    
    if (exp == 0u) {
        return 0.0;  // Simplified: treat denormals as zero
    } else if (exp == 31u) {
        return sign != 0u ? -1e30 : 1e30;  // Infinity
    }
    exp += 127u - 15u;
    return uintBitsToFloat(sign | (exp << 23u) | (mant << 13u));
}

void main() {
    int row = int(gl_FragCoord.y - 0.5);
    
    if (row >= u_M) {
        fragColor = vec4(0.0);
        return;
    }
    
    float sum = 0.0;
    
    // Each Q4_0 block has 32 weights
    // Stored as: scale (uint16), then 16 bytes of packed nibbles
    // In texture: 9 uint16 values per block
    
    for (int b = 0; b < u_blocks_per_row; b++) {
        int base_x = b * 9;
        
        // Fetch scale
        uint d_bits = texelFetch(u_weights, ivec2(base_x, row), 0).r;
        float scale = unpack_fp16(d_bits);
        
        // Process 32 weights (stored in 8 uint16 values)
        for (int i = 0; i < 8; i++) {
            uint packed = texelFetch(u_weights, ivec2(base_x + 1 + i, row), 0).r;
            
            // Each uint16 has 4 nibbles = 4 weights
            float w0 = float(int((packed >>  0u) & 0xFu) - 8) * scale;
            float w1 = float(int((packed >>  4u) & 0xFu) - 8) * scale;
            float w2 = float(int((packed >>  8u) & 0xFu) - 8) * scale;
            float w3 = float(int((packed >> 12u) & 0xFu) - 8) * scale;
            
            int k_base = b * 32 + i * 4;
            float a0 = texelFetch(u_input, ivec2(k_base + 0, 0), 0).r;
            float a1 = texelFetch(u_input, ivec2(k_base + 1, 0), 0).r;
            float a2 = texelFetch(u_input, ivec2(k_base + 2, 0), 0).r;
            float a3 = texelFetch(u_input, ivec2(k_base + 3, 0), 0).r;
            
            sum += w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3;
        }
    }
    
    fragColor = vec4(sum, 0.0, 0.0, 1.0);
}
)";

GLuint compile_shader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint len;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len);
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        fprintf(stderr, "Shader compile error:\n%s\n", log.data());
        return 0;
    }
    return shader;
}

GLuint create_program(const char* vs_src, const char* fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) return 0;
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint len;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len);
        glGetProgramInfoLog(program, len, nullptr, log.data());
        fprintf(stderr, "Program link error:\n%s\n", log.data());
        return 0;
    }
    
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

bool init_egl() {
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        fprintf(stderr, "Failed to get EGL display\n");
        return false;
    }
    
    if (!eglInitialize(display, nullptr, nullptr)) {
        fprintf(stderr, "Failed to initialize EGL\n");
        return false;
    }
    
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
    
    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(display, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
        fprintf(stderr, "Failed to choose EGL config\n");
        return false;
    }
    
    EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE
    };
    
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) {
        fprintf(stderr, "Failed to create EGL context\n");
        return false;
    }
    
    EGLint pbuffer_attribs[] = {
        EGL_WIDTH, 1,
        EGL_HEIGHT, 1,
        EGL_NONE
    };
    
    surface = eglCreatePbufferSurface(display, config, pbuffer_attribs);
    if (surface == EGL_NO_SURFACE) {
        fprintf(stderr, "Failed to create EGL surface\n");
        return false;
    }
    
    if (!eglMakeCurrent(display, surface, surface, context)) {
        fprintf(stderr, "Failed to make EGL context current\n");
        return false;
    }
    
    printf("OpenGL ES Version: %s\n", glGetString(GL_VERSION));
    printf("Renderer: %s\n", glGetString(GL_RENDERER));
    
    return true;
}

void cleanup_egl() {
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroySurface(display, surface);
    eglDestroyContext(display, context);
    eglTerminate(display);
}

// Benchmark FP32 GEMV
void benchmark_fp32(int M, int K, int iterations) {
    printf("\n=== FP32 Texture GEMV (%dx%d) ===\n", M, K);
    
    GLuint program = create_program(vertex_shader_src, fragment_shader_fp32_src);
    if (!program) return;
    
    // Generate random weights (M x K, stored as R32F texture)
    std::vector<float> weights(M * K);
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = ((float)(rand() % 2000) / 2000.0f - 0.5f) * 0.1f;
    }
    
    GLuint weight_tex;
    glGenTextures(1, &weight_tex);
    glBindTexture(GL_TEXTURE_2D, weight_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, K, M);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, K, M, GL_RED, GL_FLOAT, weights.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // Generate random input (K elements, stored as Kx1 R32F texture)
    std::vector<float> input(K);
    for (int i = 0; i < K; i++) {
        input[i] = (float)(rand() % 1000) / 1000.0f;
    }
    
    GLuint input_tex;
    glGenTextures(1, &input_tex);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, K, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, K, 1, GL_RED, GL_FLOAT, input.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // Create output framebuffer (1 x M)
    GLuint output_tex;
    glGenTextures(1, &output_tex);
    glBindTexture(GL_TEXTURE_2D, output_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, 1, M);
    
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, output_tex, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Framebuffer not complete: 0x%x\n", glCheckFramebufferStatus(GL_FRAMEBUFFER));
        return;
    }
    
    // Set up shader
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_weights"), 0);
    glUniform1i(glGetUniformLocation(program, "u_input"), 1);
    glUniform1i(glGetUniformLocation(program, "u_K"), K);
    glUniform1i(glGetUniformLocation(program, "u_M"), M);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, weight_tex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    
    glViewport(0, 0, 1, M);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    
    // Warmup + verify correctness first
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    
    std::vector<float> output(M);
    glReadPixels(0, 0, 1, M, GL_RED, GL_FLOAT, output.data());
    
    // CPU reference
    float max_err = 0;
    for (int m = 0; m < M; m++) {
        float ref = 0;
        for (int k = 0; k < K; k++) {
            ref += weights[m * K + k] * input[k];
        }
        float err = fabs(output[m] - ref);
        max_err = fmax(max_err, err);
        if (m < 3) {
            printf("Row %d: GPU=%.6f CPU=%.6f err=%.2e\n", m, output[m], ref, err);
        }
    }
    printf("Max error: %.2e\n", max_err);
    
    if (max_err > 1e-3) {
        printf("WARNING: High error, results may be incorrect\n");
    }
    
    // Benchmark
    for (int i = 0; i < 10; i++) {
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ms_per_iter = elapsed_ms / iterations;
    double gflops = (2.0 * M * K) / (ms_per_iter * 1e6);
    
    printf("Time per GEMV: %.3f ms\n", ms_per_iter);
    printf("Throughput: %.2f GFLOPS\n", gflops);
    
    // For LFM2-350M: assume ~40 GEMV ops per token (rough estimate)
    double tok_per_s = 1000.0 / (ms_per_iter * 40);
    printf("Est. tok/s (40 layers): %.1f\n", tok_per_s);
    
    // Cleanup
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &output_tex);
    glDeleteTextures(1, &input_tex);
    glDeleteTextures(1, &weight_tex);
    glDeleteProgram(program);
}

// Q4_0 GEMV benchmark
void benchmark_q4(int M, int K, int iterations) {
    printf("\n=== Q4_0 Texture GEMV (%dx%d) ===\n", M, K);
    
    if (K % 32 != 0) {
        fprintf(stderr, "K must be multiple of 32\n");
        return;
    }
    
    int blocks_per_row = K / 32;
    int tex_width = blocks_per_row * 9;  // 9 uint16 per block
    
    GLuint program = create_program(vertex_shader_src, fragment_shader_q4_src);
    if (!program) return;
    
    // Generate random Q4_0 weights
    std::vector<uint16_t> weight_data(M * tex_width);
    std::vector<float> dequant_weights(M * K);  // For verification
    
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < blocks_per_row; b++) {
            int base = m * tex_width + b * 9;
            
            // Random scale (small positive value)
            float scale = 0.01f + (float)(rand() % 100) / 10000.0f;
            
            // Convert to fp16 (simplified)
            uint32_t f32_bits;
            memcpy(&f32_bits, &scale, 4);
            int exp = ((f32_bits >> 23) & 0xFF) - 127 + 15;
            if (exp < 0) exp = 0;
            if (exp > 31) exp = 31;
            uint16_t fp16 = ((f32_bits >> 16) & 0x8000) |
                           ((exp & 0x1F) << 10) |
                           ((f32_bits >> 13) & 0x3FF);
            weight_data[base] = fp16;
            
            // Random packed weights
            for (int i = 0; i < 8; i++) {
                uint16_t packed = (rand() & 0xF) | ((rand() & 0xF) << 4) |
                                 ((rand() & 0xF) << 8) | ((rand() & 0xF) << 12);
                weight_data[base + 1 + i] = packed;
                
                // Dequantize for verification
                for (int j = 0; j < 4; j++) {
                    int val = (packed >> (j * 4)) & 0xF;
                    dequant_weights[m * K + b * 32 + i * 4 + j] = (float)(val - 8) * scale;
                }
            }
        }
    }
    
    // Create weight texture (R16UI)
    GLuint weight_tex;
    glGenTextures(1, &weight_tex);
    glBindTexture(GL_TEXTURE_2D, weight_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R16UI, tex_width, M);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_width, M, GL_RED_INTEGER, GL_UNSIGNED_SHORT, weight_data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // Generate random input
    std::vector<float> input(K);
    for (int i = 0; i < K; i++) {
        input[i] = (float)(rand() % 1000) / 1000.0f;
    }
    
    GLuint input_tex;
    glGenTextures(1, &input_tex);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, K, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, K, 1, GL_RED, GL_FLOAT, input.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // Create output framebuffer
    GLuint output_tex;
    glGenTextures(1, &output_tex);
    glBindTexture(GL_TEXTURE_2D, output_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, 1, M);
    
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, output_tex, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Framebuffer not complete\n");
        return;
    }
    
    // Set up shader
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_weights"), 0);
    glUniform1i(glGetUniformLocation(program, "u_input"), 1);
    glUniform1i(glGetUniformLocation(program, "u_K"), K);
    glUniform1i(glGetUniformLocation(program, "u_M"), M);
    glUniform1i(glGetUniformLocation(program, "u_blocks_per_row"), blocks_per_row);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, weight_tex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    
    glViewport(0, 0, 1, M);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    
    // Verify correctness
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    
    std::vector<float> output(M);
    glReadPixels(0, 0, 1, M, GL_RED, GL_FLOAT, output.data());
    
    float max_err = 0;
    for (int m = 0; m < M; m++) {
        float ref = 0;
        for (int k = 0; k < K; k++) {
            ref += dequant_weights[m * K + k] * input[k];
        }
        float err = fabs(output[m] - ref);
        max_err = fmax(max_err, err);
        if (m < 3) {
            printf("Row %d: GPU=%.6f CPU=%.6f err=%.2e\n", m, output[m], ref, err);
        }
    }
    printf("Max error: %.2e\n", max_err);
    
    // Benchmark
    for (int i = 0; i < 10; i++) {
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ms_per_iter = elapsed_ms / iterations;
    
    printf("Time per GEMV: %.3f ms\n", ms_per_iter);
    printf("Est. tok/s (40 layers): %.1f\n", 1000.0 / (ms_per_iter * 40));
    
    // Cleanup
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &output_tex);
    glDeleteTextures(1, &input_tex);
    glDeleteTextures(1, &weight_tex);
    glDeleteProgram(program);
}

int main() {
    printf("=== Texture-Based GEMV Benchmark ===\n");
    printf("Testing: GPU as shape processor\n\n");
    
    if (!init_egl()) {
        return 1;
    }
    
    // Test typical LLM layer sizes
    benchmark_fp32(512, 512, 100);
    benchmark_fp32(1024, 1024, 100);
    benchmark_fp32(2048, 1024, 50);
    
    benchmark_q4(512, 512, 100);
    benchmark_q4(1024, 1024, 100);
    benchmark_q4(2048, 1024, 50);
    
    cleanup_egl();
    
    printf("\n=== Done ===\n");
    return 0;
}
