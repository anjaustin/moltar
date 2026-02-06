// Compute Shader Based LLM Inference
// Uses GLES 3.1 compute shaders with SSBOs
// Avoids FBO switching overhead of fragment shader approach

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

#ifndef EGL_OPENGL_ES3_BIT_KHR
#define EGL_OPENGL_ES3_BIT_KHR 0x0040
#endif

constexpr int N_EMBD = 1024;
constexpr int N_FF = 4096;

EGLDisplay g_display;
EGLContext g_context;
EGLSurface g_surface;

//=============================================================================
// Compute Shaders - ONE THREAD PER OUTPUT ELEMENT (no reduction needed!)
//=============================================================================

// Q4_0 GEMV: each thread computes one output row
const char* cs_gemv_q4 = R"(#version 310 es
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

precision highp float;

layout(std430, binding = 0) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 1) readonly buffer Input { float input_vec[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_vec[]; };

uniform int u_M;  // output dim
uniform int u_K;  // input dim (must be multiple of 32)

// Unpack fp16 from uint
float unpack_fp16(uint bits) {
    uint sign = (bits & 0x8000u) << 16u;
    uint exp = (bits >> 10u) & 0x1Fu;
    uint mant = bits & 0x3FFu;
    if (exp == 0u) return 0.0;
    if (exp == 31u) return sign != 0u ? -1e30 : 1e30;
    exp += 127u - 15u;
    return uintBitsToFloat(sign | (exp << 23u) | (mant << 13u));
}

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= uint(u_M)) return;
    
    int blocks_per_row = u_K / 32;
    float sum = 0.0;
    
    // Each Q4_0 block: 18 bytes = 9 uint16 = 4.5 uint32
    // We'll pack as 5 uint32 per block for simplicity (with padding)
    // Actually, let's use 9 uint16 packed into uint buffer differently
    // Simpler: treat as array of uint16, but read via uint and extract
    
    for (int b = 0; b < blocks_per_row; b++) {
        // Block layout: 1 fp16 scale + 16 bytes (32 nibbles)
        // = 18 bytes = 9 uint16
        // In uint buffer: offset = row * blocks_per_row * 5 + b * 5
        // (Using 5 uint32 = 20 bytes per block, slight waste but aligned)
        
        uint base = uint(row) * uint(blocks_per_row) * 5u + uint(b) * 5u;
        
        // First uint32 contains scale in lower 16 bits
        uint w0 = weights[base];
        float scale = unpack_fp16(w0 & 0xFFFFu);
        
        // Remaining 4 uint32 contain 16 bytes = 32 nibbles
        for (int i = 0; i < 4; i++) {
            uint packed = weights[base + 1u + uint(i)];
            
            // 8 nibbles per uint32
            for (int j = 0; j < 8; j++) {
                int val = int((packed >> (j * 4)) & 0xFu) - 8;
                float w = float(val) * scale;
                int k = b * 32 + i * 8 + j;
                sum += w * input_vec[k];
            }
        }
    }
    
    output_vec[row] = sum;
}
)";

// RMSNorm - two-pass: first compute sum of squares, then normalize
// Pass 1: Each workgroup computes partial sum, then atomic add
const char* cs_rmsnorm_sumsq = R"(#version 310 es
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

precision highp float;

layout(std430, binding = 0) readonly buffer Input { float input_vec[]; };
layout(std430, binding = 1) buffer SumSq { float sum_sq; };

uniform int u_dim;

shared float partial_sums[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint i = gl_GlobalInvocationID.x;
    
    float val = (i < uint(u_dim)) ? input_vec[i] : 0.0;
    partial_sums[tid] = val * val;
    
    barrier();
    
    // Reduction in shared memory
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        barrier();
    }
    
    // Thread 0 adds to global sum
    if (tid == 0u) {
        // Atomic add for float (GLES 3.1 doesn't have native atomicAdd for float)
        // We'll use a workaround or just assume single workgroup for now
        sum_sq = partial_sums[0];
    }
}
)";

// Pass 2: Apply normalization
const char* cs_rmsnorm_apply = R"(#version 310 es
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

precision highp float;

layout(std430, binding = 0) readonly buffer Input { float input_vec[]; };
layout(std430, binding = 1) readonly buffer Weight { float weight[]; };
layout(std430, binding = 2) readonly buffer SumSq { float sum_sq; };
layout(std430, binding = 3) writeonly buffer Output { float output_vec[]; };

uniform int u_dim;
uniform float u_eps;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_dim)) return;
    
    float rms = sqrt(sum_sq / float(u_dim) + u_eps);
    output_vec[i] = (input_vec[i] / rms) * weight[i];
}
)";

// SiLU activation
const char* cs_silu = R"(#version 310 es
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

precision highp float;

layout(std430, binding = 0) readonly buffer Input { float input_vec[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_vec[]; };

uniform int u_dim;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_dim)) return;
    
    float x = input_vec[i];
    output_vec[i] = x / (1.0 + exp(-x));
}
)";

// Element-wise multiply
const char* cs_mul = R"(#version 310 es
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

precision highp float;

layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) readonly buffer B { float b[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_vec[]; };

uniform int u_dim;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_dim)) return;
    
    output_vec[i] = a[i] * b[i];
}
)";

// Element-wise add
const char* cs_add = R"(#version 310 es
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

precision highp float;

layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) readonly buffer B { float b[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_vec[]; };

uniform int u_dim;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_dim)) return;
    
    output_vec[i] = a[i] + b[i];
}
)";

//=============================================================================
// Shader Compilation
//=============================================================================

GLuint compile_compute(const char* source) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint len;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len);
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        fprintf(stderr, "Compute shader error:\n%s\nSource:\n%s\n", log.data(), source);
        return 0;
    }
    
    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader);
    glLinkProgram(prog);
    
    GLint linked;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint len;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len);
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        fprintf(stderr, "Program link error:\n%s\n", log.data());
        return 0;
    }
    
    glDeleteShader(shader);
    return prog;
}

//=============================================================================
// Buffer Management
//=============================================================================

struct Buffer {
    GLuint id = 0;
    size_t size = 0;
    
    void create(size_t bytes) {
        size = bytes;
        glGenBuffers(1, &id);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_COPY);
    }
    
    void upload(const void* data, size_t bytes) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, data);
    }
    
    void download(void* data, size_t bytes) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
        void* ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes, GL_MAP_READ_BIT);
        memcpy(data, ptr, bytes);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    
    void bind(int index) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, id);
    }
    
    void destroy() {
        if (id) glDeleteBuffers(1, &id);
        id = 0;
    }
};

//=============================================================================
// Compute LLM Engine
//=============================================================================

class ComputeLLM {
public:
    GLuint prog_gemv = 0;
    GLuint prog_silu = 0;
    GLuint prog_mul = 0;
    GLuint prog_add = 0;
    
    bool init() {
        prog_gemv = compile_compute(cs_gemv_q4);
        prog_silu = compile_compute(cs_silu);
        prog_mul = compile_compute(cs_mul);
        prog_add = compile_compute(cs_add);
        
        if (!prog_gemv || !prog_silu || !prog_mul || !prog_add) {
            fprintf(stderr, "Failed to compile compute shaders\n");
            return false;
        }
        
        printf("ComputeLLM initialized (GLES 3.1 compute shaders)\n");
        return true;
    }
    
    void gemv_q4(Buffer& weights, Buffer& input, Buffer& output, int M, int K) {
        glUseProgram(prog_gemv);
        weights.bind(0);
        input.bind(1);
        output.bind(2);
        glUniform1i(glGetUniformLocation(prog_gemv, "u_M"), M);
        glUniform1i(glGetUniformLocation(prog_gemv, "u_K"), K);
        glDispatchCompute((M + 63) / 64, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    
    void silu(Buffer& input, Buffer& output, int dim) {
        glUseProgram(prog_silu);
        input.bind(0);
        output.bind(1);
        glUniform1i(glGetUniformLocation(prog_silu, "u_dim"), dim);
        glDispatchCompute((dim + 63) / 64, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    
    void mul(Buffer& a, Buffer& b, Buffer& output, int dim) {
        glUseProgram(prog_mul);
        a.bind(0);
        b.bind(1);
        output.bind(2);
        glUniform1i(glGetUniformLocation(prog_mul, "u_dim"), dim);
        glDispatchCompute((dim + 63) / 64, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    
    void add(Buffer& a, Buffer& b, Buffer& output, int dim) {
        glUseProgram(prog_add);
        a.bind(0);
        b.bind(1);
        output.bind(2);
        glUniform1i(glGetUniformLocation(prog_add, "u_dim"), dim);
        glDispatchCompute((dim + 63) / 64, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    
    void destroy() {
        if (prog_gemv) glDeleteProgram(prog_gemv);
        if (prog_silu) glDeleteProgram(prog_silu);
        if (prog_mul) glDeleteProgram(prog_mul);
        if (prog_add) glDeleteProgram(prog_add);
    }
};

//=============================================================================
// EGL Setup
//=============================================================================

bool init_egl() {
    g_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(g_display, nullptr, nullptr);
    
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
        EGL_NONE
    };
    
    EGLConfig config;
    EGLint num_configs;
    eglChooseConfig(g_display, config_attribs, &config, 1, &num_configs);
    
    EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    g_context = eglCreateContext(g_display, config, EGL_NO_CONTEXT, context_attribs);
    
    EGLint pbuffer_attribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
    g_surface = eglCreatePbufferSurface(g_display, config, pbuffer_attribs);
    
    eglMakeCurrent(g_display, g_surface, g_surface, g_context);
    
    printf("OpenGL ES: %s\n", glGetString(GL_VERSION));
    printf("Renderer: %s\n", glGetString(GL_RENDERER));
    
    return true;
}

void cleanup_egl() {
    eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroySurface(g_display, g_surface);
    eglDestroyContext(g_display, g_context);
    eglTerminate(g_display);
}

//=============================================================================
// Benchmark
//=============================================================================

void benchmark(ComputeLLM& llm, int iterations) {
    printf("\n=== Compute Shader FFN Layer Benchmark ===\n");
    
    // Weight buffers (Q4_0 format: 5 uint32 per block for alignment)
    int blocks_up = N_EMBD / 32;
    int blocks_down = N_FF / 32;
    
    Buffer w_gate, w_up, w_down;
    w_gate.create(N_FF * blocks_up * 5 * sizeof(uint32_t));
    w_up.create(N_FF * blocks_up * 5 * sizeof(uint32_t));
    w_down.create(N_EMBD * blocks_down * 5 * sizeof(uint32_t));
    
    // Fill with random data
    std::vector<uint32_t> data_up(N_FF * blocks_up * 5);
    std::vector<uint32_t> data_down(N_EMBD * blocks_down * 5);
    for (auto& v : data_up) v = rand();
    for (auto& v : data_down) v = rand();
    w_gate.upload(data_up.data(), data_up.size() * sizeof(uint32_t));
    w_up.upload(data_up.data(), data_up.size() * sizeof(uint32_t));
    w_down.upload(data_down.data(), data_down.size() * sizeof(uint32_t));
    
    // Activation buffers
    Buffer buf_input, buf_gate, buf_up, buf_silu, buf_mul, buf_down;
    buf_input.create(N_EMBD * sizeof(float));
    buf_gate.create(N_FF * sizeof(float));
    buf_up.create(N_FF * sizeof(float));
    buf_silu.create(N_FF * sizeof(float));
    buf_mul.create(N_FF * sizeof(float));
    buf_down.create(N_EMBD * sizeof(float));
    
    // Random input
    std::vector<float> input(N_EMBD);
    for (auto& v : input) v = (float)(rand() % 1000) / 1000.0f;
    buf_input.upload(input.data(), input.size() * sizeof(float));
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        llm.gemv_q4(w_gate, buf_input, buf_gate, N_FF, N_EMBD);
        llm.gemv_q4(w_up, buf_input, buf_up, N_FF, N_EMBD);
        llm.silu(buf_gate, buf_silu, N_FF);
        llm.mul(buf_silu, buf_up, buf_mul, N_FF);
        llm.gemv_q4(w_down, buf_mul, buf_down, N_EMBD, N_FF);
    }
    glFinish();
    
    // Profile individual ops
    printf("\nPer-operation timing:\n");
    
    auto profile = [&](const char* name, auto fn) {
        glFinish();
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) fn();
        glFinish();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        printf("  %s: %.3f ms\n", name, ms);
        return ms;
    };
    
    double t_gate = profile("GEMV gate (1024->4096)", [&]() {
        llm.gemv_q4(w_gate, buf_input, buf_gate, N_FF, N_EMBD);
    });
    
    double t_up = profile("GEMV up (1024->4096)", [&]() {
        llm.gemv_q4(w_up, buf_input, buf_up, N_FF, N_EMBD);
    });
    
    double t_silu = profile("SiLU", [&]() {
        llm.silu(buf_gate, buf_silu, N_FF);
    });
    
    double t_mul = profile("Mul", [&]() {
        llm.mul(buf_silu, buf_up, buf_mul, N_FF);
    });
    
    double t_down = profile("GEMV down (4096->1024)", [&]() {
        llm.gemv_q4(w_down, buf_mul, buf_down, N_EMBD, N_FF);
    });
    
    double total = t_gate + t_up + t_silu + t_mul + t_down;
    printf("\nTotal per layer (sum): %.3f ms\n", total);
    
    // Full pipeline
    glFinish();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        llm.gemv_q4(w_gate, buf_input, buf_gate, N_FF, N_EMBD);
        llm.gemv_q4(w_up, buf_input, buf_up, N_FF, N_EMBD);
        llm.silu(buf_gate, buf_silu, N_FF);
        llm.mul(buf_silu, buf_up, buf_mul, N_FF);
        llm.gemv_q4(w_down, buf_mul, buf_down, N_EMBD, N_FF);
    }
    glFinish();
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    double ms_per_layer = elapsed / iterations;
    
    printf("Full pipeline per layer: %.3f ms\n", ms_per_layer);
    printf("Est. tok/s (16 layers, FFN only): %.1f\n", 1000.0 / (ms_per_layer * 16));
    printf("Est. tok/s (full model, ~2x): %.1f\n", 1000.0 / (ms_per_layer * 16 * 2));
    
    // Cleanup
    w_gate.destroy();
    w_up.destroy();
    w_down.destroy();
    buf_input.destroy();
    buf_gate.destroy();
    buf_up.destroy();
    buf_silu.destroy();
    buf_mul.destroy();
    buf_down.destroy();
}

int main() {
    printf("=== GLES 3.1 Compute Shader LLM Benchmark ===\n\n");
    
    if (!init_egl()) return 1;
    
    ComputeLLM llm;
    if (!llm.init()) {
        cleanup_egl();
        return 1;
    }
    
    benchmark(llm, 100);
    
    llm.destroy();
    cleanup_egl();
    
    printf("\n=== Done ===\n");
    return 0;
}
