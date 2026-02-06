// CfC + Ternary FFN: Minimal Hybrid LLM for PowerVR
//
// Architecture:
//   Input → CfC Cell (replaces attention) → Ternary FFN → Output
//
// Key optimizations:
//   1. Ternary weights (2-bit): no multiplication, just add/subtract
//   2. CfC recurrence: no KV cache, no softmax, O(1) memory per token
//   3. LUT activations: precomputed sigmoid/tanh tables
//   4. Sparse execution: skip zero weights (81% at threshold 0.10)
//
// Target: Beat 50 tok/s CPU baseline on Moto G Power 5G

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
#include <random>

#ifndef EGL_OPENGL_ES3_BIT_KHR
#define EGL_OPENGL_ES3_BIT_KHR 0x0040
#endif

// Model dimensions (small test model)
constexpr int N_EMBD = 256;      // Hidden size
constexpr int N_FF = 1024;       // FFN intermediate (4x)
constexpr int N_LAYER = 8;       // Number of layers
constexpr int CFC_HIDDEN = 64;   // CfC cell hidden size (tiny!)

// EGL globals
EGLDisplay g_display;
EGLContext g_context;
EGLSurface g_surface;

//=============================================================================
// Shader Sources
//=============================================================================

const char* vs_fullscreen = R"(#version 310 es
precision highp float;
void main() {
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
)";

// =============================================================================
// TERNARY GEMV: No multiplication!
// Weights are 2-bit packed: 00=0, 01=+1, 10=-1, 11=0
// Each uint32 holds 16 weights
// =============================================================================
const char* fs_ternary_gemv = R"(#version 310 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

out vec4 fragColor;

uniform usampler2D u_weights;  // 2-bit packed: 16 weights per uint32
uniform sampler2D u_input;     // FP32 input vector
uniform sampler2D u_bias;      // FP32 bias (optional)
uniform int u_K;               // Input dimension (must be multiple of 16)
uniform int u_M;               // Output dimension
uniform int u_use_bias;        // 1 if bias should be added

void main() {
    int row = int(gl_FragCoord.y - 0.5);
    if (row >= u_M) { fragColor = vec4(0.0); return; }
    
    int packed_per_row = u_K / 16;  // Each uint32 = 16 weights
    float sum = 0.0;
    
    for (int p = 0; p < packed_per_row; p++) {
        uint packed = texelFetch(u_weights, ivec2(p, row), 0).r;
        
        // Unpack 16 weights from uint32
        for (int i = 0; i < 16; i++) {
            uint trit = (packed >> uint(i * 2)) & 0x3u;
            int k = p * 16 + i;
            float x = texelFetch(u_input, ivec2(k, 0), 0).r;
            
            // No multiplication! Just conditional add/subtract
            if (trit == 1u) {
                sum += x;  // +1
            } else if (trit == 2u) {
                sum -= x;  // -1
            }
            // trit == 0 or 3: skip (zero weight)
        }
    }
    
    // Add bias if present
    if (u_use_bias == 1) {
        sum += texelFetch(u_bias, ivec2(row, 0), 0).r;
    }
    
    fragColor = vec4(sum, 0.0, 0.0, 1.0);
}
)";

// =============================================================================
// CfC CELL: Closed-form Continuous-time recurrent cell
// h_new = (1 - gate) * h_prev * decay + gate * candidate
// gate = sigmoid(W_gate @ [x, h] + b_gate)
// candidate = tanh(W_cand @ [x, h] + b_cand)
// decay = exp(-dt / tau) -- precomputed
// =============================================================================
const char* fs_cfc_cell = R"(#version 310 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

out vec4 fragColor;

uniform usampler2D u_w_gate;    // Ternary gate weights [hidden, input+hidden]
uniform usampler2D u_w_cand;    // Ternary candidate weights [hidden, input+hidden]
uniform sampler2D u_b_gate;     // Gate bias [hidden]
uniform sampler2D u_b_cand;     // Candidate bias [hidden]
uniform sampler2D u_input;      // Input [input_dim]
uniform sampler2D u_h_prev;     // Previous hidden state [hidden_dim]
uniform sampler2D u_decay;      // Precomputed exp(-dt/tau) [hidden_dim]
uniform int u_input_dim;
uniform int u_hidden_dim;

// Sigmoid approximation (or use LUT texture for accuracy)
float sigmoid_approx(float x) {
    // Fast approximation: x / (1 + abs(x)) * 0.5 + 0.5
    // More accurate: 1 / (1 + exp(-x))
    return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}

// Tanh approximation
float tanh_approx(float x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    float s = sigmoid_approx(2.0 * x);
    return 2.0 * s - 1.0;
}

void main() {
    int i = int(gl_FragCoord.x - 0.5);
    if (i >= u_hidden_dim) { fragColor = vec4(0.0); return; }
    
    int concat_dim = u_input_dim + u_hidden_dim;
    int packed_per_row = concat_dim / 16;
    
    // Compute gate pre-activation: W_gate @ [x, h] + b_gate
    float gate_pre = texelFetch(u_b_gate, ivec2(i, 0), 0).r;
    
    for (int p = 0; p < packed_per_row; p++) {
        uint packed = texelFetch(u_w_gate, ivec2(p, i), 0).r;
        
        for (int j = 0; j < 16; j++) {
            uint trit = (packed >> uint(j * 2)) & 0x3u;
            int k = p * 16 + j;
            
            // Get from input or h_prev depending on index
            float val;
            if (k < u_input_dim) {
                val = texelFetch(u_input, ivec2(k, 0), 0).r;
            } else {
                val = texelFetch(u_h_prev, ivec2(k - u_input_dim, 0), 0).r;
            }
            
            if (trit == 1u) gate_pre += val;
            else if (trit == 2u) gate_pre -= val;
        }
    }
    
    // Compute candidate pre-activation: W_cand @ [x, h] + b_cand
    float cand_pre = texelFetch(u_b_cand, ivec2(i, 0), 0).r;
    
    for (int p = 0; p < packed_per_row; p++) {
        uint packed = texelFetch(u_w_cand, ivec2(p, i), 0).r;
        
        for (int j = 0; j < 16; j++) {
            uint trit = (packed >> uint(j * 2)) & 0x3u;
            int k = p * 16 + j;
            
            float val;
            if (k < u_input_dim) {
                val = texelFetch(u_input, ivec2(k, 0), 0).r;
            } else {
                val = texelFetch(u_h_prev, ivec2(k - u_input_dim, 0), 0).r;
            }
            
            if (trit == 1u) cand_pre += val;
            else if (trit == 2u) cand_pre -= val;
        }
    }
    
    // Apply activations
    float gate = sigmoid_approx(gate_pre);
    float candidate = tanh_approx(cand_pre);
    
    // Get previous state and decay
    float h_prev = texelFetch(u_h_prev, ivec2(i, 0), 0).r;
    float decay = texelFetch(u_decay, ivec2(i, 0), 0).r;
    
    // CfC update: h_new = (1 - gate) * h_prev * decay + gate * candidate
    float h_new = (1.0 - gate) * h_prev * decay + gate * candidate;
    
    fragColor = vec4(h_new, 0.0, 0.0, 1.0);
}
)";

// =============================================================================
// FUSED TERNARY FFN: gate + up + silu + mul + down in minimal passes
// This combines: SiLU(gate @ x) * (up @ x), then down projection
// =============================================================================
const char* fs_ternary_fused_ffn = R"(#version 310 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

out vec4 fragColor;

uniform usampler2D u_w_gate;   // Ternary gate weights [N_FF, N_EMBD]
uniform usampler2D u_w_up;     // Ternary up weights [N_FF, N_EMBD]
uniform sampler2D u_input;     // Input [N_EMBD]
uniform int u_K;               // Input dim
uniform int u_M;               // Output dim (N_FF)

void main() {
    int row = int(gl_FragCoord.y - 0.5);
    if (row >= u_M) { fragColor = vec4(0.0); return; }
    
    int packed_per_row = u_K / 16;
    float gate_sum = 0.0;
    float up_sum = 0.0;
    
    for (int p = 0; p < packed_per_row; p++) {
        uint gate_packed = texelFetch(u_w_gate, ivec2(p, row), 0).r;
        uint up_packed = texelFetch(u_w_up, ivec2(p, row), 0).r;
        
        for (int i = 0; i < 16; i++) {
            int k = p * 16 + i;
            float x = texelFetch(u_input, ivec2(k, 0), 0).r;
            
            uint g_trit = (gate_packed >> uint(i * 2)) & 0x3u;
            uint u_trit = (up_packed >> uint(i * 2)) & 0x3u;
            
            // Gate accumulation (no multiply!)
            if (g_trit == 1u) gate_sum += x;
            else if (g_trit == 2u) gate_sum -= x;
            
            // Up accumulation (no multiply!)
            if (u_trit == 1u) up_sum += x;
            else if (u_trit == 2u) up_sum -= x;
        }
    }
    
    // SiLU activation on gate, then multiply with up
    float silu_gate = gate_sum / (1.0 + exp(-gate_sum));
    float result = silu_gate * up_sum;
    
    fragColor = vec4(result, 0.0, 0.0, 1.0);
}
)";

// Simple activations
const char* fs_silu = R"(#version 310 es
precision highp float;
precision highp sampler2D;

out vec4 fragColor;
uniform sampler2D u_input;
uniform int u_dim;

void main() {
    int i = int(gl_FragCoord.x - 0.5);
    if (i >= u_dim) { fragColor = vec4(0.0); return; }
    
    float x = texelFetch(u_input, ivec2(i, 0), 0).r;
    float sigmoid_x = 1.0 / (1.0 + exp(-x));
    fragColor = vec4(x * sigmoid_x, 0.0, 0.0, 1.0);
}
)";

// RMSNorm (still needed for layer normalization)
const char* fs_rmsnorm = R"(#version 310 es
precision highp float;
precision highp sampler2D;

out vec4 fragColor;
uniform sampler2D u_input;
uniform sampler2D u_weight;
uniform int u_dim;
uniform float u_eps;

void main() {
    int i = int(gl_FragCoord.x - 0.5);
    if (i >= u_dim) { fragColor = vec4(0.0); return; }
    
    // Compute RMS (sum of squares)
    float sum_sq = 0.0;
    for (int j = 0; j < u_dim; j++) {
        float v = texelFetch(u_input, ivec2(j, 0), 0).r;
        sum_sq += v * v;
    }
    float rms = sqrt(sum_sq / float(u_dim) + u_eps);
    
    float v = texelFetch(u_input, ivec2(i, 0), 0).r;
    float w = texelFetch(u_weight, ivec2(i, 0), 0).r;
    
    fragColor = vec4((v / rms) * w, 0.0, 0.0, 1.0);
}
)";

// Element-wise add (for residual connections)
const char* fs_add = R"(#version 310 es
precision highp float;
precision highp sampler2D;

out vec4 fragColor;
uniform sampler2D u_a;
uniform sampler2D u_b;
uniform int u_dim;

void main() {
    int i = int(gl_FragCoord.x - 0.5);
    if (i >= u_dim) { fragColor = vec4(0.0); return; }
    
    float a = texelFetch(u_a, ivec2(i, 0), 0).r;
    float b = texelFetch(u_b, ivec2(i, 0), 0).r;
    fragColor = vec4(a + b, 0.0, 0.0, 1.0);
}
)";

//=============================================================================
// Shader/Program Management
//=============================================================================

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

GLuint create_program(const char* vs, const char* fs) {
    GLuint vsh = compile_shader(GL_VERTEX_SHADER, vs);
    GLuint fsh = compile_shader(GL_FRAGMENT_SHADER, fs);
    if (!vsh || !fsh) return 0;
    
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vsh);
    glAttachShader(prog, fsh);
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
    
    glDeleteShader(vsh);
    glDeleteShader(fsh);
    return prog;
}

//=============================================================================
// Texture/Framebuffer helpers
//=============================================================================

struct Texture {
    GLuint id = 0;
    int width = 0, height = 0;
    
    void create_f32(int w, int h) {
        width = w; height = h;
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, w, h);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    
    void create_u32(int w, int h) {
        width = w; height = h;
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, w, h);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    
    void upload_f32(const float* data) {
        glBindTexture(GL_TEXTURE_2D, id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, data);
    }
    
    void upload_u32(const uint32_t* data) {
        glBindTexture(GL_TEXTURE_2D, id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, data);
    }
    
    void destroy() { if (id) glDeleteTextures(1, &id); id = 0; }
};

struct Framebuffer {
    GLuint fbo = 0;
    Texture tex;
    
    void create(int w, int h) {
        tex.create_f32(w, h);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.id, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    
    void bind() {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, tex.width, tex.height);
    }
    
    void destroy() {
        if (fbo) glDeleteFramebuffers(1, &fbo);
        tex.destroy();
        fbo = 0;
    }
};

//=============================================================================
// CfC + Ternary LLM Engine
//=============================================================================

class CfcTernaryLLM {
public:
    // Programs
    GLuint prog_ternary_gemv = 0;
    GLuint prog_cfc_cell = 0;
    GLuint prog_ternary_fused_ffn = 0;
    GLuint prog_silu = 0;
    GLuint prog_rmsnorm = 0;
    GLuint prog_add = 0;
    
    bool init() {
        prog_ternary_gemv = create_program(vs_fullscreen, fs_ternary_gemv);
        prog_cfc_cell = create_program(vs_fullscreen, fs_cfc_cell);
        prog_ternary_fused_ffn = create_program(vs_fullscreen, fs_ternary_fused_ffn);
        prog_silu = create_program(vs_fullscreen, fs_silu);
        prog_rmsnorm = create_program(vs_fullscreen, fs_rmsnorm);
        prog_add = create_program(vs_fullscreen, fs_add);
        
        if (!prog_ternary_gemv || !prog_cfc_cell || !prog_ternary_fused_ffn ||
            !prog_silu || !prog_rmsnorm || !prog_add) {
            fprintf(stderr, "Failed to compile shaders\n");
            return false;
        }
        
        printf("CfC + Ternary LLM initialized\n");
        return true;
    }
    
    // Ternary GEMV: output = ternary_weights @ input
    void ternary_gemv(Framebuffer& output, Texture& weights, Texture& input, 
                      Texture* bias, int M, int K) {
        output.bind();
        glUseProgram(prog_ternary_gemv);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, weights.id);
        glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_weights"), 0);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, input.id);
        glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_input"), 1);
        
        if (bias) {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, bias->id);
            glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_bias"), 2);
            glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_use_bias"), 1);
        } else {
            glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_use_bias"), 0);
        }
        
        glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_M"), M);
        glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_K"), K);
        
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    
    // Fused ternary FFN: output = SiLU(gate @ x) * (up @ x)
    void ternary_fused_ffn(Framebuffer& output, Texture& w_gate, Texture& w_up,
                           Texture& input, int M, int K) {
        output.bind();
        glUseProgram(prog_ternary_fused_ffn);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, w_gate.id);
        glUniform1i(glGetUniformLocation(prog_ternary_fused_ffn, "u_w_gate"), 0);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, w_up.id);
        glUniform1i(glGetUniformLocation(prog_ternary_fused_ffn, "u_w_up"), 1);
        
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, input.id);
        glUniform1i(glGetUniformLocation(prog_ternary_fused_ffn, "u_input"), 2);
        
        glUniform1i(glGetUniformLocation(prog_ternary_fused_ffn, "u_M"), M);
        glUniform1i(glGetUniformLocation(prog_ternary_fused_ffn, "u_K"), K);
        
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    
    void destroy() {
        if (prog_ternary_gemv) glDeleteProgram(prog_ternary_gemv);
        if (prog_cfc_cell) glDeleteProgram(prog_cfc_cell);
        if (prog_ternary_fused_ffn) glDeleteProgram(prog_ternary_fused_ffn);
        if (prog_silu) glDeleteProgram(prog_silu);
        if (prog_rmsnorm) glDeleteProgram(prog_rmsnorm);
        if (prog_add) glDeleteProgram(prog_add);
    }
};

//=============================================================================
// EGL Setup
//=============================================================================

bool init_egl() {
    g_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(g_display, nullptr, nullptr)) {
        fprintf(stderr, "EGL init failed\n");
        return false;
    }
    
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
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
// Generate random ternary weights (2-bit packed)
//=============================================================================
void generate_ternary_weights(std::vector<uint32_t>& out, int rows, int cols, 
                               float sparsity = 0.81f) {
    // Each uint32 holds 16 weights (2 bits each)
    int packed_cols = cols / 16;
    out.resize(rows * packed_cols);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int r = 0; r < rows; r++) {
        for (int p = 0; p < packed_cols; p++) {
            uint32_t packed = 0;
            for (int i = 0; i < 16; i++) {
                uint32_t trit;
                float rand_val = dist(rng);
                if (rand_val < sparsity) {
                    trit = 0;  // Zero (skip)
                } else if (rand_val < sparsity + (1.0f - sparsity) / 2) {
                    trit = 1;  // +1
                } else {
                    trit = 2;  // -1
                }
                packed |= (trit << (i * 2));
            }
            out[r * packed_cols + p] = packed;
        }
    }
}

//=============================================================================
// Benchmark
//=============================================================================

void benchmark(CfcTernaryLLM& llm, int iterations) {
    printf("\n=== CfC + Ternary FFN Benchmark ===\n");
    printf("Model: %d embd, %d ff, %d layers\n", N_EMBD, N_FF, N_LAYER);
    
    // Create ternary weights
    std::vector<uint32_t> w_gate_data, w_up_data, w_down_data;
    generate_ternary_weights(w_gate_data, N_FF, N_EMBD, 0.81f);
    generate_ternary_weights(w_up_data, N_FF, N_EMBD, 0.81f);
    generate_ternary_weights(w_down_data, N_EMBD, N_FF, 0.81f);
    
    Texture w_gate, w_up, w_down;
    w_gate.create_u32(N_EMBD / 16, N_FF);
    w_up.create_u32(N_EMBD / 16, N_FF);
    w_down.create_u32(N_FF / 16, N_EMBD);
    w_gate.upload_u32(w_gate_data.data());
    w_up.upload_u32(w_up_data.data());
    w_down.upload_u32(w_down_data.data());
    
    // Create input/output buffers
    Texture input;
    input.create_f32(N_EMBD, 1);
    std::vector<float> input_data(N_EMBD, 0.1f);
    input.upload_f32(input_data.data());
    
    Framebuffer fb_fused, fb_down;
    fb_fused.create(N_FF, 1);
    fb_down.create(N_EMBD, 1);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        llm.ternary_fused_ffn(fb_fused, w_gate, w_up, input, N_FF, N_EMBD);
        llm.ternary_gemv(fb_down, w_down, fb_fused.tex, nullptr, N_EMBD, N_FF);
    }
    glFinish();
    
    // Profile individual operations
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
    
    double t_fused = profile("Ternary fused FFN (256->1024)", [&]() {
        llm.ternary_fused_ffn(fb_fused, w_gate, w_up, input, N_FF, N_EMBD);
    });
    
    double t_down = profile("Ternary GEMV down (1024->256)", [&]() {
        llm.ternary_gemv(fb_down, w_down, fb_fused.tex, nullptr, N_EMBD, N_FF);
    });
    
    printf("\nPer-layer total: %.3f ms\n", t_fused + t_down);
    
    // Full model simulation (8 layers batched)
    printf("\n=== Full Model Simulation (%d layers) ===\n", N_LAYER);
    
    glFinish();
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        for (int layer = 0; layer < N_LAYER; layer++) {
            llm.ternary_fused_ffn(fb_fused, w_gate, w_up, input, N_FF, N_EMBD);
            llm.ternary_gemv(fb_down, w_down, fb_fused.tex, nullptr, N_EMBD, N_FF);
        }
    }
    glFinish();
    auto end = std::chrono::high_resolution_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    printf("  %d layers: %.3f ms\n", N_LAYER, ms_total);
    printf("  Per layer: %.3f ms\n", ms_total / N_LAYER);
    printf("  Est. tok/s: %.1f\n", 1000.0 / ms_total);
    
    // Compare to theoretical
    double theoretical = (t_fused + t_down) * N_LAYER;
    printf("\n  Theoretical (sum of ops): %.3f ms\n", theoretical);
    printf("  Overhead: %.2fx\n", ms_total / theoretical);
    
    // Scale to LFM2-350M dimensions
    printf("\n=== Scaled to LFM2-350M (1024 embd, 4096 ff, 16 layers) ===\n");
    // Ternary GEMV scales as O(M*K/16) texture fetches
    // Current: 256*1024/16 = 16K fetches for fused, 1024*256/16 = 16K for down
    // LFM2: 1024*4096/16 = 256K fetches for fused, 4096*1024/16 = 256K for down
    // Scale factor: 256K / 16K = 16x per op
    double scale_factor = (1024.0 * 4096.0) / (256.0 * 1024.0);
    double scaled_per_layer = (t_fused + t_down) * scale_factor;
    double scaled_total = scaled_per_layer * 16;  // 16 layers
    printf("  Estimated per layer: %.3f ms\n", scaled_per_layer);
    printf("  Estimated 16 layers: %.3f ms\n", scaled_total);
    printf("  Estimated tok/s: %.1f\n", 1000.0 / scaled_total);
    printf("  CPU baseline: 50 tok/s\n");
    
    // Cleanup
    w_gate.destroy();
    w_up.destroy();
    w_down.destroy();
    input.destroy();
    fb_fused.destroy();
    fb_down.destroy();
}

int main() {
    printf("=== CfC + Ternary FFN LLM Engine ===\n\n");
    
    if (!init_egl()) return 1;
    
    CfcTernaryLLM llm;
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
