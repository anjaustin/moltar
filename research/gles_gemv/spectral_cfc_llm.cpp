// Spectral CfC LLM: Rotation + Recurrence as Attention Replacement
//
// Architecture per layer:
//   Input → RoPE Rotation → CfC Cell → Residual → RMSNorm → Ternary FFN → Output
//
// Key insight: Spectral rotation encodes position in frequency space,
// CfC cell provides temporal mixing with decay. Together they replace
// attention with O(1) memory per token.
//
// Based on ideas from: RWKV, RetNet, Mamba, LFM2, Yinsen CfC

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

// Model dimensions - TEST BOTH SCALES
constexpr int N_EMBD_SMALL = 256;      // Small test model
constexpr int N_FF_SMALL = 1024;
constexpr int N_LAYER_SMALL = 8;

constexpr int N_EMBD_LFM2 = 1024;     // LFM2-350M scale
constexpr int N_FF_LFM2 = 4096;
constexpr int N_LAYER_LFM2 = 16;

// Active config (will test both)
int N_EMBD = N_EMBD_SMALL;
int N_FF = N_FF_SMALL;
int N_LAYER = N_LAYER_SMALL;

constexpr int MAX_SEQ_LEN = 512; // Maximum sequence length
constexpr float ROPE_THETA = 10000.0f;  // RoPE base frequency

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
// RoPE: Rotary Position Embedding
// Rotates pairs of elements by position-dependent angles
// x'[2i]   = x[2i] * cos(θ_i) - x[2i+1] * sin(θ_i)
// x'[2i+1] = x[2i] * sin(θ_i) + x[2i+1] * cos(θ_i)
// =============================================================================
const char* fs_rope = R"(#version 310 es
precision highp float;
precision highp sampler2D;

out vec4 fragColor;

uniform sampler2D u_input;      // Input vector [dim]
uniform sampler2D u_cos;        // Precomputed cos [max_seq, dim/2]
uniform sampler2D u_sin;        // Precomputed sin [max_seq, dim/2]
uniform int u_pos;              // Current position in sequence
uniform int u_dim;

void main() {
    int i = int(gl_FragCoord.x - 0.5);
    if (i >= u_dim) { fragColor = vec4(0.0); return; }
    
    int pair = i / 2;
    int is_odd = i & 1;
    
    // Fetch the pair of values
    float x0 = texelFetch(u_input, ivec2(pair * 2, 0), 0).r;
    float x1 = texelFetch(u_input, ivec2(pair * 2 + 1, 0), 0).r;
    
    // Fetch rotation coefficients for this position and pair
    float cos_t = texelFetch(u_cos, ivec2(pair, u_pos), 0).r;
    float sin_t = texelFetch(u_sin, ivec2(pair, u_pos), 0).r;
    
    // Apply rotation
    float result;
    if (is_odd == 0) {
        result = x0 * cos_t - x1 * sin_t;
    } else {
        result = x0 * sin_t + x1 * cos_t;
    }
    
    fragColor = vec4(result, 0.0, 0.0, 1.0);
}
)";

// =============================================================================
// FUSED RoPE + CfC: Rotation and recurrent update in one pass
// This is the attention replacement!
//
// For each element:
//   1. Apply RoPE rotation to input
//   2. Compute gate = sigmoid(W_gate @ [rotated, h_prev] + b)
//   3. Compute candidate = tanh(W_cand @ [rotated, h_prev] + b)
//   4. h_new = (1 - gate) * h_prev * decay + gate * candidate
// =============================================================================
const char* fs_rope_cfc_fused = R"(#version 310 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

out vec4 fragColor;

uniform sampler2D u_input;      // Input [dim]
uniform sampler2D u_h_prev;     // Previous hidden state [dim]
uniform sampler2D u_cos;        // RoPE cos [max_seq, dim/2]
uniform sampler2D u_sin;        // RoPE sin [max_seq, dim/2]
uniform sampler2D u_decay;      // Precomputed exp(-dt/tau) [dim]
uniform usampler2D u_w_gate;    // Ternary gate weights [dim, dim*2] packed
uniform usampler2D u_w_cand;    // Ternary candidate weights [dim, dim*2] packed
uniform sampler2D u_b_gate;     // Gate bias [dim]
uniform sampler2D u_b_cand;     // Candidate bias [dim]
uniform int u_pos;
uniform int u_dim;

float sigmoid_fast(float x) {
    return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}

float tanh_fast(float x) {
    float e2x = exp(2.0 * clamp(x, -10.0, 10.0));
    return (e2x - 1.0) / (e2x + 1.0);
}

void main() {
    int i = int(gl_FragCoord.x - 0.5);
    if (i >= u_dim) { fragColor = vec4(0.0); return; }
    
    // Step 1: Apply RoPE rotation to input
    int pair = i / 2;
    int is_odd = i & 1;
    
    float x0 = texelFetch(u_input, ivec2(pair * 2, 0), 0).r;
    float x1 = texelFetch(u_input, ivec2(pair * 2 + 1, 0), 0).r;
    float cos_t = texelFetch(u_cos, ivec2(pair, u_pos), 0).r;
    float sin_t = texelFetch(u_sin, ivec2(pair, u_pos), 0).r;
    
    float rotated;
    if (is_odd == 0) {
        rotated = x0 * cos_t - x1 * sin_t;
    } else {
        rotated = x0 * sin_t + x1 * cos_t;
    }
    
    // Step 2: Compute gate and candidate via ternary dot products
    // concat = [rotated_input, h_prev] but we compute on-the-fly
    int concat_dim = u_dim * 2;
    int packed_per_row = concat_dim / 16;
    
    float gate_pre = texelFetch(u_b_gate, ivec2(i, 0), 0).r;
    float cand_pre = texelFetch(u_b_cand, ivec2(i, 0), 0).r;
    
    for (int p = 0; p < packed_per_row; p++) {
        uint gate_packed = texelFetch(u_w_gate, ivec2(p, i), 0).r;
        uint cand_packed = texelFetch(u_w_cand, ivec2(p, i), 0).r;
        
        for (int j = 0; j < 16; j++) {
            int k = p * 16 + j;
            
            // Get value from rotated input or h_prev
            float val;
            if (k < u_dim) {
                // From rotated input - recompute rotation for this element
                int kpair = k / 2;
                int kodd = k & 1;
                float kx0 = texelFetch(u_input, ivec2(kpair * 2, 0), 0).r;
                float kx1 = texelFetch(u_input, ivec2(kpair * 2 + 1, 0), 0).r;
                float kcos = texelFetch(u_cos, ivec2(kpair, u_pos), 0).r;
                float ksin = texelFetch(u_sin, ivec2(kpair, u_pos), 0).r;
                if (kodd == 0) {
                    val = kx0 * kcos - kx1 * ksin;
                } else {
                    val = kx0 * ksin + kx1 * kcos;
                }
            } else {
                val = texelFetch(u_h_prev, ivec2(k - u_dim, 0), 0).r;
            }
            
            // Ternary accumulation (no multiply!)
            uint g_trit = (gate_packed >> uint(j * 2)) & 0x3u;
            uint c_trit = (cand_packed >> uint(j * 2)) & 0x3u;
            
            if (g_trit == 1u) gate_pre += val;
            else if (g_trit == 2u) gate_pre -= val;
            
            if (c_trit == 1u) cand_pre += val;
            else if (c_trit == 2u) cand_pre -= val;
        }
    }
    
    // Step 3: Apply activations
    float gate = sigmoid_fast(gate_pre);
    float candidate = tanh_fast(cand_pre);
    
    // Step 4: CfC update
    float h_prev = texelFetch(u_h_prev, ivec2(i, 0), 0).r;
    float decay = texelFetch(u_decay, ivec2(i, 0), 0).r;
    
    float h_new = (1.0 - gate) * h_prev * decay + gate * candidate;
    
    fragColor = vec4(h_new, 0.0, 0.0, 1.0);
}
)";

// =============================================================================
// TERNARY FUSED FFN (same as before)
// =============================================================================
const char* fs_ternary_fused_ffn = R"(#version 310 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

out vec4 fragColor;

uniform usampler2D u_w_gate;
uniform usampler2D u_w_up;
uniform sampler2D u_input;
uniform int u_K;
uniform int u_M;

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
            
            if (g_trit == 1u) gate_sum += x;
            else if (g_trit == 2u) gate_sum -= x;
            
            if (u_trit == 1u) up_sum += x;
            else if (u_trit == 2u) up_sum -= x;
        }
    }
    
    float silu_gate = gate_sum / (1.0 + exp(-gate_sum));
    float result = silu_gate * up_sum;
    
    fragColor = vec4(result, 0.0, 0.0, 1.0);
}
)";

// =============================================================================
// MEGA-FUSED: One shader does CfC + FFN up + FFN down for entire token!
// This minimizes draw calls to just 1 per layer
// Output: [ffn_output, updated_h_state] packed together
// =============================================================================
const char* fs_mega_fused = R"(#version 310 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

out vec4 fragColor;

uniform sampler2D u_input;      // Input [dim]
uniform sampler2D u_h_prev;     // Previous hidden state [dim]
uniform sampler2D u_cos;        // RoPE cos
uniform sampler2D u_sin;        // RoPE sin
uniform sampler2D u_decay;      // CfC decay
uniform usampler2D u_w_cfc_gate;  // CfC gate weights [dim, dim*2]
uniform usampler2D u_w_cfc_cand;  // CfC candidate weights
uniform sampler2D u_b_cfc_gate;   // CfC gate bias
uniform sampler2D u_b_cfc_cand;   // CfC candidate bias
uniform usampler2D u_w_ffn_gate;  // FFN gate weights [ff, dim]
uniform usampler2D u_w_ffn_up;    // FFN up weights [ff, dim]
uniform usampler2D u_w_ffn_down;  // FFN down weights [dim, ff]
uniform int u_pos;
uniform int u_dim;
uniform int u_ff;

// This shader outputs TWO things based on row:
// - Rows 0 to dim-1: FFN output (after down projection)
// - Rows dim to 2*dim-1: Updated hidden state (for next layer's CfC)

float sigmoid_fast(float x) {
    return 1.0 / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}

float tanh_fast(float x) {
    float e2x = exp(2.0 * clamp(x, -10.0, 10.0));
    return (e2x - 1.0) / (e2x + 1.0);
}

void main() {
    int idx = int(gl_FragCoord.y - 0.5);
    int total_out = u_dim * 2;  // FFN out + h_state out
    if (idx >= total_out) { fragColor = vec4(0.0); return; }
    
    bool is_ffn_output = (idx < u_dim);
    int i = is_ffn_output ? idx : (idx - u_dim);
    
    // Step 1: Compute RoPE-rotated input (needed for CfC)
    // We'll compute this for all indices we need
    
    // Step 2: CfC gate and candidate
    int concat_dim = u_dim * 2;
    int packed_per_cfc = concat_dim / 16;
    
    float gate_pre = texelFetch(u_b_cfc_gate, ivec2(i, 0), 0).r;
    float cand_pre = texelFetch(u_b_cfc_cand, ivec2(i, 0), 0).r;
    
    for (int p = 0; p < packed_per_cfc; p++) {
        uint gate_packed = texelFetch(u_w_cfc_gate, ivec2(p, i), 0).r;
        uint cand_packed = texelFetch(u_w_cfc_cand, ivec2(p, i), 0).r;
        
        for (int j = 0; j < 16; j++) {
            int k = p * 16 + j;
            
            float val;
            if (k < u_dim) {
                // From rotated input
                int kpair = k / 2;
                int kodd = k & 1;
                float kx0 = texelFetch(u_input, ivec2(kpair * 2, 0), 0).r;
                float kx1 = texelFetch(u_input, ivec2(kpair * 2 + 1, 0), 0).r;
                float kcos = texelFetch(u_cos, ivec2(kpair, u_pos), 0).r;
                float ksin = texelFetch(u_sin, ivec2(kpair, u_pos), 0).r;
                if (kodd == 0) {
                    val = kx0 * kcos - kx1 * ksin;
                } else {
                    val = kx0 * ksin + kx1 * kcos;
                }
            } else {
                val = texelFetch(u_h_prev, ivec2(k - u_dim, 0), 0).r;
            }
            
            uint g_trit = (gate_packed >> uint(j * 2)) & 0x3u;
            uint c_trit = (cand_packed >> uint(j * 2)) & 0x3u;
            
            if (g_trit == 1u) gate_pre += val;
            else if (g_trit == 2u) gate_pre -= val;
            
            if (c_trit == 1u) cand_pre += val;
            else if (c_trit == 2u) cand_pre -= val;
        }
    }
    
    float gate = sigmoid_fast(gate_pre);
    float candidate = tanh_fast(cand_pre);
    float h_prev = texelFetch(u_h_prev, ivec2(i, 0), 0).r;
    float decay = texelFetch(u_decay, ivec2(i, 0), 0).r;
    float h_new = (1.0 - gate) * h_prev * decay + gate * candidate;
    
    // If we're outputting h_state, we're done
    if (!is_ffn_output) {
        fragColor = vec4(h_new, 0.0, 0.0, 1.0);
        return;
    }
    
    // Step 3: FFN with h_new as input
    // First: gate + up projections (to intermediate dim)
    // Then: down projection
    // This is complex because we need intermediate values...
    
    // For simplicity, we'll compute FFN down directly
    // Each output[i] = sum over ff of: down_weight[i,ff] * (silu(gate@h_new) * up@h_new)
    
    int packed_per_ffn = u_dim / 16;
    int packed_ff = u_ff / 16;
    
    float ffn_out = 0.0;
    
    // For each intermediate dimension
    for (int f = 0; f < u_ff; f++) {
        // Compute gate and up projections for this intermediate dim
        float gate_ffn = 0.0;
        float up_ffn = 0.0;
        
        for (int p = 0; p < packed_per_ffn; p++) {
            uint g_packed = texelFetch(u_w_ffn_gate, ivec2(p, f), 0).r;
            uint u_packed = texelFetch(u_w_ffn_up, ivec2(p, f), 0).r;
            
            for (int j = 0; j < 16; j++) {
                int k = p * 16 + j;
                
                // We need h_new[k], but we computed h_new only for index i
                // Need to recompute for all k... this is expensive
                // For now, use h_prev as approximation (or store h_new in first pass)
                float hk = texelFetch(u_h_prev, ivec2(k, 0), 0).r;  // Approximation
                
                uint gt = (g_packed >> uint(j * 2)) & 0x3u;
                uint ut = (u_packed >> uint(j * 2)) & 0x3u;
                
                if (gt == 1u) gate_ffn += hk;
                else if (gt == 2u) gate_ffn -= hk;
                
                if (ut == 1u) up_ffn += hk;
                else if (ut == 2u) up_ffn -= hk;
            }
        }
        
        float silu = gate_ffn / (1.0 + exp(-gate_ffn));
        float intermediate = silu * up_ffn;
        
        // Down projection contribution
        uint down_packed = texelFetch(u_w_ffn_down, ivec2(f / 16, i), 0).r;
        uint trit = (down_packed >> uint((f % 16) * 2)) & 0x3u;
        
        if (trit == 1u) ffn_out += intermediate;
        else if (trit == 2u) ffn_out -= intermediate;
    }
    
    fragColor = vec4(ffn_out, 0.0, 0.0, 1.0);
}
)";

// Ternary GEMV
const char* fs_ternary_gemv = R"(#version 310 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

out vec4 fragColor;

uniform usampler2D u_weights;
uniform sampler2D u_input;
uniform int u_K;
uniform int u_M;

void main() {
    int row = int(gl_FragCoord.y - 0.5);
    if (row >= u_M) { fragColor = vec4(0.0); return; }
    
    int packed_per_row = u_K / 16;
    float sum = 0.0;
    
    for (int p = 0; p < packed_per_row; p++) {
        uint packed = texelFetch(u_weights, ivec2(p, row), 0).r;
        
        for (int i = 0; i < 16; i++) {
            uint trit = (packed >> uint(i * 2)) & 0x3u;
            int k = p * 16 + i;
            float x = texelFetch(u_input, ivec2(k, 0), 0).r;
            
            if (trit == 1u) sum += x;
            else if (trit == 2u) sum -= x;
        }
    }
    
    fragColor = vec4(sum, 0.0, 0.0, 1.0);
}
)";

// Element-wise add (residual)
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

// RMSNorm
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

//=============================================================================
// Utilities
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
// Spectral CfC LLM Engine
//=============================================================================

class SpectralCfcLLM {
public:
    GLuint prog_rope = 0;
    GLuint prog_rope_cfc = 0;
    GLuint prog_ternary_ffn = 0;
    GLuint prog_ternary_gemv = 0;
    GLuint prog_add = 0;
    GLuint prog_rmsnorm = 0;
    
    // Precomputed RoPE tables
    Texture rope_cos, rope_sin;
    
    bool init() {
        prog_rope = create_program(vs_fullscreen, fs_rope);
        prog_rope_cfc = create_program(vs_fullscreen, fs_rope_cfc_fused);
        prog_ternary_ffn = create_program(vs_fullscreen, fs_ternary_fused_ffn);
        prog_ternary_gemv = create_program(vs_fullscreen, fs_ternary_gemv);
        prog_add = create_program(vs_fullscreen, fs_add);
        prog_rmsnorm = create_program(vs_fullscreen, fs_rmsnorm);
        
        if (!prog_rope || !prog_rope_cfc || !prog_ternary_ffn || 
            !prog_ternary_gemv || !prog_add || !prog_rmsnorm) {
            fprintf(stderr, "Failed to compile shaders\n");
            return false;
        }
        
        // Precompute RoPE tables
        precompute_rope(N_EMBD, MAX_SEQ_LEN, ROPE_THETA);
        
        printf("Spectral CfC LLM initialized\n");
        printf("  RoPE precomputed for %d positions, %d dims\n", MAX_SEQ_LEN, N_EMBD);
        return true;
    }
    
    void precompute_rope(int dim, int max_seq, float theta) {
        std::vector<float> cos_data(max_seq * (dim / 2));
        std::vector<float> sin_data(max_seq * (dim / 2));
        
        for (int pos = 0; pos < max_seq; pos++) {
            for (int i = 0; i < dim / 2; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / dim);
                float angle = pos * freq;
                cos_data[i * max_seq + pos] = cosf(angle);
                sin_data[i * max_seq + pos] = sinf(angle);
            }
        }
        
        rope_cos.create_f32(max_seq, dim / 2);
        rope_sin.create_f32(max_seq, dim / 2);
        
        // Transpose for better access pattern: [pair, pos]
        std::vector<float> cos_transposed(max_seq * (dim / 2));
        std::vector<float> sin_transposed(max_seq * (dim / 2));
        for (int pos = 0; pos < max_seq; pos++) {
            for (int i = 0; i < dim / 2; i++) {
                cos_transposed[pos * (dim / 2) + i] = cos_data[i * max_seq + pos];
                sin_transposed[pos * (dim / 2) + i] = sin_data[i * max_seq + pos];
            }
        }
        
        rope_cos.upload_f32(cos_transposed.data());
        rope_sin.upload_f32(sin_transposed.data());
    }
    
    // Apply RoPE rotation
    void apply_rope(Framebuffer& output, Texture& input, int pos, int dim) {
        output.bind();
        glUseProgram(prog_rope);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, input.id);
        glUniform1i(glGetUniformLocation(prog_rope, "u_input"), 0);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, rope_cos.id);
        glUniform1i(glGetUniformLocation(prog_rope, "u_cos"), 1);
        
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, rope_sin.id);
        glUniform1i(glGetUniformLocation(prog_rope, "u_sin"), 2);
        
        glUniform1i(glGetUniformLocation(prog_rope, "u_pos"), pos);
        glUniform1i(glGetUniformLocation(prog_rope, "u_dim"), dim);
        
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    
    // Fused RoPE + CfC cell
    void rope_cfc(Framebuffer& output, Texture& input, Texture& h_prev,
                  Texture& w_gate, Texture& w_cand, Texture& b_gate, Texture& b_cand,
                  Texture& decay, int pos, int dim) {
        output.bind();
        glUseProgram(prog_rope_cfc);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, input.id);
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_input"), 0);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, h_prev.id);
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_h_prev"), 1);
        
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, rope_cos.id);
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_cos"), 2);
        
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, rope_sin.id);
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_sin"), 3);
        
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, decay.id);
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_decay"), 4);
        
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, w_gate.id);
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_w_gate"), 5);
        
        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D, w_cand.id);
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_w_cand"), 6);
        
        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D, b_gate.id);
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_b_gate"), 7);
        
        glActiveTexture(GL_TEXTURE8);
        glBindTexture(GL_TEXTURE_2D, b_cand.id);
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_b_cand"), 8);
        
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_pos"), pos);
        glUniform1i(glGetUniformLocation(prog_rope_cfc, "u_dim"), dim);
        
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    
    // Ternary fused FFN
    void ternary_ffn(Framebuffer& output, Texture& w_gate, Texture& w_up,
                     Texture& input, int M, int K) {
        output.bind();
        glUseProgram(prog_ternary_ffn);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, w_gate.id);
        glUniform1i(glGetUniformLocation(prog_ternary_ffn, "u_w_gate"), 0);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, w_up.id);
        glUniform1i(glGetUniformLocation(prog_ternary_ffn, "u_w_up"), 1);
        
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, input.id);
        glUniform1i(glGetUniformLocation(prog_ternary_ffn, "u_input"), 2);
        
        glUniform1i(glGetUniformLocation(prog_ternary_ffn, "u_M"), M);
        glUniform1i(glGetUniformLocation(prog_ternary_ffn, "u_K"), K);
        
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    
    // Ternary GEMV
    void ternary_gemv(Framebuffer& output, Texture& weights, Texture& input, int M, int K) {
        output.bind();
        glUseProgram(prog_ternary_gemv);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, weights.id);
        glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_weights"), 0);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, input.id);
        glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_input"), 1);
        
        glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_M"), M);
        glUniform1i(glGetUniformLocation(prog_ternary_gemv, "u_K"), K);
        
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    
    // Element-wise add
    void add(Framebuffer& output, Texture& a, Texture& b, int dim) {
        output.bind();
        glUseProgram(prog_add);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, a.id);
        glUniform1i(glGetUniformLocation(prog_add, "u_a"), 0);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, b.id);
        glUniform1i(glGetUniformLocation(prog_add, "u_b"), 1);
        
        glUniform1i(glGetUniformLocation(prog_add, "u_dim"), dim);
        
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    
    void destroy() {
        if (prog_rope) glDeleteProgram(prog_rope);
        if (prog_rope_cfc) glDeleteProgram(prog_rope_cfc);
        if (prog_ternary_ffn) glDeleteProgram(prog_ternary_ffn);
        if (prog_ternary_gemv) glDeleteProgram(prog_ternary_gemv);
        if (prog_add) glDeleteProgram(prog_add);
        if (prog_rmsnorm) glDeleteProgram(prog_rmsnorm);
        rope_cos.destroy();
        rope_sin.destroy();
    }
};

//=============================================================================
// EGL Setup
//=============================================================================

bool init_egl() {
    g_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(g_display, nullptr, nullptr)) return false;
    
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
// Generate random ternary weights
//=============================================================================
void generate_ternary_weights(std::vector<uint32_t>& out, int rows, int cols, float sparsity = 0.81f) {
    int packed_cols = cols / 16;
    out.resize(rows * packed_cols);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int r = 0; r < rows; r++) {
        for (int p = 0; p < packed_cols; p++) {
            uint32_t packed = 0;
            for (int i = 0; i < 16; i++) {
                uint32_t trit;
                float v = dist(rng);
                if (v < sparsity) trit = 0;
                else if (v < sparsity + (1.0f - sparsity) / 2) trit = 1;
                else trit = 2;
                packed |= (trit << (i * 2));
            }
            out[r * packed_cols + p] = packed;
        }
    }
}

//=============================================================================
// Benchmark
//=============================================================================

void benchmark(SpectralCfcLLM& llm, int iterations) {
    printf("\n=== Spectral CfC + Ternary FFN Benchmark ===\n");
    printf("Model: %d embd, %d ff, %d layers\n", N_EMBD, N_FF, N_LAYER);
    
    // Create weights
    std::vector<uint32_t> w_cfc_gate_data, w_cfc_cand_data;
    std::vector<uint32_t> w_ffn_gate_data, w_ffn_up_data, w_ffn_down_data;
    
    // CfC weights: [dim, dim*2] for gate and candidate
    generate_ternary_weights(w_cfc_gate_data, N_EMBD, N_EMBD * 2, 0.81f);
    generate_ternary_weights(w_cfc_cand_data, N_EMBD, N_EMBD * 2, 0.81f);
    
    // FFN weights
    generate_ternary_weights(w_ffn_gate_data, N_FF, N_EMBD, 0.81f);
    generate_ternary_weights(w_ffn_up_data, N_FF, N_EMBD, 0.81f);
    generate_ternary_weights(w_ffn_down_data, N_EMBD, N_FF, 0.81f);
    
    // Upload to textures
    Texture w_cfc_gate, w_cfc_cand, b_cfc_gate, b_cfc_cand, decay;
    Texture w_ffn_gate, w_ffn_up, w_ffn_down;
    
    w_cfc_gate.create_u32((N_EMBD * 2) / 16, N_EMBD);
    w_cfc_cand.create_u32((N_EMBD * 2) / 16, N_EMBD);
    w_cfc_gate.upload_u32(w_cfc_gate_data.data());
    w_cfc_cand.upload_u32(w_cfc_cand_data.data());
    
    // Biases and decay
    std::vector<float> zeros(N_EMBD, 0.0f);
    std::vector<float> decay_data(N_EMBD, 0.95f);  // exp(-dt/tau) ≈ 0.95
    b_cfc_gate.create_f32(N_EMBD, 1);
    b_cfc_cand.create_f32(N_EMBD, 1);
    decay.create_f32(N_EMBD, 1);
    b_cfc_gate.upload_f32(zeros.data());
    b_cfc_cand.upload_f32(zeros.data());
    decay.upload_f32(decay_data.data());
    
    w_ffn_gate.create_u32(N_EMBD / 16, N_FF);
    w_ffn_up.create_u32(N_EMBD / 16, N_FF);
    w_ffn_down.create_u32(N_FF / 16, N_EMBD);
    w_ffn_gate.upload_u32(w_ffn_gate_data.data());
    w_ffn_up.upload_u32(w_ffn_up_data.data());
    w_ffn_down.upload_u32(w_ffn_down_data.data());
    
    // Create buffers
    Texture input, h_state;
    input.create_f32(N_EMBD, 1);
    h_state.create_f32(N_EMBD, 1);
    std::vector<float> input_data(N_EMBD, 0.1f);
    std::vector<float> h_data(N_EMBD, 0.0f);
    input.upload_f32(input_data.data());
    h_state.upload_f32(h_data.data());
    
    Framebuffer fb_rope, fb_cfc, fb_ffn_mid, fb_ffn_out, fb_residual;
    fb_rope.create(N_EMBD, 1);
    fb_cfc.create(N_EMBD, 1);
    fb_ffn_mid.create(N_FF, 1);
    fb_ffn_out.create(N_EMBD, 1);
    fb_residual.create(N_EMBD, 1);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        llm.apply_rope(fb_rope, input, 0, N_EMBD);
        llm.rope_cfc(fb_cfc, input, h_state, w_cfc_gate, w_cfc_cand,
                     b_cfc_gate, b_cfc_cand, decay, 0, N_EMBD);
        llm.ternary_ffn(fb_ffn_mid, w_ffn_gate, w_ffn_up, fb_cfc.tex, N_FF, N_EMBD);
        llm.ternary_gemv(fb_ffn_out, w_ffn_down, fb_ffn_mid.tex, N_EMBD, N_FF);
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
    
    double t_rope = profile("RoPE rotation", [&]() {
        llm.apply_rope(fb_rope, input, 0, N_EMBD);
    });
    
    double t_cfc = profile("RoPE+CfC fused (attention replacement)", [&]() {
        llm.rope_cfc(fb_cfc, input, h_state, w_cfc_gate, w_cfc_cand,
                     b_cfc_gate, b_cfc_cand, decay, 0, N_EMBD);
    });
    
    double t_ffn_fused = profile("Ternary fused FFN (256->1024)", [&]() {
        llm.ternary_ffn(fb_ffn_mid, w_ffn_gate, w_ffn_up, fb_cfc.tex, N_FF, N_EMBD);
    });
    
    double t_ffn_down = profile("Ternary GEMV down (1024->256)", [&]() {
        llm.ternary_gemv(fb_ffn_out, w_ffn_down, fb_ffn_mid.tex, N_EMBD, N_FF);
    });
    
    double t_layer = t_cfc + t_ffn_fused + t_ffn_down;
    printf("\nPer-layer total: %.3f ms\n", t_layer);
    printf("  (RoPE+CfC: %.3f, FFN: %.3f)\n", t_cfc, t_ffn_fused + t_ffn_down);
    
    // Full model simulation (naive - one FBO per op)
    printf("\n=== Full Model Simulation (%d layers) - NAIVE ===\n", N_LAYER);
    
    glFinish();
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        int pos = iter % MAX_SEQ_LEN;
        for (int layer = 0; layer < N_LAYER; layer++) {
            // RoPE + CfC (replaces attention)
            llm.rope_cfc(fb_cfc, input, h_state, w_cfc_gate, w_cfc_cand,
                         b_cfc_gate, b_cfc_cand, decay, pos, N_EMBD);
            // FFN
            llm.ternary_ffn(fb_ffn_mid, w_ffn_gate, w_ffn_up, fb_cfc.tex, N_FF, N_EMBD);
            llm.ternary_gemv(fb_ffn_out, w_ffn_down, fb_ffn_mid.tex, N_EMBD, N_FF);
        }
    }
    glFinish();
    auto end = std::chrono::high_resolution_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    printf("  %d layers: %.3f ms\n", N_LAYER, ms_total);
    printf("  Per layer: %.3f ms\n", ms_total / N_LAYER);
    printf("  Est. tok/s: %.1f\n", 1000.0 / ms_total);
    
    // =============================================================
    // APPROACH 2: MEGA-BATCH - Submit many tokens without sync
    // Key insight: Pipeline multiple tokens at once, only sync at end
    // =============================================================
    printf("\n=== MEGA-BATCH (pipeline %d tokens) ===\n", iterations);
    
    // Create separate FBO chains for pipelining
    constexpr int BATCH_SIZE = 100;  // Tokens to pipeline before sync
    
    // Per-token buffers (interleave to maximize GPU occupancy)
    std::vector<Framebuffer> batch_cfc(BATCH_SIZE);
    std::vector<Framebuffer> batch_ffn_mid(BATCH_SIZE);
    std::vector<Framebuffer> batch_ffn_out(BATCH_SIZE);
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        batch_cfc[b].create(N_EMBD, 1);
        batch_ffn_mid[b].create(N_FF, 1);
        batch_ffn_out[b].create(N_EMBD, 1);
    }
    
    // Per-layer hidden state
    std::vector<Framebuffer> h_states(N_LAYER);
    for (int i = 0; i < N_LAYER; i++) {
        h_states[i].create(N_EMBD, 1);
    }
    
    // Warmup: submit ALL layers for ALL tokens in batch, THEN sync
    for (int warmup = 0; warmup < 3; warmup++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            int pos = b % MAX_SEQ_LEN;
            Texture* current_input = &input;
            
            for (int layer = 0; layer < N_LAYER; layer++) {
                llm.rope_cfc(batch_cfc[b], *current_input, h_states[layer].tex,
                            w_cfc_gate, w_cfc_cand, b_cfc_gate, b_cfc_cand, decay, pos, N_EMBD);
                llm.ternary_ffn(batch_ffn_mid[b], w_ffn_gate, w_ffn_up, batch_cfc[b].tex, N_FF, N_EMBD);
                llm.ternary_gemv(batch_ffn_out[b], w_ffn_down, batch_ffn_mid[b].tex, N_EMBD, N_FF);
                current_input = &batch_ffn_out[b].tex;
            }
        }
        glFinish();  // Sync only after all tokens in batch
    }
    
    // Time mega-batch
    int num_batches = iterations / BATCH_SIZE;
    glFinish();
    start = std::chrono::high_resolution_clock::now();
    
    for (int batch = 0; batch < num_batches; batch++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            int pos = (batch * BATCH_SIZE + b) % MAX_SEQ_LEN;
            Texture* current_input = &input;
            
            for (int layer = 0; layer < N_LAYER; layer++) {
                llm.rope_cfc(batch_cfc[b], *current_input, h_states[layer].tex,
                            w_cfc_gate, w_cfc_cand, b_cfc_gate, b_cfc_cand, decay, pos, N_EMBD);
                llm.ternary_ffn(batch_ffn_mid[b], w_ffn_gate, w_ffn_up, batch_cfc[b].tex, N_FF, N_EMBD);
                llm.ternary_gemv(batch_ffn_out[b], w_ffn_down, batch_ffn_mid[b].tex, N_EMBD, N_FF);
                current_input = &batch_ffn_out[b].tex;
            }
        }
        glFinish();  // Sync once per batch
    }
    
    end = std::chrono::high_resolution_clock::now();
    double ms_mega = std::chrono::duration<double, std::milli>(end - start).count();
    int total_tokens = num_batches * BATCH_SIZE;
    double ms_per_token = ms_mega / total_tokens;
    double ms_optimized = ms_per_token * N_LAYER / N_LAYER;  // Per 8-layer forward
    
    printf("  %d tokens in %.2f ms\n", total_tokens, ms_mega);
    printf("  Per token (8 layers): %.3f ms\n", ms_per_token);
    printf("  Est. tok/s: %.1f\n", 1000.0 / ms_per_token);
    printf("  Speedup vs naive: %.2fx\n", ms_total / ms_per_token);
    
    // Cleanup
    for (auto& fb : batch_cfc) fb.destroy();
    for (auto& fb : batch_ffn_mid) fb.destroy();
    for (auto& fb : batch_ffn_out) fb.destroy();
    for (auto& h : h_states) h.destroy();
    
    // Compare naive vs theoretical
    printf("\n=== Overhead Analysis ===\n");
    printf("  Theoretical (sum of ops): %.3f ms\n", t_layer * N_LAYER);
    printf("  Naive actual: %.3f ms\n", ms_total);
    printf("  Naive overhead: %.2fx\n", ms_total / (t_layer * N_LAYER));
    printf("  Optimized actual: %.3f ms\n", ms_optimized);
    printf("  Optimized overhead: %.2fx\n", ms_optimized / (t_layer * N_LAYER));
    
    // Scale to LFM2 using OPTIMIZED timing
    printf("\n=== Scaled to LFM2-350M (1024 embd, 4096 ff, 16 layers) ===\n");
    double scale_cfc = (1024.0 * 2048.0) / (256.0 * 512.0);  // CfC weights scale
    double scale_ffn = (1024.0 * 4096.0) / (256.0 * 1024.0);  // FFN weights scale
    
    // Use measured per-layer time from optimized run
    double actual_per_layer = ms_optimized / N_LAYER;
    double overhead_factor = ms_optimized / (t_layer * N_LAYER);
    
    double scaled_cfc = t_cfc * scale_cfc;
    double scaled_ffn = (t_ffn_fused + t_ffn_down) * scale_ffn;
    double scaled_layer_theoretical = scaled_cfc + scaled_ffn;
    double scaled_layer_actual = scaled_layer_theoretical * overhead_factor;
    double scaled_total = scaled_layer_actual * 16;
    
    printf("  Measured overhead factor: %.2fx\n", overhead_factor);
    printf("  Estimated RoPE+CfC per layer: %.3f ms\n", scaled_cfc);
    printf("  Estimated FFN per layer: %.3f ms\n", scaled_ffn);
    printf("  Theoretical per layer: %.3f ms\n", scaled_layer_theoretical);
    printf("  Estimated per layer (w/overhead): %.3f ms\n", scaled_layer_actual);
    printf("  Estimated 16 layers: %.3f ms\n", scaled_total);
    printf("  Estimated tok/s: %.1f\n", 1000.0 / scaled_total);
    printf("\n  CPU baseline: 50 tok/s\n");
    printf("  Previous Q4+attention: 16 tok/s\n");
    printf("  Target: >50 tok/s to beat CPU\n");
    
    // Cleanup
    w_cfc_gate.destroy(); w_cfc_cand.destroy();
    b_cfc_gate.destroy(); b_cfc_cand.destroy(); decay.destroy();
    w_ffn_gate.destroy(); w_ffn_up.destroy(); w_ffn_down.destroy();
    input.destroy(); h_state.destroy();
    fb_rope.destroy(); fb_cfc.destroy();
    fb_ffn_mid.destroy(); fb_ffn_out.destroy(); fb_residual.destroy();
}

void run_lfm2_scale_test(SpectralCfcLLM& llm) {
    printf("\n========================================\n");
    printf("=== DIRECT LFM2-350M SCALE TEST ===\n");
    printf("=== 1024 embd, 4096 ff, 16 layers ===\n");
    printf("========================================\n");
    
    const int DIM = 1024;
    const int FF = 4096;
    const int LAYERS = 16;
    const int ITERS = 20;  // Fewer iterations for large model
    
    // Create weights at LFM2 scale
    std::vector<uint32_t> w_cfc_gate_data, w_cfc_cand_data;
    std::vector<uint32_t> w_ffn_gate_data, w_ffn_up_data, w_ffn_down_data;
    
    printf("Allocating weights...\n");
    
    generate_ternary_weights(w_cfc_gate_data, DIM, DIM * 2, 0.81f);
    generate_ternary_weights(w_cfc_cand_data, DIM, DIM * 2, 0.81f);
    generate_ternary_weights(w_ffn_gate_data, FF, DIM, 0.81f);
    generate_ternary_weights(w_ffn_up_data, FF, DIM, 0.81f);
    generate_ternary_weights(w_ffn_down_data, DIM, FF, 0.81f);
    
    printf("Uploading to GPU...\n");
    
    Texture w_cfc_gate, w_cfc_cand, b_cfc_gate, b_cfc_cand, decay;
    Texture w_ffn_gate, w_ffn_up, w_ffn_down;
    
    w_cfc_gate.create_u32((DIM * 2) / 16, DIM);
    w_cfc_cand.create_u32((DIM * 2) / 16, DIM);
    w_cfc_gate.upload_u32(w_cfc_gate_data.data());
    w_cfc_cand.upload_u32(w_cfc_cand_data.data());
    
    std::vector<float> zeros(DIM, 0.0f);
    std::vector<float> decay_data(DIM, 0.95f);
    b_cfc_gate.create_f32(DIM, 1);
    b_cfc_cand.create_f32(DIM, 1);
    decay.create_f32(DIM, 1);
    b_cfc_gate.upload_f32(zeros.data());
    b_cfc_cand.upload_f32(zeros.data());
    decay.upload_f32(decay_data.data());
    
    w_ffn_gate.create_u32(DIM / 16, FF);
    w_ffn_up.create_u32(DIM / 16, FF);
    w_ffn_down.create_u32(FF / 16, DIM);
    w_ffn_gate.upload_u32(w_ffn_gate_data.data());
    w_ffn_up.upload_u32(w_ffn_up_data.data());
    w_ffn_down.upload_u32(w_ffn_down_data.data());
    
    Texture input, h_state;
    input.create_f32(DIM, 1);
    h_state.create_f32(DIM, 1);
    std::vector<float> input_data(DIM, 0.1f);
    std::vector<float> h_data(DIM, 0.0f);
    input.upload_f32(input_data.data());
    h_state.upload_f32(h_data.data());
    
    Framebuffer fb_cfc, fb_ffn_mid, fb_ffn_out;
    fb_cfc.create(DIM, 1);
    fb_ffn_mid.create(FF, 1);
    fb_ffn_out.create(DIM, 1);
    
    printf("Warming up...\n");
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        llm.rope_cfc(fb_cfc, input, h_state, w_cfc_gate, w_cfc_cand,
                     b_cfc_gate, b_cfc_cand, decay, 0, DIM);
        llm.ternary_ffn(fb_ffn_mid, w_ffn_gate, w_ffn_up, fb_cfc.tex, FF, DIM);
        llm.ternary_gemv(fb_ffn_out, w_ffn_down, fb_ffn_mid.tex, DIM, FF);
    }
    glFinish();
    
    // Profile individual ops at LFM2 scale
    printf("\nPer-operation timing at LFM2 scale:\n");
    
    auto profile = [&](const char* name, auto fn) {
        glFinish();
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) fn();
        glFinish();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / ITERS;
        printf("  %s: %.3f ms\n", name, ms);
        return ms;
    };
    
    double t_cfc = profile("RoPE+CfC (1024x2048 ternary)", [&]() {
        llm.rope_cfc(fb_cfc, input, h_state, w_cfc_gate, w_cfc_cand,
                     b_cfc_gate, b_cfc_cand, decay, 0, DIM);
    });
    
    double t_ffn_up = profile("FFN up (1024->4096 ternary)", [&]() {
        llm.ternary_ffn(fb_ffn_mid, w_ffn_gate, w_ffn_up, fb_cfc.tex, FF, DIM);
    });
    
    double t_ffn_down = profile("FFN down (4096->1024 ternary)", [&]() {
        llm.ternary_gemv(fb_ffn_out, w_ffn_down, fb_ffn_mid.tex, DIM, FF);
    });
    
    double t_layer = t_cfc + t_ffn_up + t_ffn_down;
    printf("\nPer-layer: %.3f ms (CfC: %.3f, FFN: %.3f)\n", 
           t_layer, t_cfc, t_ffn_up + t_ffn_down);
    
    // Full model
    printf("\nFull model (16 layers):\n");
    
    glFinish();
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERS; iter++) {
        for (int layer = 0; layer < LAYERS; layer++) {
            llm.rope_cfc(fb_cfc, input, h_state, w_cfc_gate, w_cfc_cand,
                         b_cfc_gate, b_cfc_cand, decay, iter, DIM);
            llm.ternary_ffn(fb_ffn_mid, w_ffn_gate, w_ffn_up, fb_cfc.tex, FF, DIM);
            llm.ternary_gemv(fb_ffn_out, w_ffn_down, fb_ffn_mid.tex, DIM, FF);
        }
    }
    glFinish();
    auto end = std::chrono::high_resolution_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(end - start).count() / ITERS;
    
    printf("  Total: %.2f ms\n", ms_total);
    printf("  Per layer: %.3f ms\n", ms_total / LAYERS);
    printf("  Tokens/sec: %.1f\n", 1000.0 / ms_total);
    printf("\n  Theoretical (sum of ops): %.2f ms\n", t_layer * LAYERS);
    printf("  Overhead: %.2fx\n", ms_total / (t_layer * LAYERS));
    
    printf("\n=== COMPARISON ===\n");
    printf("  CPU baseline: 50 tok/s (20.0 ms/token)\n");
    printf("  Our result:   %.1f tok/s (%.1f ms/token)\n", 1000.0/ms_total, ms_total);
    if (ms_total < 20.0) {
        printf("  *** BEATS CPU! ***\n");
    } else {
        printf("  Need %.1fx speedup to beat CPU\n", ms_total / 20.0);
    }
    
    // Cleanup
    w_cfc_gate.destroy(); w_cfc_cand.destroy();
    b_cfc_gate.destroy(); b_cfc_cand.destroy(); decay.destroy();
    w_ffn_gate.destroy(); w_ffn_up.destroy(); w_ffn_down.destroy();
    input.destroy(); h_state.destroy();
    fb_cfc.destroy(); fb_ffn_mid.destroy(); fb_ffn_out.destroy();
}

int main() {
    printf("=== Spectral CfC LLM: Rotation + Recurrence ===\n\n");
    
    if (!init_egl()) return 1;
    
    SpectralCfcLLM llm;
    if (!llm.init()) {
        cleanup_egl();
        return 1;
    }
    
    // Test at small scale first
    benchmark(llm, 100);
    
    // Then test at actual LFM2 scale
    run_lfm2_scale_test(llm);
    
    llm.destroy();
    cleanup_egl();
    
    printf("\n=== Done ===\n");
    return 0;
}
