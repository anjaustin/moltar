// Spectral-Rotational-CfC FFN: A Unified Primitive
//
// Hypothesis: Instead of RoPE → CfC → FFN as separate steps,
// what if we fuse them into a single "SRC-FFN" operation?
//
// Traditional FFN:
//   mid = SiLU(x @ W_gate) * (x @ W_up)
//   out = mid @ W_down
//
// SRC-FFN (Spectral-Rotational-CfC FFN):
//   x_rot = RoPE(x, position)  // Spectral rotation
//   gate = x_rot @ W_gate
//   up = CfC(x_rot @ W_up, h_prev)  // Continuous recurrence IN the FFN
//   mid = SiLU(gate) * up
//   out = mid @ W_down
//
// Why this might be powerful:
// 1. Position encoding is INSIDE the nonlinearity, not before it
// 2. The "up" projection has memory - each FFN row remembers its history
// 3. Gate controls BOTH feature activation AND temporal mixing
//
// This is essentially: "What if every FFN neuron was a tiny RNN?"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <atomic>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

//=============================================================================
// Configuration
//=============================================================================

constexpr int EMBD = 1024;
constexpr int FF = 4096;
constexpr int LAYERS = 16;
constexpr int MAX_SEQ = 512;
constexpr float ROPE_THETA = 10000.0f;

// Core mapping
constexpr int A55_CORES[] = {0, 1, 2, 3};
constexpr int A78_CORES[] = {6, 7};
constexpr int NUM_A55 = 4;
constexpr int NUM_A78 = 2;
constexpr int TOTAL_WORKERS = 6;

//=============================================================================
// Ternary NEON primitives (same as before)
//=============================================================================

#ifdef __ARM_NEON
alignas(64) static float BYTE_TO_SIGNS[256][4];

__attribute__((constructor))
static void init_trit_lut() {
    for (int b = 0; b < 256; b++) {
        BYTE_TO_SIGNS[b][0] = ((b & 0x03) == 1) ? 1.0f : (((b & 0x03) == 2) ? -1.0f : 0.0f);
        BYTE_TO_SIGNS[b][1] = (((b >> 2) & 0x03) == 1) ? 1.0f : ((((b >> 2) & 0x03) == 2) ? -1.0f : 0.0f);
        BYTE_TO_SIGNS[b][2] = (((b >> 4) & 0x03) == 1) ? 1.0f : ((((b >> 4) & 0x03) == 2) ? -1.0f : 0.0f);
        BYTE_TO_SIGNS[b][3] = (((b >> 6) & 0x03) == 1) ? 1.0f : ((((b >> 6) & 0x03) == 2) ? -1.0f : 0.0f);
    }
}

inline float ternary_dot(const uint8_t* w, const float* x, int n) {
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
    
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        for (int j = 0; j < 16; j++) {
            float32x4_t signs = vld1q_f32(BYTE_TO_SIGNS[w[(i/4) + j]]);
            float32x4_t xv = vld1q_f32(x + i + j*4);
            acc0 = vfmaq_f32(acc0, xv, signs);
        }
    }
    
    acc0 = vaddq_f32(acc0, acc1);
    acc0 = vaddq_f32(acc0, acc2);
    acc0 = vaddq_f32(acc0, acc3);
    
    float result = vaddvq_f32(acc0);
    
    for (; i < n; i++) {
        uint8_t trit = (w[i/4] >> ((i%4)*2)) & 0x03;
        if (trit == 1) result += x[i];
        else if (trit == 2) result -= x[i];
    }
    return result;
}
#else
inline float ternary_dot(const uint8_t* w, const float* x, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        uint8_t trit = (w[i/4] >> ((i%4)*2)) & 0x03;
        if (trit == 1) sum += x[i];
        else if (trit == 2) sum -= x[i];
    }
    return sum;
}
#endif

inline float fast_sigmoid(float x) {
    if (x < -4.0f) return 0.0f;
    if (x > 4.0f) return 1.0f;
    return 0.5f + x * (0.25f - 0.03125f * fabsf(x));
}

inline float fast_tanh(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

inline float fast_silu(float x) {
    return x * fast_sigmoid(x);
}

//=============================================================================
// SRC-FFN: Spectral-Rotational-CfC Feed-Forward Network
//=============================================================================

struct SRC_FFN_Layer {
    // Weights (ternary packed)
    std::vector<uint8_t> w_gate;   // [ff, embd] - gate projection
    std::vector<uint8_t> w_up;     // [ff, embd] - up projection  
    std::vector<uint8_t> w_down;   // [embd, ff] - down projection
    
    // Per-FFN-row CfC state (this is the novel part!)
    // Each of the 4096 FFN "neurons" has its own hidden state
    std::vector<float> h_ffn;      // [ff] - hidden state per FFN row
    std::vector<float> decay;      // [ff] - learned decay per row
    std::vector<float> gate_bias;  // [ff] - CfC gate bias
    
    // RMSNorm
    std::vector<float> norm_w;
    
    void init() {
        int bpr_in = (EMBD + 3) / 4;
        int bpr_ff = (FF + 3) / 4;
        
        w_gate.resize(FF * bpr_in);
        w_up.resize(FF * bpr_in);
        w_down.resize(EMBD * bpr_ff);
        
        for (auto& b : w_gate) b = rand() & 0xFF;
        for (auto& b : w_up) b = rand() & 0xFF;
        for (auto& b : w_down) b = rand() & 0xFF;
        
        h_ffn.resize(FF, 0.0f);
        decay.resize(FF);
        gate_bias.resize(FF);
        norm_w.resize(EMBD, 1.0f);
        
        // Initialize decay as learned parameter (0.9-0.99 range typical)
        for (int i = 0; i < FF; i++) {
            decay[i] = 0.9f + 0.09f * (rand() / (float)RAND_MAX);
            gate_bias[i] = -1.0f + 2.0f * (rand() / (float)RAND_MAX);
        }
    }
    
    void reset_state() {
        std::fill(h_ffn.begin(), h_ffn.end(), 0.0f);
    }
};

struct SRC_FFN_Model {
    std::vector<SRC_FFN_Layer> layers;
    std::vector<float> cos_table, sin_table;
    
    // Global hidden state (for inter-layer communication)
    std::vector<float> h_global;
    
    void init() {
        layers.resize(LAYERS);
        for (auto& l : layers) l.init();
        
        h_global.resize(EMBD, 0.0f);
        
        // Precompute RoPE tables
        cos_table.resize(MAX_SEQ * (EMBD/2));
        sin_table.resize(MAX_SEQ * (EMBD/2));
        for (int pos = 0; pos < MAX_SEQ; pos++) {
            for (int i = 0; i < EMBD/2; i++) {
                float freq = 1.0f / powf(ROPE_THETA, (float)(i*2) / EMBD);
                float angle = pos * freq;
                cos_table[pos * (EMBD/2) + i] = cosf(angle);
                sin_table[pos * (EMBD/2) + i] = sinf(angle);
            }
        }
    }
    
    void reset() {
        std::fill(h_global.begin(), h_global.end(), 0.0f);
        for (auto& l : layers) l.reset_state();
    }
    
    // The novel SRC-FFN forward pass
    void src_ffn_forward(SRC_FFN_Layer& layer, const float* x_rot, float* out) {
        int bpr_in = (EMBD + 3) / 4;
        int bpr_ff = (FF + 3) / 4;
        
        // Temporary for FFN intermediate
        alignas(64) float mid[FF];
        
        // For each FFN row, compute:
        //   gate_val = x_rot @ W_gate[i]
        //   up_val = x_rot @ W_up[i]
        //   
        //   // CfC update for this FFN neuron
        //   cfc_gate = sigmoid(up_val + h_ffn[i] + gate_bias[i])
        //   h_ffn[i] = (1 - cfc_gate) * h_ffn[i] * decay[i] + cfc_gate * tanh(up_val)
        //   
        //   mid[i] = SiLU(gate_val) * h_ffn[i]
        //
        // This makes each FFN "neuron" a tiny CfC cell!
        
        for (int i = 0; i < FF; i++) {
            float gate_val = ternary_dot(layer.w_gate.data() + i * bpr_in, x_rot, EMBD);
            float up_val = ternary_dot(layer.w_up.data() + i * bpr_in, x_rot, EMBD);
            
            // CfC dynamics within the FFN neuron
            float cfc_gate_input = up_val + layer.h_ffn[i] * 0.5f + layer.gate_bias[i];
            float cfc_gate = fast_sigmoid(cfc_gate_input);
            float candidate = fast_tanh(up_val);
            
            // Update FFN neuron's hidden state
            layer.h_ffn[i] = (1.0f - cfc_gate) * layer.h_ffn[i] * layer.decay[i] + cfc_gate * candidate;
            
            // Gated output using updated hidden state
            mid[i] = fast_silu(gate_val) * layer.h_ffn[i];
        }
        
        // Down projection
        for (int i = 0; i < EMBD; i++) {
            out[i] = ternary_dot(layer.w_down.data() + i * bpr_ff, mid, FF);
        }
    }
    
    void apply_rope(const float* x, float* out, int pos) {
        int half = EMBD / 2;
        const float* c = cos_table.data() + pos * half;
        const float* s = sin_table.data() + pos * half;
        
        for (int i = 0; i < half; i++) {
            float x0 = x[i * 2], x1 = x[i * 2 + 1];
            out[i * 2] = x0 * c[i] - x1 * s[i];
            out[i * 2 + 1] = x0 * s[i] + x1 * c[i];
        }
    }
    
    void rmsnorm(const float* x, const float* w, float* out) {
        float sum_sq = 0;
        for (int i = 0; i < EMBD; i++) sum_sq += x[i] * x[i];
        float scale = 1.0f / sqrtf(sum_sq / EMBD + 1e-5f);
        for (int i = 0; i < EMBD; i++) out[i] = x[i] * scale * w[i];
    }
    
    void forward(const float* input, float* output, int pos) {
        alignas(64) float x[EMBD], x_rot[EMBD], x_norm[EMBD], ffn_out[EMBD];
        
        memcpy(x, input, EMBD * sizeof(float));
        
        for (int L = 0; L < LAYERS; L++) {
            SRC_FFN_Layer& layer = layers[L];
            
            // 1. RoPE rotation
            apply_rope(x, x_rot, pos);
            
            // 2. RMSNorm
            rmsnorm(x_rot, layer.norm_w.data(), x_norm);
            
            // 3. SRC-FFN (the novel part)
            src_ffn_forward(layer, x_norm, ffn_out);
            
            // 4. Residual
            for (int i = 0; i < EMBD; i++) x[i] += ffn_out[i];
        }
        
        memcpy(output, x, EMBD * sizeof(float));
    }
};

//=============================================================================
// Comparison: Traditional FFN vs SRC-FFN
//=============================================================================

struct Traditional_FFN_Layer {
    std::vector<uint8_t> w_gate, w_up, w_down;
    std::vector<float> norm_w;
    
    void init() {
        int bpr_in = (EMBD + 3) / 4;
        int bpr_ff = (FF + 3) / 4;
        w_gate.resize(FF * bpr_in);
        w_up.resize(FF * bpr_in);
        w_down.resize(EMBD * bpr_ff);
        for (auto& b : w_gate) b = rand() & 0xFF;
        for (auto& b : w_up) b = rand() & 0xFF;
        for (auto& b : w_down) b = rand() & 0xFF;
        norm_w.resize(EMBD, 1.0f);
    }
};

void traditional_ffn_forward(Traditional_FFN_Layer& layer, const float* x, float* out) {
    int bpr_in = (EMBD + 3) / 4;
    int bpr_ff = (FF + 3) / 4;
    
    alignas(64) float mid[FF];
    
    for (int i = 0; i < FF; i++) {
        float gate = ternary_dot(layer.w_gate.data() + i * bpr_in, x, EMBD);
        float up = ternary_dot(layer.w_up.data() + i * bpr_in, x, EMBD);
        mid[i] = fast_silu(gate) * up;
    }
    
    for (int i = 0; i < EMBD; i++) {
        out[i] = ternary_dot(layer.w_down.data() + i * bpr_ff, mid, FF);
    }
}

//=============================================================================
// Benchmark
//=============================================================================

int main() {
    printf("=== Spectral-Rotational-CfC FFN Exploration ===\n\n");
    
    printf("Concept: What if every FFN neuron was a tiny CfC cell?\n");
    printf("  Traditional: mid[i] = SiLU(gate) * up\n");
    printf("  SRC-FFN:     mid[i] = SiLU(gate) * CfC_update(up, h[i])\n\n");
    
    printf("Each of the %d FFN neurons maintains its own hidden state,\n", FF);
    printf("creating a massively parallel recurrent network within the FFN.\n\n");
    
    // Initialize models
    SRC_FFN_Model src_model;
    src_model.init();
    
    Traditional_FFN_Layer trad_layer;
    trad_layer.init();
    
    std::vector<float> input(EMBD, 0.1f);
    std::vector<float> output(EMBD);
    
    // Benchmark traditional FFN (single layer)
    printf("=== Single Layer Comparison ===\n");
    
    int warmup = 100, iters = 1000;
    
    for (int i = 0; i < warmup; i++) {
        traditional_ffn_forward(trad_layer, input.data(), output.data());
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        traditional_ffn_forward(trad_layer, input.data(), output.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    double us_trad = std::chrono::duration<double, std::micro>(end - start).count() / iters;
    
    printf("  Traditional FFN: %.1f us/layer\n", us_trad);
    
    // Benchmark SRC-FFN (single layer)
    src_model.reset();
    
    alignas(64) float x_rot[EMBD], x_norm[EMBD], ffn_out[EMBD];
    src_model.apply_rope(input.data(), x_rot, 0);
    src_model.rmsnorm(x_rot, src_model.layers[0].norm_w.data(), x_norm);
    
    for (int i = 0; i < warmup; i++) {
        src_model.src_ffn_forward(src_model.layers[0], x_norm, ffn_out);
    }
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        src_model.src_ffn_forward(src_model.layers[0], x_norm, ffn_out);
    }
    end = std::chrono::high_resolution_clock::now();
    double us_src = std::chrono::duration<double, std::micro>(end - start).count() / iters;
    
    printf("  SRC-FFN: %.1f us/layer\n", us_src);
    printf("  Overhead: %.1f%%\n", (us_src - us_trad) / us_trad * 100);
    
    // Full model benchmark
    printf("\n=== Full Model (%d layers) ===\n", LAYERS);
    
    src_model.reset();
    
    for (int i = 0; i < warmup/10; i++) {
        src_model.forward(input.data(), output.data(), i);
    }
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters/10; i++) {
        src_model.forward(input.data(), output.data(), warmup/10 + i);
    }
    end = std::chrono::high_resolution_clock::now();
    double ms_full = std::chrono::duration<double, std::milli>(end - start).count() / (iters/10);
    
    printf("  Time per token: %.2f ms\n", ms_full);
    printf("  Tokens/sec: %.1f\n", 1000.0 / ms_full);
    
    // Analyze what we built
    printf("\n=== Analysis ===\n");
    printf("Total FFN hidden states: %d neurons × %d layers = %d\n", 
           FF, LAYERS, FF * LAYERS);
    printf("Memory for FFN states: %.1f KB\n", FF * LAYERS * 4 / 1024.0f);
    printf("\nThis is effectively a %d-wide parallel RNN running\n", FF);
    printf("INSIDE each FFN layer, with learned per-neuron dynamics.\n");
    
    printf("\n=== Theoretical Properties ===\n");
    printf("1. Position-aware: RoPE applied before FFN projection\n");
    printf("2. History-aware: Each FFN neuron has temporal memory\n");
    printf("3. Gated mixing: CfC gate controls feature vs memory balance\n");
    printf("4. Learned dynamics: Per-neuron decay rates\n");
    printf("5. No attention: O(1) per token, not O(n)\n");
    
    printf("\n=== Open Questions ===\n");
    printf("1. Does FFN-level recurrence improve language modeling?\n");
    printf("2. Can we train this end-to-end with backprop?\n");
    printf("3. What's the optimal decay initialization?\n");
    printf("4. Should gate/up/down projections share CfC state?\n");
    
    printf("\n=== Done ===\n");
    return 0;
}
