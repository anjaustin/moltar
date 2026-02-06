// Dual-Core Ternary LLM Engine
//
// Strategy: PARALLEL execution on two A78 cores
// Both cores work on the SAME layer simultaneously:
// - Split FFN rows (4096 rows -> 2048 per core)
// - Split CfC outputs (1024 -> 512 per core)
// 
// This gives true 2x parallelism with minimal sync overhead.

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
// Model Configuration
//=============================================================================

constexpr int LFM2_EMBD = 1024;
constexpr int LFM2_FF = 4096;
constexpr int LFM2_LAYERS = 16;
constexpr int MAX_SEQ_LEN = 512;
constexpr float ROPE_THETA = 10000.0f;

//=============================================================================
// NEON Ternary Primitives (same as before)
//=============================================================================

#ifdef __ARM_NEON

alignas(64) static float BYTE_TO_SIGNS[256][4];

__attribute__((constructor))
static void init_trit_lut() {
    for (int b = 0; b < 256; b++) {
        uint8_t t0 = b & 0x03;
        uint8_t t1 = (b >> 2) & 0x03;
        uint8_t t2 = (b >> 4) & 0x03;
        uint8_t t3 = (b >> 6) & 0x03;
        
        BYTE_TO_SIGNS[b][0] = (t0 == 1) ? 1.0f : ((t0 == 2) ? -1.0f : 0.0f);
        BYTE_TO_SIGNS[b][1] = (t1 == 1) ? 1.0f : ((t1 == 2) ? -1.0f : 0.0f);
        BYTE_TO_SIGNS[b][2] = (t2 == 1) ? 1.0f : ((t2 == 2) ? -1.0f : 0.0f);
        BYTE_TO_SIGNS[b][3] = (t3 == 1) ? 1.0f : ((t3 == 2) ? -1.0f : 0.0f);
    }
}

inline void process_byte_lut(uint8_t byte, float32x4_t x, float32x4_t* acc) {
    float32x4_t signs = vld1q_f32(BYTE_TO_SIGNS[byte]);
    *acc = vfmaq_f32(*acc, x, signs);
}

// Prefetch hint for upcoming weight reads
#define PREFETCH_W(ptr) __builtin_prefetch((ptr), 0, 3)
#define PREFETCH_X(ptr) __builtin_prefetch((ptr), 0, 3)

inline float ternary_dot_neon(const uint8_t* w_packed, const float* x, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    float32x4_t acc4 = vdupq_n_f32(0.0f);
    float32x4_t acc5 = vdupq_n_f32(0.0f);
    float32x4_t acc6 = vdupq_n_f32(0.0f);
    float32x4_t acc7 = vdupq_n_f32(0.0f);
    
    int i = 0;
    const uint8_t* wp = w_packed;
    
    // Prefetch first block
    PREFETCH_W(wp + 64);
    PREFETCH_X(x + 128);
    
    for (; i + 128 <= n; i += 128, wp += 32) {
        // Prefetch next iteration
        PREFETCH_W(wp + 64);
        PREFETCH_X(x + i + 256);
        
        process_byte_lut(wp[0], vld1q_f32(x + i + 0), &acc0);
        process_byte_lut(wp[1], vld1q_f32(x + i + 4), &acc1);
        process_byte_lut(wp[2], vld1q_f32(x + i + 8), &acc2);
        process_byte_lut(wp[3], vld1q_f32(x + i + 12), &acc3);
        process_byte_lut(wp[4], vld1q_f32(x + i + 16), &acc4);
        process_byte_lut(wp[5], vld1q_f32(x + i + 20), &acc5);
        process_byte_lut(wp[6], vld1q_f32(x + i + 24), &acc6);
        process_byte_lut(wp[7], vld1q_f32(x + i + 28), &acc7);
        process_byte_lut(wp[8], vld1q_f32(x + i + 32), &acc0);
        process_byte_lut(wp[9], vld1q_f32(x + i + 36), &acc1);
        process_byte_lut(wp[10], vld1q_f32(x + i + 40), &acc2);
        process_byte_lut(wp[11], vld1q_f32(x + i + 44), &acc3);
        process_byte_lut(wp[12], vld1q_f32(x + i + 48), &acc4);
        process_byte_lut(wp[13], vld1q_f32(x + i + 52), &acc5);
        process_byte_lut(wp[14], vld1q_f32(x + i + 56), &acc6);
        process_byte_lut(wp[15], vld1q_f32(x + i + 60), &acc7);
        process_byte_lut(wp[16], vld1q_f32(x + i + 64), &acc0);
        process_byte_lut(wp[17], vld1q_f32(x + i + 68), &acc1);
        process_byte_lut(wp[18], vld1q_f32(x + i + 72), &acc2);
        process_byte_lut(wp[19], vld1q_f32(x + i + 76), &acc3);
        process_byte_lut(wp[20], vld1q_f32(x + i + 80), &acc4);
        process_byte_lut(wp[21], vld1q_f32(x + i + 84), &acc5);
        process_byte_lut(wp[22], vld1q_f32(x + i + 88), &acc6);
        process_byte_lut(wp[23], vld1q_f32(x + i + 92), &acc7);
        process_byte_lut(wp[24], vld1q_f32(x + i + 96), &acc0);
        process_byte_lut(wp[25], vld1q_f32(x + i + 100), &acc1);
        process_byte_lut(wp[26], vld1q_f32(x + i + 104), &acc2);
        process_byte_lut(wp[27], vld1q_f32(x + i + 108), &acc3);
        process_byte_lut(wp[28], vld1q_f32(x + i + 112), &acc4);
        process_byte_lut(wp[29], vld1q_f32(x + i + 116), &acc5);
        process_byte_lut(wp[30], vld1q_f32(x + i + 120), &acc6);
        process_byte_lut(wp[31], vld1q_f32(x + i + 124), &acc7);
    }
    
    acc0 = vaddq_f32(acc0, acc4);
    acc1 = vaddq_f32(acc1, acc5);
    acc2 = vaddq_f32(acc2, acc6);
    acc3 = vaddq_f32(acc3, acc7);
    acc0 = vaddq_f32(acc0, acc2);
    acc1 = vaddq_f32(acc1, acc3);
    acc0 = vaddq_f32(acc0, acc1);
    
    for (; i + 16 <= n; i += 16, wp += 4) {
        process_byte_lut(wp[0], vld1q_f32(x + i + 0), &acc0);
        process_byte_lut(wp[1], vld1q_f32(x + i + 4), &acc0);
        process_byte_lut(wp[2], vld1q_f32(x + i + 8), &acc0);
        process_byte_lut(wp[3], vld1q_f32(x + i + 12), &acc0);
    }
    
    float32x2_t sum2 = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
    float result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    
    for (; i < n; i++) {
        int byte_idx = i / 4;
        int bit_idx = (i % 4) * 2;
        uint8_t trit = (w_packed[byte_idx] >> bit_idx) & 0x03;
        float sign = (trit == 1) ? 1.0f : ((trit == 2) ? -1.0f : 0.0f);
        result += x[i] * sign;
    }
    
    return result;
}

#else
inline float ternary_dot_neon(const uint8_t* w, const float* x, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        uint8_t trit = (w[i/4] >> ((i%4)*2)) & 0x03;
        if (trit == 1) sum += x[i];
        else if (trit == 2) sum -= x[i];
    }
    return sum;
}
#endif

//=============================================================================
// Layer Operations
//=============================================================================

void apply_rope(const float* x, float* out, const float* cos_t, const float* sin_t, 
                int pos, int dim) {
    int half = dim / 2;
    const float* c_ptr = cos_t + pos * half;
    const float* s_ptr = sin_t + pos * half;
    
    #ifdef __ARM_NEON
    // Process 4 pairs (8 elements) at a time
    int i = 0;
    for (; i + 4 <= half; i += 4) {
        // Load 4 pairs: (x0,x1), (x2,x3), (x4,x5), (x6,x7)
        float32x4x2_t xy = vld2q_f32(x + i * 2);  // xy.val[0] = evens, xy.val[1] = odds
        float32x4_t c = vld1q_f32(c_ptr + i);
        float32x4_t s = vld1q_f32(s_ptr + i);
        
        // rotated_even = x0*c - x1*s
        // rotated_odd  = x0*s + x1*c
        float32x4_t re = vfmsq_f32(vmulq_f32(xy.val[0], c), xy.val[1], s);
        float32x4_t ro = vfmaq_f32(vmulq_f32(xy.val[0], s), xy.val[1], c);
        
        float32x4x2_t result = {re, ro};
        vst2q_f32(out + i * 2, result);
    }
    
    for (; i < half; i++) {
        float x0 = x[i * 2], x1 = x[i * 2 + 1];
        float c = c_ptr[i], s = s_ptr[i];
        out[i * 2] = x0 * c - x1 * s;
        out[i * 2 + 1] = x0 * s + x1 * c;
    }
    #else
    for (int i = 0; i < half; i++) {
        float x0 = x[i * 2], x1 = x[i * 2 + 1];
        float c = c_ptr[i], s = s_ptr[i];
        out[i * 2] = x0 * c - x1 * s;
        out[i * 2 + 1] = x0 * s + x1 * c;
    }
    #endif
}

void rmsnorm(const float* x, const float* w, float* out, int dim) {
    #ifdef __ARM_NEON
    float32x4_t sum_sq_v = vdupq_n_f32(0);
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        sum_sq_v = vfmaq_f32(sum_sq_v, v, v);
    }
    float sum_sq = vaddvq_f32(sum_sq_v);
    for (; i < dim; i++) sum_sq += x[i] * x[i];
    
    float scale = 1.0f / sqrtf(sum_sq / dim + 1e-5f);
    float32x4_t scale_v = vdupq_n_f32(scale);
    
    for (i = 0; i + 4 <= dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, scale_v), wv));
    }
    for (; i < dim; i++) out[i] = x[i] * scale * w[i];
    #else
    float sum_sq = 0;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum_sq / dim + 1e-5f);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * w[i];
    #endif
}

// Fast sigmoid approximation: 1/(1+exp(-x)) ≈ 0.5 + 0.5*tanh(x/2)
// Even faster: piecewise linear
inline float fast_sigmoid(float x) {
    if (x < -4.0f) return 0.0f;
    if (x > 4.0f) return 1.0f;
    return 0.5f + x * (0.25f - 0.03125f * fabsf(x));  // Piecewise quadratic
}

// Fast tanh approximation
inline float fast_tanh(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);  // Padé approximant
}

void cfc_cell(const uint8_t* Wg, const uint8_t* Wc, const float* bg, const float* bc,
              const float* decay, const float* x_rot, const float* h_prev,
              float* h_new, int dim) {
    int bpr = (dim * 2 + 3) / 4;
    int bh = (dim + 3) / 4;
    
    for (int i = 0; i < dim; i++) {
        float gp = ternary_dot_neon(Wg + i*bpr, x_rot, dim) +
                   ternary_dot_neon(Wg + i*bpr + bh, h_prev, dim) + bg[i];
        float cp = ternary_dot_neon(Wc + i*bpr, x_rot, dim) +
                   ternary_dot_neon(Wc + i*bpr + bh, h_prev, dim) + bc[i];
        
        float g = fast_sigmoid(gp);
        float c = fast_tanh(cp);
        h_new[i] = (1.0f - g) * h_prev[i] * decay[i] + g * c;
    }
}

// Fast SiLU: x * sigmoid(x)
inline float fast_silu(float x) {
    return x * fast_sigmoid(x);
}

void ffn_fused(const uint8_t* Wgate, const uint8_t* Wup, const float* x,
               float* out, int ff, int dim) {
    int bpr = (dim + 3) / 4;
    for (int i = 0; i < ff; i++) {
        float gate = ternary_dot_neon(Wgate + i*bpr, x, dim);
        float up = ternary_dot_neon(Wup + i*bpr, x, dim);
        out[i] = fast_silu(gate) * up;
    }
}

void ffn_down(const uint8_t* Wdown, const float* x, float* out, int dim, int ff) {
    int bpr = (ff + 3) / 4;
    for (int i = 0; i < dim; i++) {
        out[i] = ternary_dot_neon(Wdown + i*bpr, x, ff);
    }
}

//=============================================================================
// Layer Bundle (weights for one transformer layer)
//=============================================================================

struct Layer {
    std::vector<uint8_t> w_cfc_gate, w_cfc_cand;
    std::vector<float> b_cfc_gate, b_cfc_cand, decay, norm1_w;
    std::vector<uint8_t> w_ffn_gate, w_ffn_up, w_ffn_down;
    std::vector<float> norm2_w;
    
    void init(int dim, int ff) {
        auto gen = [](std::vector<uint8_t>& v, int rows, int cols) {
            int bpr = (cols + 3) / 4;
            v.resize(rows * bpr);
            for (auto& b : v) b = rand() & 0xFF;
        };
        
        gen(w_cfc_gate, dim, dim * 2);
        gen(w_cfc_cand, dim, dim * 2);
        b_cfc_gate.resize(dim, 0); b_cfc_cand.resize(dim, 0);
        decay.resize(dim, 0.95f); norm1_w.resize(dim, 1.0f);
        
        gen(w_ffn_gate, ff, dim);
        gen(w_ffn_up, ff, dim);
        gen(w_ffn_down, dim, ff);
        norm2_w.resize(dim, 1.0f);
    }
};

//=============================================================================
// Dual-Core PARALLEL Execution - Simplified Sync
//=============================================================================

struct DualCoreLLM {
    std::vector<Layer> layers;
    std::vector<float> cos_table, sin_table;
    std::vector<float> h_states;
    
    int dim, ff, n_layers;
    
    // Simple spin-wait synchronization
    std::atomic<int> worker_done[2];
    std::atomic<int> work_ready{0};
    std::atomic<bool> running{false};
    
    // Current layer context
    int cur_layer_idx;
    int cur_pos;
    
    // Pre-allocated per-layer buffers (shared)
    std::vector<float> x_buf, x_rot_buf, x_norm_buf;
    std::vector<float> h_new_buf, ffn_mid_buf, ffn_out_buf;
    
    pthread_t workers[2];
    
    void init(int d, int f, int n) {
        dim = d; ff = f; n_layers = n;
        
        layers.resize(n);
        for (int i = 0; i < n; i++) layers[i].init(d, f);
        
        cos_table.resize(MAX_SEQ_LEN * (d/2));
        sin_table.resize(MAX_SEQ_LEN * (d/2));
        for (int pos = 0; pos < MAX_SEQ_LEN; pos++) {
            for (int i = 0; i < d/2; i++) {
                float freq = 1.0f / powf(ROPE_THETA, (float)(i*2) / d);
                float angle = pos * freq;
                cos_table[pos * (d/2) + i] = cosf(angle);
                sin_table[pos * (d/2) + i] = sinf(angle);
            }
        }
        
        h_states.resize(n * d, 0);
        
        // Pre-allocate buffers
        x_buf.resize(d); x_rot_buf.resize(d); x_norm_buf.resize(d);
        h_new_buf.resize(d); ffn_mid_buf.resize(ff); ffn_out_buf.resize(d);
        
        worker_done[0] = 0;
        worker_done[1] = 0;
    }
    
    // Worker processes ONE layer for its half
    void process_layer_half(int tid, int L, int pos) {
        Layer& layer = layers[L];
        float* h_prev = h_states.data() + L * dim;
        
        int half_dim = dim / 2;
        int half_ff = ff / 2;
        int dim_start = tid * half_dim;
        int dim_end = dim_start + half_dim;
        int ff_start = tid * half_ff;
        int ff_end = ff_start + half_ff;
        
        int bpr_cfc = (dim * 2 + 3) / 4;
        int bh = (dim + 3) / 4;
        int bpr_ffn = (dim + 3) / 4;
        int bpr_down = (ff + 3) / 4;
        
        // CfC cell - half the outputs
        for (int i = dim_start; i < dim_end; i++) {
            float gp = ternary_dot_neon(layer.w_cfc_gate.data() + i*bpr_cfc, x_rot_buf.data(), dim) +
                       ternary_dot_neon(layer.w_cfc_gate.data() + i*bpr_cfc + bh, h_prev, dim) +
                       layer.b_cfc_gate[i];
            float cp = ternary_dot_neon(layer.w_cfc_cand.data() + i*bpr_cfc, x_rot_buf.data(), dim) +
                       ternary_dot_neon(layer.w_cfc_cand.data() + i*bpr_cfc + bh, h_prev, dim) +
                       layer.b_cfc_cand[i];
            
            float g = fast_sigmoid(gp);
            float c = fast_tanh(cp);
            h_new_buf[i] = (1.0f - g) * h_prev[i] * layer.decay[i] + g * c;
        }
        
        // Barrier 1: wait for other half of CfC
        worker_done[tid].store(1, std::memory_order_release);
        while (worker_done[1-tid].load(std::memory_order_acquire) < 1) {
            #ifdef __aarch64__
            asm volatile("yield");
            #endif
        }
        
        // FFN up - half the outputs
        for (int i = ff_start; i < ff_end; i++) {
            float gate = ternary_dot_neon(layer.w_ffn_gate.data() + i*bpr_ffn, x_norm_buf.data(), dim);
            float up = ternary_dot_neon(layer.w_ffn_up.data() + i*bpr_ffn, x_norm_buf.data(), dim);
            ffn_mid_buf[i] = fast_silu(gate) * up;
        }
        
        // Barrier 2: wait for other half of FFN up
        worker_done[tid].store(2, std::memory_order_release);
        while (worker_done[1-tid].load(std::memory_order_acquire) < 2) {
            #ifdef __aarch64__
            asm volatile("yield");
            #endif
        }
        
        // FFN down - half the outputs
        for (int i = dim_start; i < dim_end; i++) {
            ffn_out_buf[i] = ternary_dot_neon(layer.w_ffn_down.data() + i*bpr_down, ffn_mid_buf.data(), ff);
        }
        
        // Signal done
        worker_done[tid].store(3, std::memory_order_release);
    }
    
    static void* worker_fn(void* arg) {
        auto* ctx = (std::pair<DualCoreLLM*, int>*)arg;
        DualCoreLLM* llm = ctx->first;
        int tid = ctx->second;
        delete ctx;
        
        // Pin to big cores (6 and 7)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(6 + tid, &cpuset);
        sched_setaffinity(0, sizeof(cpuset), &cpuset);
        
        int expected_work = 0;
        
        while (llm->running) {
            // Wait for new work
            while (llm->running && llm->work_ready.load(std::memory_order_acquire) <= expected_work) {
                #ifdef __aarch64__
                asm volatile("yield");
                #endif
            }
            if (!llm->running) break;
            
            expected_work = llm->work_ready.load(std::memory_order_acquire);
            llm->worker_done[tid].store(0, std::memory_order_relaxed);
            
            // Process the layer
            llm->process_layer_half(tid, llm->cur_layer_idx, llm->cur_pos);
        }
        
        return nullptr;
    }
    
    void start_workers() {
        running = true;
        work_ready = 0;
        for (int i = 0; i < 2; i++) {
            worker_done[i] = 0;
            auto* ctx = new std::pair<DualCoreLLM*, int>(this, i);
            pthread_create(&workers[i], nullptr, worker_fn, ctx);
        }
        usleep(10000);
    }
    
    void stop_workers() {
        running = false;
        work_ready.fetch_add(1);  // Unblock
        for (int i = 0; i < 2; i++) {
            pthread_join(workers[i], nullptr);
        }
    }
    
    void forward(const float* input, float* output, int pos) {
        memcpy(x_buf.data(), input, dim * sizeof(float));
        cur_pos = pos;
        
        for (int L = 0; L < n_layers; L++) {
            Layer& layer = layers[L];
            float* h_prev = h_states.data() + L * dim;
            
            // RoPE (main thread - fast, ~0.3us with NEON)
            apply_rope(x_buf.data(), x_rot_buf.data(), cos_table.data(), sin_table.data(), pos, dim);
            
            // Kick off parallel work
            cur_layer_idx = L;
            worker_done[0] = 0;
            worker_done[1] = 0;
            work_ready.fetch_add(1, std::memory_order_release);
            
            // Wait for CfC to complete (workers at stage 1)
            while (worker_done[0].load(std::memory_order_acquire) < 1 ||
                   worker_done[1].load(std::memory_order_acquire) < 1) {
                #ifdef __aarch64__
                asm volatile("yield");
                #endif
            }
            
            // Main thread does residual + norm while workers do FFN up
            // This overlaps main thread work with worker compute
            memcpy(h_prev, h_new_buf.data(), dim * sizeof(float));
            
            #ifdef __ARM_NEON
            // Fused residual add using NEON
            for (int i = 0; i < dim; i += 4) {
                float32x4_t x = vld1q_f32(x_buf.data() + i);
                float32x4_t h = vld1q_f32(h_new_buf.data() + i);
                vst1q_f32(x_buf.data() + i, vaddq_f32(x, h));
            }
            #else
            for (int i = 0; i < dim; i++) x_buf[i] += h_new_buf[i];
            #endif
            
            rmsnorm(x_buf.data(), layer.norm1_w.data(), x_norm_buf.data(), dim);
            
            // Wait for all work complete (workers at stage 3)
            while (worker_done[0].load(std::memory_order_acquire) < 3 ||
                   worker_done[1].load(std::memory_order_acquire) < 3) {
                #ifdef __aarch64__
                asm volatile("yield");
                #endif
            }
            
            // Final residual with NEON
            #ifdef __ARM_NEON
            for (int i = 0; i < dim; i += 4) {
                float32x4_t x = vld1q_f32(x_buf.data() + i);
                float32x4_t f = vld1q_f32(ffn_out_buf.data() + i);
                vst1q_f32(x_buf.data() + i, vaddq_f32(x, f));
            }
            #else
            for (int i = 0; i < dim; i++) x_buf[i] += ffn_out_buf[i];
            #endif
        }
        
        memcpy(output, x_buf.data(), dim * sizeof(float));
    }
    
    void forward_single(const float* input, float* output, int pos) {
        std::vector<float> x(dim), x_rot(dim), x_norm(dim);
        std::vector<float> h_new(dim), ffn_mid(ff), ffn_out(dim);
        
        memcpy(x.data(), input, dim * sizeof(float));
        
        for (int L = 0; L < n_layers; L++) {
            Layer& layer = layers[L];
            float* h_prev = h_states.data() + L * dim;
            
            apply_rope(x.data(), x_rot.data(), cos_table.data(), sin_table.data(), pos, dim);
            cfc_cell(layer.w_cfc_gate.data(), layer.w_cfc_cand.data(),
                    layer.b_cfc_gate.data(), layer.b_cfc_cand.data(),
                    layer.decay.data(), x_rot.data(), h_prev, h_new.data(), dim);
            memcpy(h_prev, h_new.data(), dim * sizeof(float));
            
            for (int i = 0; i < dim; i++) x[i] += h_new[i];
            rmsnorm(x.data(), layer.norm1_w.data(), x_norm.data(), dim);
            
            ffn_fused(layer.w_ffn_gate.data(), layer.w_ffn_up.data(),
                     x_norm.data(), ffn_mid.data(), ff, dim);
            ffn_down(layer.w_ffn_down.data(), ffn_mid.data(), ffn_out.data(), dim, ff);
            
            for (int i = 0; i < dim; i++) x[i] += ffn_out[i];
        }
        
        memcpy(output, x.data(), dim * sizeof(float));
    }
};

//=============================================================================
// Benchmark
//=============================================================================

int main() {
    printf("=== Dual-Core Ternary LLM Engine ===\n");
    printf("Strategy: Pipeline layers across 2 A78 cores\n\n");
    
    DualCoreLLM llm;
    printf("Initializing LFM2-350M model (%d embd, %d ff, %d layers)...\n",
           LFM2_EMBD, LFM2_FF, LFM2_LAYERS);
    llm.init(LFM2_EMBD, LFM2_FF, LFM2_LAYERS);
    
    std::vector<float> input(LFM2_EMBD, 0.1f);
    std::vector<float> output(LFM2_EMBD);
    
    // Benchmark single-threaded first
    printf("\n=== Single-Core Baseline ===\n");
    int warmup = 3, iters = 10;
    
    for (int i = 0; i < warmup; i++) {
        llm.forward_single(input.data(), output.data(), i);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        llm.forward_single(input.data(), output.data(), warmup + i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms_single = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    
    printf("  Time per token: %.2f ms\n", ms_single);
    printf("  Tokens/sec: %.1f\n", 1000.0 / ms_single);
    
    // Benchmark dual-core
    printf("\n=== Dual-Core Parallel ===\n");
    llm.start_workers();
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        llm.forward(input.data(), output.data(), i);
    }
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        llm.forward(input.data(), output.data(), warmup + i);
    }
    end = std::chrono::high_resolution_clock::now();
    double ms_dual = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    
    llm.stop_workers();
    
    printf("  Time per token: %.2f ms\n", ms_dual);
    printf("  Tokens/sec: %.1f\n", 1000.0 / ms_dual);
    printf("  Speedup: %.2fx\n", ms_single / ms_dual);
    
    printf("\n=== COMPARISON ===\n");
    printf("  CPU baseline (llama.cpp Q4): 50 tok/s\n");
    printf("  Single-core: %.1f tok/s\n", 1000.0 / ms_single);
    printf("  Dual-core: %.1f tok/s\n", 1000.0 / ms_dual);
    
    if (1000.0 / ms_dual > 50) {
        printf("\n  *** BEATS CPU BASELINE! ***\n");
    } else {
        printf("\n  Need %.1fx more speedup\n", 50.0 / (1000.0 / ms_dual));
    }
    
    printf("\n=== Done ===\n");
    return 0;
}
