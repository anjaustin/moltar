// Six-Core Ternary LLM Engine
//
// Strategy: Use ALL available cores on Dimensity 930
// - 2x A78 (big cores, CPU 6-7): CfC cell + FFN down (latency critical)
// - 4x A55 (little cores, CPU 0-3): FFN gate/up (parallel, throughput)
//
// Work distribution for each layer:
// 1. A78 cores: CfC cell (512 outputs each, needs h_prev)
// 2. A55 cores: FFN gate+up (1024 rows each, embarrassingly parallel)
// 3. A78 cores: FFN down (512 outputs each, needs ffn_mid complete)
//
// Expected: ~1.5-1.8x speedup over dual A78 only

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

// Core mapping for Dimensity 930
constexpr int A55_CORES[] = {0, 1, 2, 3};  // Little cores
constexpr int A78_CORES[] = {6, 7};        // Big cores
constexpr int NUM_A55 = 4;
constexpr int NUM_A78 = 2;
constexpr int TOTAL_WORKERS = NUM_A55 + NUM_A78;

//=============================================================================
// NEON Ternary Primitives
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
    
    PREFETCH_W(wp + 64);
    PREFETCH_X(x + 128);
    
    for (; i + 128 <= n; i += 128, wp += 32) {
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
    int i = 0;
    for (; i + 4 <= half; i += 4) {
        float32x4x2_t xy = vld2q_f32(x + i * 2);
        float32x4_t c = vld1q_f32(c_ptr + i);
        float32x4_t s = vld1q_f32(s_ptr + i);
        
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
// Layer Bundle
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
// Six-Core Parallel Execution
//=============================================================================

// Work phases
enum Phase {
    PHASE_IDLE = 0,
    PHASE_CFC,          // A78 cores do CfC
    PHASE_FFN_UP,       // All cores do FFN gate+up
    PHASE_FFN_DOWN,     // A78 cores do FFN down
    PHASE_DONE
};

struct SixCoreLLM {
    std::vector<Layer> layers;
    std::vector<float> cos_table, sin_table;
    std::vector<float> h_states;
    
    int dim, ff, n_layers;
    
    // Synchronization
    std::atomic<int> phase{PHASE_IDLE};
    std::atomic<int> workers_ready{0};
    std::atomic<int> workers_done{0};
    std::atomic<bool> running{false};
    
    // Current layer context
    int cur_layer_idx;
    int cur_pos;
    
    // Shared buffers (cache-line aligned)
    alignas(64) std::vector<float> x_buf;
    alignas(64) std::vector<float> x_rot_buf;
    alignas(64) std::vector<float> x_norm_buf;
    alignas(64) std::vector<float> h_new_buf;
    alignas(64) std::vector<float> ffn_mid_buf;
    alignas(64) std::vector<float> ffn_out_buf;
    
    pthread_t workers[TOTAL_WORKERS];
    
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
        
        x_buf.resize(d); x_rot_buf.resize(d); x_norm_buf.resize(d);
        h_new_buf.resize(d); ffn_mid_buf.resize(ff); ffn_out_buf.resize(d);
    }
    
    // Worker processes assigned work based on worker ID and phase
    void process_work(int worker_id) {
        Layer& layer = layers[cur_layer_idx];
        float* h_prev = h_states.data() + cur_layer_idx * dim;
        
        int bpr_cfc = (dim * 2 + 3) / 4;
        int bh = (dim + 3) / 4;
        int bpr_ffn = (dim + 3) / 4;
        int bpr_down = (ff + 3) / 4;
        
        int current_phase = phase.load(std::memory_order_acquire);
        
        if (current_phase == PHASE_CFC) {
            // Only A78 cores (workers 4,5) do CfC
            if (worker_id >= NUM_A55) {
                int a78_id = worker_id - NUM_A55;  // 0 or 1
                int half_dim = dim / 2;
                int start = a78_id * half_dim;
                int end = start + half_dim;
                
                for (int i = start; i < end; i++) {
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
            }
            // A55 cores idle during CfC (could do prefetch or other prep)
        }
        else if (current_phase == PHASE_FFN_UP) {
            // All 6 cores do FFN gate+up
            // 4096 rows / 6 workers = 682-683 rows each
            int rows_per_worker = ff / TOTAL_WORKERS;
            int extra = ff % TOTAL_WORKERS;
            int start = worker_id * rows_per_worker + (worker_id < extra ? worker_id : extra);
            int count = rows_per_worker + (worker_id < extra ? 1 : 0);
            int end = start + count;
            
            for (int i = start; i < end; i++) {
                float gate = ternary_dot_neon(layer.w_ffn_gate.data() + i*bpr_ffn, x_norm_buf.data(), dim);
                float up = ternary_dot_neon(layer.w_ffn_up.data() + i*bpr_ffn, x_norm_buf.data(), dim);
                ffn_mid_buf[i] = fast_silu(gate) * up;
            }
        }
        else if (current_phase == PHASE_FFN_DOWN) {
            // Only A78 cores do FFN down (latency critical path)
            if (worker_id >= NUM_A55) {
                int a78_id = worker_id - NUM_A55;
                int half_dim = dim / 2;
                int start = a78_id * half_dim;
                int end = start + half_dim;
                
                for (int i = start; i < end; i++) {
                    ffn_out_buf[i] = ternary_dot_neon(layer.w_ffn_down.data() + i*bpr_down, ffn_mid_buf.data(), ff);
                }
            }
            // A55 cores idle during FFN down
        }
    }
    
    static void* worker_fn(void* arg) {
        auto* ctx = (std::pair<SixCoreLLM*, int>*)arg;
        SixCoreLLM* llm = ctx->first;
        int worker_id = ctx->second;
        delete ctx;
        
        // Pin to appropriate core
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (worker_id < NUM_A55) {
            CPU_SET(A55_CORES[worker_id], &cpuset);
        } else {
            CPU_SET(A78_CORES[worker_id - NUM_A55], &cpuset);
        }
        sched_setaffinity(0, sizeof(cpuset), &cpuset);
        
        int last_seen_phase = PHASE_IDLE;
        
        while (llm->running) {
            // Wait for new phase
            int current_phase;
            while (llm->running) {
                current_phase = llm->phase.load(std::memory_order_acquire);
                if (current_phase != last_seen_phase && current_phase != PHASE_IDLE && current_phase != PHASE_DONE) {
                    break;
                }
                #ifdef __aarch64__
                asm volatile("yield");
                #endif
            }
            if (!llm->running) break;
            
            last_seen_phase = current_phase;
            
            // Do work
            llm->process_work(worker_id);
            
            // Signal done
            llm->workers_done.fetch_add(1, std::memory_order_release);
        }
        
        return nullptr;
    }
    
    void start_workers() {
        running = true;
        phase = PHASE_IDLE;
        workers_done = 0;
        
        for (int i = 0; i < TOTAL_WORKERS; i++) {
            auto* ctx = new std::pair<SixCoreLLM*, int>(this, i);
            pthread_create(&workers[i], nullptr, worker_fn, ctx);
        }
        usleep(10000);  // Let workers spin up
    }
    
    void stop_workers() {
        running = false;
        phase = PHASE_DONE;
        for (int i = 0; i < TOTAL_WORKERS; i++) {
            pthread_join(workers[i], nullptr);
        }
    }
    
    void wait_workers(int expected_count) {
        while (workers_done.load(std::memory_order_acquire) < expected_count) {
            #ifdef __aarch64__
            asm volatile("yield");
            #endif
        }
    }
    
    void forward(const float* input, float* output, int pos) {
        memcpy(x_buf.data(), input, dim * sizeof(float));
        cur_pos = pos;
        
        for (int L = 0; L < n_layers; L++) {
            Layer& layer = layers[L];
            float* h_prev = h_states.data() + L * dim;
            cur_layer_idx = L;
            
            // RoPE (main thread)
            apply_rope(x_buf.data(), x_rot_buf.data(), cos_table.data(), sin_table.data(), pos, dim);
            
            // Phase 1: CfC (A78 cores only = 2 workers)
            workers_done = 0;
            phase.store(PHASE_CFC, std::memory_order_release);
            wait_workers(NUM_A78);  // Only wait for A78 cores
            
            // Update h_states and compute residual + norm (main thread)
            memcpy(h_prev, h_new_buf.data(), dim * sizeof(float));
            
            #ifdef __ARM_NEON
            for (int i = 0; i < dim; i += 4) {
                float32x4_t x = vld1q_f32(x_buf.data() + i);
                float32x4_t h = vld1q_f32(h_new_buf.data() + i);
                vst1q_f32(x_buf.data() + i, vaddq_f32(x, h));
            }
            #else
            for (int i = 0; i < dim; i++) x_buf[i] += h_new_buf[i];
            #endif
            
            rmsnorm(x_buf.data(), layer.norm1_w.data(), x_norm_buf.data(), dim);
            
            // Phase 2: FFN up (all 6 cores)
            workers_done = 0;
            phase.store(PHASE_FFN_UP, std::memory_order_release);
            wait_workers(TOTAL_WORKERS);
            
            // Phase 3: FFN down (A78 cores only)
            workers_done = 0;
            phase.store(PHASE_FFN_DOWN, std::memory_order_release);
            wait_workers(NUM_A78);
            
            // Final residual (main thread)
            #ifdef __ARM_NEON
            for (int i = 0; i < dim; i += 4) {
                float32x4_t x = vld1q_f32(x_buf.data() + i);
                float32x4_t f = vld1q_f32(ffn_out_buf.data() + i);
                vst1q_f32(x_buf.data() + i, vaddq_f32(x, f));
            }
            #else
            for (int i = 0; i < dim; i++) x_buf[i] += ffn_out_buf[i];
            #endif
            
            // Reset to idle before next layer
            phase.store(PHASE_IDLE, std::memory_order_release);
        }
        
        memcpy(output, x_buf.data(), dim * sizeof(float));
    }
    
    // Reference single-threaded implementation
    void forward_single(const float* input, float* output, int pos) {
        std::vector<float> x(dim), x_rot(dim), x_norm(dim);
        std::vector<float> h_new(dim), ffn_mid(ff), ffn_out(dim);
        
        memcpy(x.data(), input, dim * sizeof(float));
        
        for (int L = 0; L < n_layers; L++) {
            Layer& layer = layers[L];
            float* h_prev = h_states.data() + L * dim;
            
            int bpr_cfc = (dim * 2 + 3) / 4;
            int bh = (dim + 3) / 4;
            int bpr_ffn = (dim + 3) / 4;
            int bpr_down = (ff + 3) / 4;
            
            apply_rope(x.data(), x_rot.data(), cos_table.data(), sin_table.data(), pos, dim);
            
            // CfC
            for (int i = 0; i < dim; i++) {
                float gp = ternary_dot_neon(layer.w_cfc_gate.data() + i*bpr_cfc, x_rot.data(), dim) +
                           ternary_dot_neon(layer.w_cfc_gate.data() + i*bpr_cfc + bh, h_prev, dim) +
                           layer.b_cfc_gate[i];
                float cp = ternary_dot_neon(layer.w_cfc_cand.data() + i*bpr_cfc, x_rot.data(), dim) +
                           ternary_dot_neon(layer.w_cfc_cand.data() + i*bpr_cfc + bh, h_prev, dim) +
                           layer.b_cfc_cand[i];
                
                float g = fast_sigmoid(gp);
                float c = fast_tanh(cp);
                h_new[i] = (1.0f - g) * h_prev[i] * layer.decay[i] + g * c;
            }
            memcpy(h_prev, h_new.data(), dim * sizeof(float));
            
            for (int i = 0; i < dim; i++) x[i] += h_new[i];
            rmsnorm(x.data(), layer.norm1_w.data(), x_norm.data(), dim);
            
            // FFN
            for (int i = 0; i < ff; i++) {
                float gate = ternary_dot_neon(layer.w_ffn_gate.data() + i*bpr_ffn, x_norm.data(), dim);
                float up = ternary_dot_neon(layer.w_ffn_up.data() + i*bpr_ffn, x_norm.data(), dim);
                ffn_mid[i] = fast_silu(gate) * up;
            }
            
            for (int i = 0; i < dim; i++) {
                ffn_out[i] = ternary_dot_neon(layer.w_ffn_down.data() + i*bpr_down, ffn_mid.data(), ff);
            }
            
            for (int i = 0; i < dim; i++) x[i] += ffn_out[i];
        }
        
        memcpy(output, x.data(), dim * sizeof(float));
    }
};

//=============================================================================
// Benchmark
//=============================================================================

int main() {
    printf("=== Six-Core Ternary LLM Engine ===\n");
    printf("Strategy: 2x A78 + 4x A55 parallel execution\n");
    printf("  - A78 cores: CfC cell + FFN down (latency critical)\n");
    printf("  - A55 cores: FFN gate+up (throughput parallel)\n\n");
    
    SixCoreLLM llm;
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
    
    // Benchmark six-core
    printf("\n=== Six-Core Parallel ===\n");
    llm.start_workers();
    
    for (int i = 0; i < warmup; i++) {
        llm.forward(input.data(), output.data(), i);
    }
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        llm.forward(input.data(), output.data(), warmup + i);
    }
    end = std::chrono::high_resolution_clock::now();
    double ms_six = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    
    llm.stop_workers();
    
    printf("  Time per token: %.2f ms\n", ms_six);
    printf("  Tokens/sec: %.1f\n", 1000.0 / ms_six);
    printf("  Speedup vs single: %.2fx\n", ms_single / ms_six);
    
    printf("\n=== COMPARISON ===\n");
    printf("  CPU baseline (llama.cpp Q4): 50 tok/s\n");
    printf("  Single-core ternary: %.1f tok/s\n", 1000.0 / ms_single);
    printf("  Six-core ternary: %.1f tok/s\n", 1000.0 / ms_six);
    
    if (1000.0 / ms_six > 50) {
        printf("\n  *** BEATS CPU BASELINE! ***\n");
    } else {
        printf("\n  Gap to target: %.1f%%\n", (50.0 - 1000.0/ms_six) / 50.0 * 100);
        printf("  Need: %.1fx more speedup\n", 50.0 / (1000.0 / ms_six));
    }
    
    printf("\n=== Done ===\n");
    return 0;
}
