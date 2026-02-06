// SRC-FFN Production Inference Engine
//
// Spectral-Rotational-CfC Feed-Forward Network
// Optimized for ARM NEON on Dimensity 930 (2x A78 + 4x A55)
//
// Features:
// - Multi-scale CfC initialization for proper gradient horizons
// - 6-core parallel execution
// - Ternary weights (2-bit)
// - O(1) memory per token
//
// Reference: PRD-001, PRD-002

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <atomic>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

//=============================================================================
// Configuration (LFM2-350M scale)
//=============================================================================

constexpr int EMBED_DIM = 1024;
constexpr int FF_DIM = 4096;
constexpr int N_LAYERS = 16;
constexpr int MAX_SEQ_LEN = 8192;
constexpr float ROPE_THETA = 10000.0f;

// Multi-scale CfC initialization
// Quarter of neurons at each temporal scale
constexpr float DECAY_SCALES[4] = {0.9f, 0.95f, 0.99f, 0.995f};
constexpr float GATE_BIAS_SCALES[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
constexpr float CFC_ALPHA = 0.5f;

// Core mapping
constexpr int A55_CORES[] = {0, 1, 2, 3};
constexpr int A78_CORES[] = {6, 7};
constexpr int NUM_A55 = 4;
constexpr int NUM_A78 = 2;
constexpr int TOTAL_WORKERS = 6;

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

float ternary_dot(const uint8_t* w_packed, const float* x, int n) {
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
float ternary_dot(const uint8_t* w, const float* x, int n) {
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
// Fast Activation Functions
//=============================================================================

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
// NEON Helper Operations
//=============================================================================

#ifdef __ARM_NEON
void rmsnorm(const float* x, const float* w, float* out, int dim) {
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
}

void apply_rope(const float* x, float* out, const float* cos_t, const float* sin_t, 
                int pos, int dim) {
    int half = dim / 2;
    const float* c_ptr = cos_t + pos * half;
    const float* s_ptr = sin_t + pos * half;
    
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
        out[i * 2] = x0 * c_ptr[i] - x1 * s_ptr[i];
        out[i * 2 + 1] = x0 * s_ptr[i] + x1 * c_ptr[i];
    }
}

void vector_add(float* a, const float* b, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(a + i, vaddq_f32(va, vb));
    }
    for (; i < n; i++) a[i] += b[i];
}
#else
void rmsnorm(const float* x, const float* w, float* out, int dim) {
    float sum_sq = 0;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum_sq / dim + 1e-5f);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * w[i];
}

void apply_rope(const float* x, float* out, const float* cos_t, const float* sin_t, 
                int pos, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float x0 = x[i*2], x1 = x[i*2+1];
        float c = cos_t[pos*half + i], s = sin_t[pos*half + i];
        out[i*2] = x0*c - x1*s;
        out[i*2+1] = x0*s + x1*c;
    }
}

void vector_add(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}
#endif

//=============================================================================
// SRC-FFN Layer
//=============================================================================

struct SRCFFNLayer {
    // Projections (ternary packed)
    std::vector<uint8_t> w_gate;  // [ff_dim, embed_dim/4]
    std::vector<uint8_t> w_up;    // [ff_dim, embed_dim/4]
    std::vector<uint8_t> w_down;  // [embed_dim, ff_dim/4]
    
    // CfC parameters (per neuron) - PROPERLY INITIALIZED
    std::vector<float> decay;      // [ff_dim]
    std::vector<float> gate_bias;  // [ff_dim]
    
    // Hidden state
    std::vector<float> h;          // [ff_dim]
    
    // RMSNorm
    std::vector<float> norm_w;     // [embed_dim]
    
    void init() {
        int bpr_in = (EMBED_DIM + 3) / 4;
        int bpr_ff = (FF_DIM + 3) / 4;
        
        // Allocate weights
        w_gate.resize(FF_DIM * bpr_in);
        w_up.resize(FF_DIM * bpr_in);
        w_down.resize(EMBED_DIM * bpr_ff);
        
        // Random ternary weights (would be loaded from file in production)
        for (auto& b : w_gate) b = rand() & 0xFF;
        for (auto& b : w_up) b = rand() & 0xFF;
        for (auto& b : w_down) b = rand() & 0xFF;
        
        // Multi-scale CfC initialization
        decay.resize(FF_DIM);
        gate_bias.resize(FF_DIM);
        
        int quarter = FF_DIM / 4;
        for (int scale = 0; scale < 4; scale++) {
            int start = scale * quarter;
            int end = (scale == 3) ? FF_DIM : (scale + 1) * quarter;
            for (int i = start; i < end; i++) {
                decay[i] = DECAY_SCALES[scale];
                gate_bias[i] = GATE_BIAS_SCALES[scale];
            }
        }
        
        // Hidden state (zeros)
        h.resize(FF_DIM, 0.0f);
        
        // Norm weights
        norm_w.resize(EMBED_DIM, 1.0f);
    }
    
    void reset_hidden() {
        std::fill(h.begin(), h.end(), 0.0f);
    }
};

//=============================================================================
// Full Model
//=============================================================================

struct SRCFFNModel {
    std::vector<SRCFFNLayer> layers;
    std::vector<float> cos_table, sin_table;
    
    // Shared buffers
    alignas(64) std::vector<float> x_buf;
    alignas(64) std::vector<float> x_rot;
    alignas(64) std::vector<float> x_norm;
    alignas(64) std::vector<float> gate_buf;
    alignas(64) std::vector<float> up_buf;
    alignas(64) std::vector<float> mid_buf;
    alignas(64) std::vector<float> out_buf;
    alignas(64) std::vector<float> h_new;
    
    void init() {
        // Initialize layers
        layers.resize(N_LAYERS);
        for (auto& layer : layers) layer.init();
        
        // Precompute RoPE tables
        cos_table.resize(MAX_SEQ_LEN * (EMBED_DIM / 2));
        sin_table.resize(MAX_SEQ_LEN * (EMBED_DIM / 2));
        for (int pos = 0; pos < MAX_SEQ_LEN; pos++) {
            for (int i = 0; i < EMBED_DIM / 2; i++) {
                float freq = 1.0f / powf(ROPE_THETA, (float)(i * 2) / EMBED_DIM);
                float angle = pos * freq;
                cos_table[pos * (EMBED_DIM / 2) + i] = cosf(angle);
                sin_table[pos * (EMBED_DIM / 2) + i] = sinf(angle);
            }
        }
        
        // Allocate buffers
        x_buf.resize(EMBED_DIM);
        x_rot.resize(EMBED_DIM);
        x_norm.resize(EMBED_DIM);
        gate_buf.resize(FF_DIM);
        up_buf.resize(FF_DIM);
        mid_buf.resize(FF_DIM);
        out_buf.resize(EMBED_DIM);
        h_new.resize(FF_DIM);
    }
    
    void reset() {
        for (auto& layer : layers) layer.reset_hidden();
    }
    
    // Process single layer (for parallel execution)
    void process_layer_range(int layer_idx, int ff_start, int ff_end, int out_start, int out_end) {
        SRCFFNLayer& layer = layers[layer_idx];
        int bpr_in = (EMBED_DIM + 3) / 4;
        int bpr_ff = (FF_DIM + 3) / 4;
        
        // Compute gate and up projections for assigned rows
        for (int i = ff_start; i < ff_end; i++) {
            gate_buf[i] = ternary_dot(layer.w_gate.data() + i * bpr_in, x_norm.data(), EMBED_DIM);
            up_buf[i] = ternary_dot(layer.w_up.data() + i * bpr_in, x_norm.data(), EMBED_DIM);
            
            // CfC update
            float g = fast_sigmoid(up_buf[i] + CFC_ALPHA * layer.h[i] + layer.gate_bias[i]);
            float candidate = fast_tanh(up_buf[i]);
            h_new[i] = (1.0f - g) * layer.h[i] * layer.decay[i] + g * candidate;
            
            // Gated output
            mid_buf[i] = fast_silu(gate_buf[i]) * h_new[i];
        }
        
        // Compute down projection for assigned outputs
        for (int i = out_start; i < out_end; i++) {
            out_buf[i] = ternary_dot(layer.w_down.data() + i * bpr_ff, mid_buf.data(), FF_DIM);
        }
    }
    
    void forward_single(const float* input, float* output, int pos) {
        memcpy(x_buf.data(), input, EMBED_DIM * sizeof(float));
        
        for (int L = 0; L < N_LAYERS; L++) {
            SRCFFNLayer& layer = layers[L];
            
            // Apply RoPE
            apply_rope(x_buf.data(), x_rot.data(), cos_table.data(), sin_table.data(), pos, EMBED_DIM);
            
            // RMSNorm
            rmsnorm(x_rot.data(), layer.norm_w.data(), x_norm.data(), EMBED_DIM);
            
            // Process full layer (single-threaded)
            process_layer_range(L, 0, FF_DIM, 0, EMBED_DIM);
            
            // Update hidden state
            memcpy(layer.h.data(), h_new.data(), FF_DIM * sizeof(float));
            
            // Residual
            vector_add(x_buf.data(), out_buf.data(), EMBED_DIM);
        }
        
        memcpy(output, x_buf.data(), EMBED_DIM * sizeof(float));
    }
};

//=============================================================================
// Parallel Execution (6-core)
//=============================================================================

enum Phase { PHASE_IDLE = 0, PHASE_FFN_UP, PHASE_FFN_DOWN, PHASE_DONE };

struct ParallelContext {
    SRCFFNModel* model;
    std::atomic<int> phase{PHASE_IDLE};
    std::atomic<int> workers_done{0};
    std::atomic<bool> running{false};
    int cur_layer;
    pthread_t workers[TOTAL_WORKERS];
};

void* worker_thread(void* arg) {
    auto* ctx_pair = (std::pair<ParallelContext*, int>*)arg;
    ParallelContext* ctx = ctx_pair->first;
    int worker_id = ctx_pair->second;
    delete ctx_pair;
    
    // Pin to core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (worker_id < NUM_A55) {
        CPU_SET(A55_CORES[worker_id], &cpuset);
    } else {
        CPU_SET(A78_CORES[worker_id - NUM_A55], &cpuset);
    }
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    
    int last_phase = PHASE_IDLE;
    
    while (ctx->running) {
        int current_phase;
        while (ctx->running) {
            current_phase = ctx->phase.load(std::memory_order_acquire);
            if (current_phase != last_phase && current_phase != PHASE_IDLE) break;
            #ifdef __aarch64__
            asm volatile("yield");
            #endif
        }
        if (!ctx->running) break;
        last_phase = current_phase;
        
        // Compute work range
        if (current_phase == PHASE_FFN_UP) {
            // All workers do FFN gate+up+CfC
            int rows_per = FF_DIM / TOTAL_WORKERS;
            int extra = FF_DIM % TOTAL_WORKERS;
            int start = worker_id * rows_per + (worker_id < extra ? worker_id : extra);
            int count = rows_per + (worker_id < extra ? 1 : 0);
            
            ctx->model->process_layer_range(ctx->cur_layer, start, start + count, 0, 0);
        }
        else if (current_phase == PHASE_FFN_DOWN) {
            // All workers do FFN down
            int rows_per = EMBED_DIM / TOTAL_WORKERS;
            int extra = EMBED_DIM % TOTAL_WORKERS;
            int start = worker_id * rows_per + (worker_id < extra ? worker_id : extra);
            int count = rows_per + (worker_id < extra ? 1 : 0);
            
            ctx->model->process_layer_range(ctx->cur_layer, 0, 0, start, start + count);
        }
        
        ctx->workers_done.fetch_add(1, std::memory_order_release);
    }
    
    return nullptr;
}

void start_workers(ParallelContext* ctx) {
    ctx->running = true;
    ctx->phase = PHASE_IDLE;
    
    for (int i = 0; i < TOTAL_WORKERS; i++) {
        auto* arg = new std::pair<ParallelContext*, int>(ctx, i);
        pthread_create(&ctx->workers[i], nullptr, worker_thread, arg);
    }
    usleep(10000);
}

void stop_workers(ParallelContext* ctx) {
    ctx->running = false;
    ctx->phase = PHASE_DONE;
    for (int i = 0; i < TOTAL_WORKERS; i++) {
        pthread_join(ctx->workers[i], nullptr);
    }
}

void wait_workers(ParallelContext* ctx) {
    while (ctx->workers_done.load(std::memory_order_acquire) < TOTAL_WORKERS) {
        #ifdef __aarch64__
        asm volatile("yield");
        #endif
    }
}

void forward_parallel(SRCFFNModel* model, ParallelContext* ctx, 
                      const float* input, float* output, int pos) {
    memcpy(model->x_buf.data(), input, EMBED_DIM * sizeof(float));
    
    for (int L = 0; L < N_LAYERS; L++) {
        SRCFFNLayer& layer = model->layers[L];
        ctx->cur_layer = L;
        
        // RoPE + Norm (main thread)
        apply_rope(model->x_buf.data(), model->x_rot.data(), 
                   model->cos_table.data(), model->sin_table.data(), pos, EMBED_DIM);
        rmsnorm(model->x_rot.data(), layer.norm_w.data(), model->x_norm.data(), EMBED_DIM);
        
        // Phase 1: FFN up + CfC (all workers)
        ctx->workers_done = 0;
        ctx->phase.store(PHASE_FFN_UP, std::memory_order_release);
        wait_workers(ctx);
        
        // Update hidden state (main thread while workers wait)
        memcpy(layer.h.data(), model->h_new.data(), FF_DIM * sizeof(float));
        
        // Phase 2: FFN down (all workers)
        ctx->workers_done = 0;
        ctx->phase.store(PHASE_FFN_DOWN, std::memory_order_release);
        wait_workers(ctx);
        
        // Reset phase
        ctx->phase.store(PHASE_IDLE, std::memory_order_release);
        
        // Residual (main thread)
        vector_add(model->x_buf.data(), model->out_buf.data(), EMBED_DIM);
    }
    
    memcpy(output, model->x_buf.data(), EMBED_DIM * sizeof(float));
}

//=============================================================================
// Benchmark
//=============================================================================

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     SRC-FFN Production Inference Engine                      ║\n");
    printf("║     Spectral-Rotational-CfC Feed-Forward Network             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Configuration:\n");
    printf("  Embed dim:    %d\n", EMBED_DIM);
    printf("  FF dim:       %d\n", FF_DIM);
    printf("  Layers:       %d\n", N_LAYERS);
    printf("  Max seq len:  %d\n", MAX_SEQ_LEN);
    printf("\n");
    
    printf("Multi-scale CfC initialization:\n");
    for (int i = 0; i < 4; i++) {
        float g = 1.0f / (1.0f + expf(-GATE_BIAS_SCALES[i]));  // sigmoid approximation
        float eff_decay = (1.0f - g) * DECAY_SCALES[i];
        int horizon = (int)(logf(0.01f) / logf(eff_decay));
        printf("  Scale %d: decay=%.3f, bias=%.1f, g≈%.2f, horizon≈%d tokens\n",
               i+1, DECAY_SCALES[i], GATE_BIAS_SCALES[i], g, horizon);
    }
    printf("\n");
    
    // Initialize model
    SRCFFNModel model;
    printf("Initializing model...\n");
    model.init();
    
    // Calculate memory
    size_t weight_mem = 0;
    for (auto& layer : model.layers) {
        weight_mem += layer.w_gate.size() + layer.w_up.size() + layer.w_down.size();
        weight_mem += (layer.decay.size() + layer.gate_bias.size() + layer.h.size() + layer.norm_w.size()) * 4;
    }
    size_t hidden_mem = FF_DIM * N_LAYERS * 4;
    printf("  Weight memory:  %.1f MB\n", weight_mem / 1e6);
    printf("  Hidden memory:  %.1f KB\n", hidden_mem / 1024.0);
    printf("\n");
    
    std::vector<float> input(EMBED_DIM, 0.1f);
    std::vector<float> output(EMBED_DIM);
    
    // Single-threaded benchmark
    printf("=== Single-Core Baseline ===\n");
    int warmup = 3, iters = 10;
    
    model.reset();
    for (int i = 0; i < warmup; i++) {
        model.forward_single(input.data(), output.data(), i);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    model.reset();
    for (int i = 0; i < iters; i++) {
        model.forward_single(input.data(), output.data(), i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms_single = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    
    printf("  Time per token: %.2f ms\n", ms_single);
    printf("  Tokens/sec:     %.1f\n", 1000.0 / ms_single);
    
    // Six-core benchmark
    printf("\n=== Six-Core Parallel ===\n");
    
    ParallelContext ctx;
    ctx.model = &model;
    start_workers(&ctx);
    
    model.reset();
    for (int i = 0; i < warmup; i++) {
        forward_parallel(&model, &ctx, input.data(), output.data(), i);
    }
    
    start = std::chrono::high_resolution_clock::now();
    model.reset();
    for (int i = 0; i < iters; i++) {
        forward_parallel(&model, &ctx, input.data(), output.data(), i);
    }
    end = std::chrono::high_resolution_clock::now();
    double ms_parallel = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    
    stop_workers(&ctx);
    
    printf("  Time per token: %.2f ms\n", ms_parallel);
    printf("  Tokens/sec:     %.1f\n", 1000.0 / ms_parallel);
    printf("  Speedup:        %.2fx\n", ms_single / ms_parallel);
    
    // Summary
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                      RESULTS SUMMARY                         ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  CPU baseline (llama.cpp Q4):    50.0 tok/s                  ║\n");
    printf("║  SRC-FFN single-core:            %.1f tok/s                  ║\n", 1000.0/ms_single);
    printf("║  SRC-FFN six-core:               %.1f tok/s                  ║\n", 1000.0/ms_parallel);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    
    if (1000.0 / ms_parallel > 50) {
        printf("║               ★★★ BEATS CPU BASELINE! ★★★                    ║\n");
    }
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
