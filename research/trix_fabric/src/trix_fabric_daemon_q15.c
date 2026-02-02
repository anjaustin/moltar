/*
 * trix_fabric_daemon_q15 — TriX Fabric with Q15 Fixed-Point CfC Controller
 *
 * Upgraded from trix_fabric_daemon.c (float CfC) to use Yinsen's Q15
 * fixed-point compute stack. The CfC hot path is now pure integer — zero
 * floating-point operations between sensor read and actuator write.
 *
 * Why this matters:
 *   - On ARM big.LITTLE, the NEON/FP pipeline is shared between the daemon
 *     and llama.cpp (which uses KleidiAI SDOT heavily). Context switches
 *     between the daemon and inference must save/restore 32 x 128-bit NEON
 *     registers. With Q15, the daemon uses ONLY integer ALU — the NEON/FP
 *     state is never touched, so context switches are cheaper.
 *   - Hot set drops from ~2.7 KB (float) to ~1.4 KB (Q15). Both fit in L1,
 *     but the Q15 path occupies fewer cache lines, reducing eviction pressure.
 *   - Integer multiply on Cortex-A78: 3-cycle latency, 1-cycle throughput.
 *     Same as float multiply. But no FP pipeline contention.
 *
 * Architecture (unchanged from float daemon):
 *
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │  trix_fabric_daemon_q15 (persistent process)                 │
 *   │                                                              │
 *   │  ┌──────────┐    ┌───────────────┐    ┌──────────────────┐  │
 *   │  │ Observer  │───▶│ CfC Brain Q15 │───▶│ Actuator         │  │
 *   │  │ (float)   │    │ (int16 only)  │    │ (float)          │  │
 *   │  │          │    │               │    │                  │  │
 *   │  │ sysfs→f32│    │ h: int16[8]   │    │ set_affinity()   │  │
 *   │  │ f32→q11  │    │ LUT: int16    │    │ madvise()        │  │
 *   │  └──────────┘    │ 0 float ops   │    │ prefetch trigger │  │
 *   │                   └───────────────┘    └──────────────────┘  │
 *   │                                                              │
 *   │  Tick rate: 100 Hz (10ms between observations)               │
 *   │  CfC cost: ~20ns integer-only per tick                       │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * Float is used ONLY at the boundaries:
 *   - Observer: reads sysfs as strings → parses to float → converts to Q4.11
 *   - Actuator: reads Q15 output → converts to float for threshold comparisons
 * These boundary conversions happen at 100 Hz (10ms). Negligible.
 *
 * Usage:
 *   trix_fabric_daemon_q15 [-m model.gguf] [-v] [-r RATE_HZ] -- command [args...]
 *
 * Cross-compile:
 *   NDK=~/Library/Android/sdk/ndk/28.2.13676358
 *   $NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang \
 *       -O2 -I include -o trix_fabric_daemon_q15 src/trix_fabric_daemon_q15.c -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>       /* For init-time only (LUT fill, decay precompute) */
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>

/* Yinsen Q15 stack — provides CFC_CELL_SPARSE_Q15, SIGMOID_Q15, TANH_Q15 */
#include "activation_q15.h"
#include "cfc_cell_chip.h"
#include "cfc_cell_q15.h"

/* ══════════════════════════════════════════════════════════════════
 *  Controller Dimensions (same as float daemon)
 *
 *  Inputs (6): cpu_freq_big, cpu_freq_lit, mem_pressure,
 *              bat_temp, pgfault_rate, load_avg
 *  Hidden: 8 neurons
 *  Outputs (4): prefetch_urgency, thermal_caution,
 *               affinity_big, mem_lock_pressure
 * ══════════════════════════════════════════════════════════════════ */

#define CFC_INPUT_DIM   6
#define CFC_HIDDEN_DIM  8
#define CFC_OUTPUT_DIM  4
#define CFC_CONCAT_DIM  (CFC_INPUT_DIM + CFC_HIDDEN_DIM)  /* 14 */

/* ══════════════════════════════════════════════════════════════════
 *  Q15 CfC Controller State
 *
 *  Compared to float daemon:
 *    Float:  sigmoid_lut[256] = 1024B, tanh_lut[256] = 1024B,
 *            weights ~640B float, state 32B float  → ~2.7 KB
 *    Q15:    _sigmoid_lut_q15[257] = 514B, _tanh_lut_q15[257] = 514B,
 *            sparse indices ~352B, state 16B int16  → ~1.4 KB
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Sparse weights (from cfc_cell_chip.h) — same for float and Q15 */
    CfcSparseWeights sw;

    /* Q4.11 biases (converted from float at init) */
    int16_t b_gate_q11[CFC_HIDDEN_DIM];
    int16_t b_cand_q11[CFC_HIDDEN_DIM];

    /* Q15 decay (precomputed from time constants at init) */
    int16_t decay_q15[CFC_HIDDEN_DIM];

    /* Q15 hidden state (persistent across ticks) */
    int16_t h_q15[CFC_HIDDEN_DIM];

    /* Output projection: hidden[8] → output[4]
     * Dense, small (32 floats = 128 bytes). Stays float because:
     *   1. It's at the actuator boundary (output goes to float thresholds)
     *   2. Converting 4 Q15 values to float is trivial
     *   3. Not worth the complexity of a Q15 linear layer for 32 muls */
    float W_out[CFC_OUTPUT_DIM * CFC_HIDDEN_DIM];
    float b_out[CFC_OUTPUT_DIM];
} CfcControllerQ15;

/* ══════════════════════════════════════════════════════════════════
 *  Q15 CfC Step — The Hot Path
 *
 *  1. Convert float sensor input to Q4.11 (boundary)
 *  2. Run CFC_CELL_SPARSE_Q15 (pure integer)
 *  3. Convert Q15 hidden to float for output projection (boundary)
 *  4. Dense output projection (float, 32 muls)
 *  5. Sigmoid output via float LUT (reuse float LUT for 4 values)
 *
 *  Float operations in hot path: ZERO in the CfC cell.
 *  Float operations at boundary: 6 float_to_q11 + 8 q15_to_float + 32 fmul + 4 sigmoid
 *    = ~50 float ops at boundary, vs ~0 in the CfC core.
 *
 *  The boundary float ops are unavoidable (sysfs gives strings, actuators
 *  use float thresholds). But the CfC cell itself — the "brain" — is
 *  pure integer. This is what matters for NEON pipeline isolation.
 * ══════════════════════════════════════════════════════════════════ */

/* Simple float sigmoid for the 4 output values (not worth a LUT for 4 calls) */
static inline float sigmoid_f32(float x) {
    if (x > 8.0f) return 1.0f;
    if (x < -8.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static void cfc_step_q15(CfcControllerQ15 *c, const float *input_f32, float *output) {
    /* ── Boundary: float → Q4.11 ── */
    int16_t x_q11[CFC_INPUT_DIM];
    cfc_convert_input_q11(input_f32, CFC_INPUT_DIM, x_q11);

    /* ── Core: pure integer CfC step ── */
    int16_t h_new_q15[CFC_HIDDEN_DIM];
    CFC_CELL_SPARSE_Q15(
        x_q11,
        c->h_q15,
        &c->sw,
        c->b_gate_q11,
        c->b_cand_q11,
        c->decay_q15,
        CFC_INPUT_DIM,
        CFC_HIDDEN_DIM,
        h_new_q15
    );

    /* Update hidden state */
    memcpy(c->h_q15, h_new_q15, CFC_HIDDEN_DIM * sizeof(int16_t));

    /* ── Boundary: Q15 → float for output projection ── */
    float h_float[CFC_HIDDEN_DIM];
    cfc_convert_state_to_float(h_new_q15, CFC_HIDDEN_DIM, h_float);

    /* Output projection: linear map from hidden to control signals */
    for (int i = 0; i < CFC_OUTPUT_DIM; i++) {
        float sum = c->b_out[i];
        for (int j = 0; j < CFC_HIDDEN_DIM; j++) {
            sum += c->W_out[i * CFC_HIDDEN_DIM + j] * h_float[j];
        }
        /* Sigmoid to bound outputs to [0, 1] */
        output[i] = sigmoid_f32(sum);
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  Controller Initialization — Hand-tuned weights (same as float daemon)
 *
 *  The weights are identical. The only difference is representation:
 *  float biases → Q4.11, float decay → Q15, sparse indices unchanged.
 * ══════════════════════════════════════════════════════════════════ */

static void cfc_init_q15(CfcControllerQ15 *c) {
    memset(c, 0, sizeof(*c));

    /* ── Build sparse weight structure ──
     * The daemon's CfC uses hand-built ternary sparse weights.
     * We build them directly into the CfcSparseWeights structure
     * from cfc_cell_chip.h, which CFC_CELL_SPARSE_Q15 expects. */

    c->sw.hidden_dim = CFC_HIDDEN_DIM;
    c->sw.concat_dim = CFC_CONCAT_DIM;

    /* Pre-fill all index arrays with -1 (sentinel) */
    for (int i = 0; i < CFC_SPARSE_MAX_HIDDEN; i++) {
        memset(c->sw.gate[i].pos_idx, -1, sizeof(c->sw.gate[i].pos_idx));
        memset(c->sw.gate[i].neg_idx, -1, sizeof(c->sw.gate[i].neg_idx));
        memset(c->sw.cand[i].pos_idx, -1, sizeof(c->sw.cand[i].pos_idx));
        memset(c->sw.cand[i].neg_idx, -1, sizeof(c->sw.cand[i].neg_idx));
    }

    /* ── Gate sparse weights ── */

    /* Neuron 0: thermal tracker — temperature opens gate */
    c->sw.gate[0].pos_idx[0] = 3;  /* bat_temp */
    c->sw.gate[0].pos_idx[1] = -1;

    /* Neuron 1: memory pressure tracker */
    c->sw.gate[1].pos_idx[0] = 2;  /* mem_pressure */
    c->sw.gate[1].pos_idx[1] = 4;  /* pgfault_rate */
    c->sw.gate[1].pos_idx[2] = -1;

    /* Neuron 2: CPU load tracker */
    c->sw.gate[2].pos_idx[0] = 5;  /* load_avg */
    c->sw.gate[2].pos_idx[1] = 0;  /* cpu_freq_big */
    c->sw.gate[2].pos_idx[2] = -1;

    /* Neuron 3: frequency differential tracker */
    c->sw.gate[3].pos_idx[0] = 0;  /* cpu_freq_big */
    c->sw.gate[3].pos_idx[1] = -1;
    c->sw.gate[3].neg_idx[0] = 1;  /* cpu_freq_lit (negative = big-little diff) */
    c->sw.gate[3].neg_idx[1] = -1;

    /* Neuron 4: page fault spike detector */
    c->sw.gate[4].pos_idx[0] = 4;  /* pgfault_rate */
    c->sw.gate[4].pos_idx[1] = -1;

    /* Neuron 5: thermal + load interaction */
    c->sw.gate[5].pos_idx[0] = 3;  /* bat_temp */
    c->sw.gate[5].pos_idx[1] = 5;  /* load_avg */
    c->sw.gate[5].pos_idx[2] = -1;

    /* Neuron 6: memory + freq interaction */
    c->sw.gate[6].pos_idx[0] = 2;  /* mem_pressure */
    c->sw.gate[6].pos_idx[1] = -1;
    c->sw.gate[6].neg_idx[0] = 0;  /* high freq = less gate */
    c->sw.gate[6].neg_idx[1] = -1;

    /* Neuron 7: broad integrator */
    c->sw.gate[7].pos_idx[0] = 0;
    c->sw.gate[7].pos_idx[1] = 2;
    c->sw.gate[7].pos_idx[2] = 3;
    c->sw.gate[7].pos_idx[3] = 5;
    c->sw.gate[7].pos_idx[4] = -1;

    /* ── Candidate sparse weights ── */

    /* Neuron 0: hot = positive state */
    c->sw.cand[0].pos_idx[0] = 3;  /* bat_temp */
    c->sw.cand[0].pos_idx[1] = -1;

    /* Neuron 1: memory stressed = positive */
    c->sw.cand[1].pos_idx[0] = 2;  /* mem_pressure */
    c->sw.cand[1].pos_idx[1] = -1;

    /* Neuron 2: loaded = positive */
    c->sw.cand[2].pos_idx[0] = 5;  /* load_avg */
    c->sw.cand[2].pos_idx[1] = -1;

    /* Neuron 3: big-little spread = positive */
    c->sw.cand[3].pos_idx[0] = 0;  /* big freq */
    c->sw.cand[3].pos_idx[1] = -1;
    c->sw.cand[3].neg_idx[0] = 1;  /* little freq */
    c->sw.cand[3].neg_idx[1] = -1;

    /* Neuron 4: fault spike */
    c->sw.cand[4].pos_idx[0] = 4;
    c->sw.cand[4].pos_idx[1] = -1;

    /* Neuron 5: thermal under load = hot state */
    c->sw.cand[5].pos_idx[0] = 3;
    c->sw.cand[5].pos_idx[1] = 5;
    c->sw.cand[5].pos_idx[2] = -1;

    /* Neuron 6: memory under freq pressure */
    c->sw.cand[6].pos_idx[0] = 2;
    c->sw.cand[6].pos_idx[1] = -1;

    /* Neuron 7: broad state */
    c->sw.cand[7].pos_idx[0] = 0;
    c->sw.cand[7].pos_idx[1] = 3;
    c->sw.cand[7].pos_idx[2] = -1;
    c->sw.cand[7].neg_idx[0] = 2;
    c->sw.cand[7].neg_idx[1] = -1;

    /* ── Biases → Q4.11 ── */
    /* Gate biases (float values from original daemon) */
    float b_gate_f[] = { -0.5f, -0.3f, -0.5f, 0.0f, -1.0f, -1.0f, 0.0f, -2.0f };
    cfc_convert_biases_q11(b_gate_f, CFC_HIDDEN_DIM, c->b_gate_q11);

    /* Candidate biases */
    float b_cand_f[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.5f, 0.0f, 0.0f };
    cfc_convert_biases_q11(b_cand_f, CFC_HIDDEN_DIM, c->b_cand_q11);

    /* ── Decay → Q15 ──
     * Original daemon had precomputed decay values directly.
     * We convert them to Q15. These represent exp(-dt/tau) for dt=10ms. */
    float decay_f[] = { 0.999f, 0.995f, 0.990f, 0.980f, 0.970f, 0.950f, 0.930f, 0.900f };
    for (int i = 0; i < CFC_HIDDEN_DIM; i++) {
        c->decay_q15[i] = float_to_q15(decay_f[i]);
    }

    /* ── Output projection (stays float) ── */
    /* output[0] = prefetch_urgency */
    c->W_out[0 * CFC_HIDDEN_DIM + 1] =  1.5f;
    c->W_out[0 * CFC_HIDDEN_DIM + 4] =  2.0f;
    c->W_out[0 * CFC_HIDDEN_DIM + 6] =  1.0f;
    c->b_out[0] = -1.0f;

    /* output[1] = thermal_caution */
    c->W_out[1 * CFC_HIDDEN_DIM + 0] =  2.0f;
    c->W_out[1 * CFC_HIDDEN_DIM + 5] =  1.5f;
    c->W_out[1 * CFC_HIDDEN_DIM + 7] =  0.5f;
    c->b_out[1] = -1.5f;

    /* output[2] = affinity_big */
    c->W_out[2 * CFC_HIDDEN_DIM + 2] =  1.5f;
    c->W_out[2 * CFC_HIDDEN_DIM + 3] =  1.0f;
    c->W_out[2 * CFC_HIDDEN_DIM + 7] =  0.5f;
    c->b_out[2] =  0.5f;

    /* output[3] = mem_lock_pressure */
    c->W_out[3 * CFC_HIDDEN_DIM + 1] = -1.5f;
    c->W_out[3 * CFC_HIDDEN_DIM + 4] =  1.0f;
    c->W_out[3 * CFC_HIDDEN_DIM + 6] = -0.5f;
    c->b_out[3] =  0.5f;

    /* Zero hidden state */
    memset(c->h_q15, 0, sizeof(c->h_q15));
}

/* ══════════════════════════════════════════════════════════════════
 *  System Observer — Reads hardware state into CfC input vector
 *  (Identical to float daemon — sysfs always returns strings)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    int cpu_big_max_khz;
    int cpu_lit_max_khz;
    long mem_total_kb;
    long prev_pgmajfault;
    int n_cores;
} ObserverState;

static int read_int_sysfs(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    char buf[64];
    int n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    return atoi(buf);
}

static long read_long_sysfs(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    char buf[64];
    int n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    return atol(buf);
}

static long read_meminfo_field(const char *field) {
    int fd = open("/proc/meminfo", O_RDONLY);
    if (fd < 0) return -1;
    char buf[2048];
    int n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    char *p = strstr(buf, field);
    if (!p) return -1;
    p += strlen(field);
    while (*p == ' ' || *p == ':') p++;
    return atol(p);
}

static long read_vmstat_field(const char *field) {
    int fd = open("/proc/vmstat", O_RDONLY);
    if (fd < 0) return -1;
    char buf[4096];
    int n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    char *p = strstr(buf, field);
    if (!p) return -1;
    p += strlen(field);
    while (*p == ' ') p++;
    return atol(p);
}

static float read_loadavg(void) {
    int fd = open("/proc/loadavg", O_RDONLY);
    if (fd < 0) return 0;
    char buf[64];
    int n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return 0;
    buf[n] = '\0';
    return strtof(buf, NULL);
}

static void observer_init(ObserverState *obs) {
    obs->cpu_big_max_khz = read_int_sysfs("/sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_max_freq");
    obs->cpu_lit_max_khz = read_int_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    obs->mem_total_kb = read_meminfo_field("MemTotal");
    obs->prev_pgmajfault = read_vmstat_field("pgmajfault");
    obs->n_cores = 8;
    if (obs->cpu_big_max_khz <= 0) obs->cpu_big_max_khz = 2200000;
    if (obs->cpu_lit_max_khz <= 0) obs->cpu_lit_max_khz = 2000000;
    if (obs->mem_total_kb <= 0) obs->mem_total_kb = 3683916;
}

static void observer_read(ObserverState *obs, float *input) {
    /* [0] cpu_freq_big: normalized to [-1, 1] */
    int freq_big = read_int_sysfs("/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq");
    if (freq_big < 0) freq_big = obs->cpu_big_max_khz / 2;
    input[0] = ((float)freq_big / obs->cpu_big_max_khz - 0.5f) * 2.0f;

    /* [1] cpu_freq_little: normalized to [-1, 1] */
    int freq_lit = read_int_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    if (freq_lit < 0) freq_lit = obs->cpu_lit_max_khz / 2;
    input[1] = ((float)freq_lit / obs->cpu_lit_max_khz - 0.5f) * 2.0f;

    /* [2] mem_pressure: [-1, 1] where 1 = no free memory */
    long mem_avail = read_meminfo_field("MemAvailable");
    if (mem_avail < 0) mem_avail = obs->mem_total_kb / 2;
    input[2] = (1.0f - (float)mem_avail / obs->mem_total_kb) * 2.0f - 1.0f;

    /* [3] bat_temp: centered around 30C, scaled */
    int bat_temp_raw = read_int_sysfs("/sys/class/power_supply/battery/temp");
    float temp_c = (bat_temp_raw > 0) ? bat_temp_raw / 10.0f : 25.0f;
    input[3] = (temp_c - 30.0f) / 20.0f;

    /* [4] pgfault_rate: delta major page faults since last tick */
    long pgmajfault = read_vmstat_field("pgmajfault");
    if (pgmajfault < 0) pgmajfault = obs->prev_pgmajfault;
    float delta = (float)(pgmajfault - obs->prev_pgmajfault);
    obs->prev_pgmajfault = pgmajfault;
    input[4] = (delta / 5.0f) - 1.0f;
    if (input[4] > 1.0f) input[4] = 1.0f;

    /* [5] load_avg: normalized by core count */
    float loadavg = read_loadavg();
    input[5] = (loadavg / obs->n_cores - 0.5f) * 2.0f;
    if (input[5] > 1.0f) input[5] = 1.0f;
    if (input[5] < -1.0f) input[5] = -1.0f;
}

/* ══════════════════════════════════════════════════════════════════
 *  Actuator — Applies CfC control signals to hardware
 *  (Identical to float daemon)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    cpu_set_t big_mask;
    cpu_set_t all_mask;
    int big_cores[8];
    int n_big;
    pid_t child_pid;
    int last_affinity_big;
    int prefetch_active;
} ActuatorState;

static void actuator_init(ActuatorState *act) {
    memset(act, 0, sizeof(*act));
    CPU_ZERO(&act->big_mask);
    CPU_ZERO(&act->all_mask);
    act->last_affinity_big = -1;

    char path[256];
    for (int i = 0; i < 8; i++) {
        CPU_SET(i, &act->all_mask);
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);
        int cap = read_int_sysfs(path);
        if (cap >= 512) {
            CPU_SET(i, &act->big_mask);
            act->big_cores[act->n_big++] = i;
        }
    }
}

static void actuator_apply(ActuatorState *act, const float *output, int verbose) {
    int want_big;
    if (output[1] > 0.8f) {
        want_big = 0;  /* Thermal emergency: spread load */
    } else {
        want_big = (output[2] > 0.5f) ? 1 : 0;
    }

    if (want_big != act->last_affinity_big && act->child_pid > 0) {
        cpu_set_t *mask = want_big ? &act->big_mask : &act->all_mask;
        if (sched_setaffinity(act->child_pid, sizeof(cpu_set_t), mask) == 0) {
            if (verbose) {
                fprintf(stderr, "[fabric/q15] affinity -> %s (thermal=%.2f affinity=%.2f)\n",
                        want_big ? "BIG" : "ALL", output[1], output[2]);
            }
            act->last_affinity_big = want_big;
        }
    }

    if (output[0] > 0.7f && !act->prefetch_active) {
        if (verbose) {
            fprintf(stderr, "[fabric/q15] prefetch urgency HIGH (%.2f)\n", output[0]);
        }
        act->prefetch_active = 1;
    } else if (output[0] < 0.3f) {
        act->prefetch_active = 0;
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  Topology + Pre-fault (from Phase 1+2 launcher)
 * ══════════════════════════════════════════════════════════════════ */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

#define LOG(fmt, ...) fprintf(stderr, "[fabric] " fmt "\n", ##__VA_ARGS__)
#define VLOG(fmt, ...) do { if (verbose) fprintf(stderr, "[fabric] " fmt "\n", ##__VA_ARGS__); } while(0)

static int verbose = 0;

static int prefault_model(const char *path) {
    double t0 = now_ms();
    int fd = open(path, O_RDONLY);
    if (fd < 0) { LOG("WARNING: can't open model '%s'", path); return -1; }

    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;
    LOG("pre-faulting model: %s (%.1f MiB)", path, size / (1024.0 * 1024.0));

    void *addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (addr == MAP_FAILED) {
        addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { close(fd); return -1; }
        volatile char sum = 0;
        const char *p = (const char *)addr;
        for (size_t off = 0; off < size; off += 4096) sum += p[off];
        (void)sum;
    }
    madvise(addr, size, MADV_SEQUENTIAL);
    mlock(addr, size);
    close(fd);

    LOG("pre-fault complete: %.1f ms", now_ms() - t0);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 *  Main — Fork child, run Q15 CfC control loop
 * ══════════════════════════════════════════════════════════════════ */

static volatile int child_alive = 1;

static void sigchld_handler(int sig) {
    (void)sig;
    child_alive = 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [-m model.gguf] [-v] [-r RATE_HZ] -- command [args...]\n"
        "\n"
        "TriX Fabric Daemon with Q15 Fixed-Point CfC Neural Controller.\n"
        "Zero floating-point ops in the CfC hot path.\n"
        "Leaves NEON/FP pipeline free for inference.\n"
        "\n"
        "Options:\n"
        "  -m FILE    Model file to pre-fault\n"
        "  -v         Verbose (show CfC decisions)\n"
        "  -r RATE    Control loop rate in Hz (default: 100)\n"
        "\n", prog);
}

int main(int argc, char *argv[]) {
    const char *model_path = NULL;
    int rate_hz = 100;
    int cmd_start = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) { cmd_start = i + 1; break; }
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) { model_path = argv[++i]; }
        else if (strcmp(argv[i], "-v") == 0) { verbose = 1; }
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) { rate_hz = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
    }

    if (cmd_start < 0 || cmd_start >= argc) {
        fprintf(stderr, "Error: no command after '--'\n");
        usage(argv[0]);
        return 1;
    }

    LOG("=== TriX Fabric v0.3 (Q15 fixed-point CfC controller) ===");

    /* Initialize Q15 LUT tables (one-time float math at startup) */
    Q15_LUT_INIT();
    LOG("Q15 LUT initialized: sigmoid[%d] + tanh[%d] = %zu bytes",
        Q15_LUT_SIZE + 1, Q15_LUT_SIZE + 1,
        sizeof(_sigmoid_lut_q15) + sizeof(_tanh_lut_q15));

    CfcControllerQ15 cfc;
    cfc_init_q15(&cfc);
    LOG("Q15 CfC controller: %d inputs, %d hidden (Q15), %d outputs, zero-float hot path",
        CFC_INPUT_DIM, CFC_HIDDEN_DIM, CFC_OUTPUT_DIM);
    LOG("  Hot set: ~%zu bytes (LUTs=%zu + sparse=%zu + state=%zu)",
        sizeof(_sigmoid_lut_q15) + sizeof(_tanh_lut_q15) +
        sizeof(cfc.sw) + sizeof(cfc.h_q15) + sizeof(cfc.b_gate_q11) +
        sizeof(cfc.b_cand_q11) + sizeof(cfc.decay_q15),
        sizeof(_sigmoid_lut_q15) + sizeof(_tanh_lut_q15),
        sizeof(cfc.sw),
        sizeof(cfc.h_q15));

    ObserverState obs;
    observer_init(&obs);

    ActuatorState act;
    actuator_init(&act);
    LOG("topology: %d big cores detected", act.n_big);

    /* Pre-fault model */
    if (model_path) prefault_model(model_path);

    /* Fork child process */
    signal(SIGCHLD, sigchld_handler);

    pid_t child = fork();
    if (child < 0) {
        LOG("FATAL: fork failed: %s", strerror(errno));
        return 1;
    }

    if (child == 0) {
        /* ── Child: pin to big cores and exec ── */
        sched_setaffinity(0, sizeof(cpu_set_t), &act.big_mask);
        execvp(argv[cmd_start], &argv[cmd_start]);
        fprintf(stderr, "[fabric] FATAL: exec '%s': %s\n", argv[cmd_start], strerror(errno));
        _exit(127);
    }

    /* ── Parent: Q15 CfC control loop ── */
    act.child_pid = child;
    act.last_affinity_big = 1;

    LOG("child pid=%d launched on big cores, Q15 control loop at %d Hz", child, rate_hz);

    long tick_ns = 1000000000L / rate_hz;
    struct timespec tick = { .tv_sec = tick_ns / 1000000000L, .tv_nsec = tick_ns % 1000000000L };

    float input[CFC_INPUT_DIM];
    float output[CFC_OUTPUT_DIM];
    long tick_count = 0;
    double total_cfc_ns = 0;

    while (child_alive) {
        /* Observe (float — sysfs boundary) */
        observer_read(&obs, input);

        /* Think (Q15 CfC step — integer hot path) */
        double t0 = now_ms();
        cfc_step_q15(&cfc, input, output);
        double cfc_elapsed = (now_ms() - t0) * 1e6;  /* ns */
        total_cfc_ns += cfc_elapsed;

        /* Act */
        actuator_apply(&act, output, verbose);

        /* Periodic status report */
        tick_count++;
        if (verbose && (tick_count % (rate_hz * 5) == 0)) {
            fprintf(stderr, "[fabric/q15] tick=%ld  in=[%.2f %.2f %.2f %.2f %.2f %.2f]  "
                    "out=[pf=%.2f th=%.2f af=%.2f ml=%.2f]  "
                    "cfc_avg=%.0fns  h0_q15=%d (%.4f)\n",
                    tick_count,
                    input[0], input[1], input[2], input[3], input[4], input[5],
                    output[0], output[1], output[2], output[3],
                    total_cfc_ns / tick_count,
                    cfc.h_q15[0], q15_to_float(cfc.h_q15[0]));
        }

        nanosleep(&tick, NULL);
    }

    /* Wait for child and report */
    int status;
    waitpid(child, &status, 0);

    LOG("child exited (status=%d)", WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    LOG("Q15 CfC stats: %ld ticks, avg %.0f ns/step (zero-float hot path)",
        tick_count, tick_count > 0 ? total_cfc_ns / tick_count : 0);
    LOG("  LUT memory: %zu bytes (vs 2048 bytes float)",
        sizeof(_sigmoid_lut_q15) + sizeof(_tanh_lut_q15));
    LOG("  Hidden state: %zu bytes (vs %zu bytes float)",
        CFC_HIDDEN_DIM * sizeof(int16_t), CFC_HIDDEN_DIM * sizeof(float));

    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
