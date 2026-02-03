/*
 * SPLINE_Q15 — Fixed-Point Cubic Hermite Spline Interpolation
 *
 * Integer-only cubic Hermite interpolation for continuous-time signal
 * reconstruction from discrete observations. Zero floating-point ops.
 *
 * This is the Q15 port of the cubic Hermite spline from Yinsen's
 * quant_probe2_fpu.c, adapted for time-series interpolation instead
 * of activation function approximation.
 *
 * Use case: The fabric daemon observes system state (sysfs reads) at
 * a LOW rate (5 Hz), fits cubic splines through the observation history,
 * and evaluates them at a HIGH rate (100 Hz) to feed the CfC controller.
 * This reduces syscall overhead 20x while maintaining smooth input.
 *
 * Cubic Hermite basis functions:
 *   h00(t) = 2t^3 - 3t^2 + 1     (value at left endpoint)
 *   h10(t) = t^3 - 2t^2 + t       (derivative at left endpoint)
 *   h01(t) = -2t^3 + 3t^2         (value at right endpoint)
 *   h11(t) = t^3 - t^2             (derivative at right endpoint)
 *
 * Interpolant:
 *   p(t) = h00*y0 + h10*dx*m0 + h01*y1 + h11*dx*m1
 *   where t in [0, 1], dx = interval width, m0/m1 = derivatives
 *
 * Q15 representation:
 *   - Signal values: Q4.11 (range [-16, +16), same as CfC input)
 *   - Time parameter t: Q15 (range [0, 1))
 *   - Derivatives: Q4.11 per Q4.11 time unit
 *   - Interval dx: Q15 (normalized to 1.0 = one observation interval)
 *
 * Created: February 2026
 * Part of the Yinsen Sentinel fixed-point compute stack.
 */

#ifndef TRIX_SPLINE_Q15_H
#define TRIX_SPLINE_Q15_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Q15 / Q4.11 ARITHMETIC (reuse from activation_q15.h if included,
 * otherwise self-contained definitions)
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifndef TRIX_ACTIVATION_Q15_H
/* Self-contained if activation_q15.h isn't included */
#define Q15_ONE      ((int16_t)32767)
#define Q15_ZERO     ((int16_t)0)
#define Q11_SHIFT    11
#define Q11_ONE      (1 << Q11_SHIFT)  /* 2048 */

static inline int16_t q15_mul_(int16_t a, int16_t b) {
    int32_t product = (int32_t)a * (int32_t)b;
    return (int16_t)((product + (1 << 14)) >> 15);
}
#else
/* Use the q15_mul from activation_q15.h */
#define q15_mul_ q15_mul
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * CUBIC HERMITE EVALUATION — Integer-Only
 *
 * Given two knot points with values and derivatives:
 *   (y0, m0) at left endpoint
 *   (y1, m1) at right endpoint
 *   t = interpolation parameter in Q15, range [0, Q15_ONE]
 *
 * Returns interpolated value in Q4.11.
 *
 * The Hermite basis functions evaluated at t (in Q15):
 *   h00 = 2t^3 - 3t^2 + 1
 *   h10 = t^3 - 2t^2 + t
 *   h01 = -2t^3 + 3t^2
 *   h11 = t^3 - t^2
 *
 * We compute in int32_t to avoid overflow, then truncate to Q4.11.
 *
 * Cost: ~12 multiplies + ~10 adds. All integer. No branches.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * hermite_eval_q15 — Evaluate cubic Hermite interpolant at parameter t.
 *
 * @param y0   Left knot value (Q4.11)
 * @param m0   Left knot derivative (Q4.11 per unit interval)
 * @param y1   Right knot value (Q4.11)
 * @param m1   Right knot derivative (Q4.11 per unit interval)
 * @param t    Interpolation parameter (Q15: 0=left, 32767=right)
 * @return     Interpolated value (Q4.11)
 *
 * All intermediate math in int32_t. Final result clamped to int16_t.
 */
static inline int16_t hermite_eval_q15(
    int16_t y0, int16_t m0,
    int16_t y1, int16_t m1,
    int16_t t
) {
    /* t is Q15. We need t^2 and t^3 in Q15.
     * t^2 = (t * t) >> 15   (Q15 * Q15 >> 15 = Q15)
     * t^3 = (t^2 * t) >> 15 (Q15 * Q15 >> 15 = Q15)
     */
    int32_t t32 = (int32_t)t;
    int32_t t2 = (t32 * t32 + (1 << 14)) >> 15;           /* Q15 */
    int32_t t3 = (t2 * t32 + (1 << 14)) >> 15;            /* Q15 */

    /* Hermite basis in Q15:
     *   h00 = 2*t3 - 3*t2 + Q15_ONE   = 2*t3 - 3*t2 + 32767
     *   h10 = t3 - 2*t2 + t
     *   h01 = -2*t3 + 3*t2
     *   h11 = t3 - t2
     */
    int32_t h00 = 2 * t3 - 3 * t2 + Q15_ONE;
    int32_t h10 = t3 - 2 * t2 + t32;
    int32_t h01 = -2 * t3 + 3 * t2;
    int32_t h11 = t3 - t2;

    /* The result is:
     *   p = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
     *
     * h00, h01 are Q15 weighting y0, y1 which are Q4.11.
     * Product: Q15 * Q4.11 → shift >> 15 to get Q4.11.
     *
     * h10, h11 are Q15 weighting m0, m1 which are Q4.11.
     * Same: Q15 * Q4.11 >> 15 → Q4.11.
     */
    int32_t result = 0;
    result += (h00 * (int32_t)y0 + (1 << 14)) >> 15;
    result += (h10 * (int32_t)m0 + (1 << 14)) >> 15;
    result += (h01 * (int32_t)y1 + (1 << 14)) >> 15;
    result += (h11 * (int32_t)m1 + (1 << 14)) >> 15;

    /* Clamp to Q4.11 range */
    if (result > 32767) result = 32767;
    if (result < -32768) result = -32768;
    return (int16_t)result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * OBSERVATION RING BUFFER + SPLINE KNOT MANAGEMENT
 *
 * Stores the last N observations per channel with timestamps.
 * Computes finite-difference derivatives at each knot point.
 * Provides cubic Hermite interpolation at any time between knots.
 *
 * Ring buffer of 4 knots gives us one complete cubic segment (the most
 * recent interval) plus enough history for Catmull-Rom-style derivative
 * estimation at the endpoints.
 *
 * Derivative estimation (Catmull-Rom):
 *   m[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
 *
 * This gives C1 continuity across segments with minimal storage.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SPLINE_RING_SIZE  4   /* 4 knots per channel */

typedef struct {
    int16_t  y[SPLINE_RING_SIZE];    /* Knot values in Q4.11 */
    uint32_t t_ms[SPLINE_RING_SIZE]; /* Timestamps in milliseconds */
    int      head;                    /* Next write position */
    int      count;                   /* How many valid knots (0..4) */
} SplineChannel;

/**
 * spline_channel_init — Reset a spline channel.
 */
static inline void spline_channel_init(SplineChannel *ch) {
    memset(ch, 0, sizeof(*ch));
}

/**
 * spline_channel_push — Add a new observation (knot point).
 *
 * @param ch     Spline channel
 * @param value  Observed value in Q4.11
 * @param t_ms   Timestamp in milliseconds
 */
static inline void spline_channel_push(SplineChannel *ch, int16_t value, uint32_t t_ms) {
    ch->y[ch->head] = value;
    ch->t_ms[ch->head] = t_ms;
    ch->head = (ch->head + 1) % SPLINE_RING_SIZE;
    if (ch->count < SPLINE_RING_SIZE) ch->count++;
}

/**
 * spline_channel_get — Get knot at index (0 = oldest valid, count-1 = newest).
 */
static inline void spline_channel_get(
    const SplineChannel *ch, int idx,
    int16_t *y_out, uint32_t *t_out
) {
    /* idx 0 = oldest, idx count-1 = newest */
    int ring_idx = (ch->head - ch->count + idx + SPLINE_RING_SIZE) % SPLINE_RING_SIZE;
    *y_out = ch->y[ring_idx];
    *t_out = ch->t_ms[ring_idx];
}

/**
 * spline_channel_eval — Evaluate spline at time t_ms.
 *
 * Uses the two most recent knots as the interpolation segment.
 * Derivatives computed via Catmull-Rom from adjacent knots.
 *
 * If fewer than 2 knots: returns the last known value (zero-order hold).
 * If exactly 2 knots: uses linear interpolation (no derivative info).
 * If 3+ knots: full cubic Hermite with Catmull-Rom derivatives.
 *
 * @param ch     Spline channel
 * @param t_ms   Query time in milliseconds
 * @return       Interpolated value in Q4.11
 */
static inline int16_t spline_channel_eval(const SplineChannel *ch, uint32_t t_ms) {
    if (ch->count == 0) return 0;
    if (ch->count == 1) {
        /* Zero-order hold */
        int16_t y; uint32_t t;
        spline_channel_get(ch, 0, &y, &t);
        return y;
    }

    /* Get the two most recent knots (right segment boundary) */
    int16_t  y0, y1;
    uint32_t t0, t1;
    spline_channel_get(ch, ch->count - 2, &y0, &t0);  /* second newest */
    spline_channel_get(ch, ch->count - 1, &y1, &t1);  /* newest */

    /* Time parameter t in [0, 1] mapped to Q15 */
    uint32_t dt = t1 - t0;
    if (dt == 0) return y1;

    /* Allow extrapolation slightly beyond t1 (for prediction) */
    int32_t elapsed = (int32_t)(t_ms - t0);
    if (elapsed < 0) return y0;  /* before segment — hold */

    /* t_q15 = elapsed / dt, mapped to Q15 */
    int32_t t_q15 = (elapsed * (int32_t)Q15_ONE) / (int32_t)dt;

    /* Clamp extrapolation to 1.5x the interval (don't predict too far) */
    int32_t max_t = Q15_ONE + (Q15_ONE >> 1);  /* 1.5 in Q15 = 49151 */
    if (t_q15 > max_t) t_q15 = max_t;

    /* Compute derivatives via Catmull-Rom */
    int16_t m0, m1;

    if (ch->count >= 3) {
        /* m0 = (y1 - y[-1]) / (t1 - t[-1]), scaled to per-interval units */
        int16_t y_prev; uint32_t t_prev;
        spline_channel_get(ch, ch->count - 3, &y_prev, &t_prev);

        /* Catmull-Rom: m0 = (y1 - y_prev) * dt / (t1 - t_prev)
         * This scales the derivative to "per dt" units so the Hermite
         * basis functions work correctly with t in [0, 1]. */
        int32_t span = (int32_t)(t1 - t_prev);
        if (span > 0) {
            m0 = (int16_t)(((int32_t)(y1 - y_prev) * (int32_t)dt) / span);
        } else {
            m0 = (int16_t)((int32_t)(y1 - y0));
        }
    } else {
        /* Only 2 knots — use forward difference */
        m0 = (int16_t)((int32_t)(y1 - y0));
    }

    if (ch->count >= 4) {
        /* m1 = (y[+1] - y[-1]) / (t[+1] - t[-1]) for the right endpoint.
         * But y1 IS the newest point — we don't have y[+1]. 
         * Use the derivative estimated from the last 3 points instead:
         * m1 = (y1 - y_prev2) / (t1 - t_prev2) * dt  
         * where prev2 is 2 back from newest. 
         *
         * Actually, for the right endpoint of the current segment, the 
         * Catmull-Rom derivative uses points on either side. Since y1 is
         * the newest, we estimate m1 from the trend of the last 2 intervals. */
        int16_t y_prev; uint32_t t_prev;
        spline_channel_get(ch, ch->count - 3, &y_prev, &t_prev);

        /* Simple: m1 uses same estimation as m0 but shifted one step.
         * m1 = (y1 - y_prev) * dt / (t1 - t_prev) */
        int32_t span = (int32_t)(t1 - t_prev);
        if (span > 0) {
            m1 = (int16_t)(((int32_t)(y1 - y_prev) * (int32_t)dt) / span);
        } else {
            m1 = m0;
        }
    } else {
        /* Not enough points for right derivative — use same as left */
        m1 = m0;
    }

    return hermite_eval_q15(y0, m0, y1, m1, (int16_t)t_q15);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MULTI-CHANNEL SPLINE OBSERVER
 *
 * Wraps N spline channels into a single observer that replaces the
 * syscall-heavy observer_read() with spline interpolation.
 *
 * Usage pattern:
 *   1. At LOW rate (5 Hz): call spline_observe() → does real syscalls,
 *      pushes observations into spline channels
 *   2. At HIGH rate (100 Hz): call spline_interpolate() → evaluates
 *      splines at current time, zero syscalls
 *   3. Feed interpolated values into CfC as usual
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SPLINE_MAX_CHANNELS  8

typedef struct {
    SplineChannel ch[SPLINE_MAX_CHANNELS];
    int n_channels;
    uint32_t last_observe_ms;   /* Timestamp of last real observation */
    uint32_t observe_interval;  /* Minimum ms between real observations */
} SplineObserver;

/**
 * spline_observer_init — Initialize multi-channel spline observer.
 *
 * @param obs          Observer state
 * @param n_channels   Number of input channels
 * @param observe_hz   Real observation rate in Hz (e.g., 5)
 */
static inline void spline_observer_init(
    SplineObserver *obs, int n_channels, int observe_hz
) {
    memset(obs, 0, sizeof(*obs));
    obs->n_channels = n_channels;
    if (n_channels > SPLINE_MAX_CHANNELS) obs->n_channels = SPLINE_MAX_CHANNELS;
    for (int i = 0; i < obs->n_channels; i++) {
        spline_channel_init(&obs->ch[i]);
    }
    obs->observe_interval = 1000 / observe_hz;  /* ms per observation */
    obs->last_observe_ms = 0;
}

/**
 * spline_observer_needs_observation — Check if it's time for a real read.
 *
 * @param obs    Observer state
 * @param now_ms Current time in milliseconds
 * @return       1 if real observation is needed, 0 if spline can interpolate
 */
static inline int spline_observer_needs_observation(
    const SplineObserver *obs, uint32_t now_ms
) {
    if (obs->ch[0].count < 2) return 1;  /* Need at least 2 points */
    return (now_ms - obs->last_observe_ms) >= obs->observe_interval;
}

/**
 * spline_observer_push — Record a real observation.
 *
 * Call this after doing actual syscalls to read system state.
 *
 * @param obs     Observer state
 * @param values  Observed values in Q4.11 [n_channels]
 * @param now_ms  Current time in milliseconds
 */
static inline void spline_observer_push(
    SplineObserver *obs, const int16_t *values, uint32_t now_ms
) {
    for (int i = 0; i < obs->n_channels; i++) {
        spline_channel_push(&obs->ch[i], values[i], now_ms);
    }
    obs->last_observe_ms = now_ms;
}

/**
 * spline_observer_eval — Evaluate all channels at time t_ms.
 *
 * Zero syscalls. Pure integer arithmetic.
 *
 * @param obs     Observer state
 * @param now_ms  Query time
 * @param out     Output: interpolated values in Q4.11 [n_channels]
 */
static inline void spline_observer_eval(
    const SplineObserver *obs, uint32_t now_ms, int16_t *out
) {
    for (int i = 0; i < obs->n_channels; i++) {
        out[i] = spline_channel_eval(&obs->ch[i], now_ms);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_SPLINE_Q15_H */
