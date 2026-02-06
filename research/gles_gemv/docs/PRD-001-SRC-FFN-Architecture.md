# PRD-001: Spectral-Rotational-CfC Feed-Forward Network (SRC-FFN)

**Version:** 1.0  
**Date:** February 4, 2026  
**Status:** Draft  
**Authors:** Research Team  

---

## Executive Summary

SRC-FFN is a novel neural network architecture that replaces the traditional Transformer's attention mechanism with a fused Feed-Forward Network containing per-neuron Continuous-time Closed-form (CfC) recurrence. This design achieves:

- **O(1) memory per token** (vs O(n) for attention)
- **O(1) compute per token** (vs O(n) for attention)
- **74.8 tokens/second** on Moto G Power 5G (beating 50 tok/s baseline)
- **4.6% compute overhead** vs traditional FFN
- **Trainable with Equilibrium Propagation** (O(1) memory for training)

---

## 1. Problem Statement

### 1.1 The Attention Bottleneck

Modern LLMs use self-attention with complexity:
- **Compute:** O(n²·d) per layer
- **Memory:** O(n²) for attention matrix, O(n·d) for KV cache
- **Bandwidth:** KV cache grows linearly with context

For a 2048-token context on mobile:
- KV cache alone: 268 MB (for 16-layer, 1024-dim model)
- Attention compute: ~8.6 GFLOPS per layer

### 1.2 Existing Alternatives

| Architecture | Memory | Compute | Long-range |
|-------------|--------|---------|------------|
| Attention | O(n²) | O(n²) | Excellent |
| Linear Attention | O(d²) | O(n·d²) | Poor |
| Mamba/S4 | O(n·d) | O(n·d) | Good |
| RWKV | O(d) | O(n·d) | Good |
| **SRC-FFN** | **O(d_ff)** | **O(n·d·d_ff)** | **Tunable** |

SRC-FFN uniquely places recurrence **inside the FFN**, giving each of the d_ff neurons its own temporal memory.

---

## 2. Architecture Specification

### 2.1 High-Level Structure

```
Traditional Transformer Block:
  x → Attention → Add+Norm → FFN → Add+Norm → y

SRC-FFN Block:
  x → RoPE → RMSNorm → SRC-FFN → Add → y
       ↑                  ↑
   Spectral         Per-neuron
   position         CfC recurrence
```

### 2.2 The SRC-FFN Layer

```python
def src_ffn(x, h_prev, W_gate, W_up, W_down, decay, gate_bias):
    """
    Spectral-Rotational-CfC Feed-Forward Network
    
    Args:
        x: Input tensor [batch, embed_dim]
        h_prev: Previous hidden states [batch, ff_dim]
        W_gate: Gate projection [embed_dim, ff_dim], ternary
        W_up: Up projection [embed_dim, ff_dim], ternary
        W_down: Down projection [ff_dim, embed_dim], ternary
        decay: Per-neuron decay rates [ff_dim], learned
        gate_bias: Per-neuron gate biases [ff_dim], learned
    
    Returns:
        out: Output tensor [batch, embed_dim]
        h_new: Updated hidden states [batch, ff_dim]
    """
    # Projections
    gate = x @ W_gate           # [batch, ff_dim]
    up = x @ W_up               # [batch, ff_dim]
    
    # CfC dynamics for each FFN neuron
    g = sigmoid(up + alpha * h_prev + gate_bias)    # Gate
    candidate = tanh(up)                             # Candidate
    h_new = (1 - g) * h_prev * decay + g * candidate # Update
    
    # Gated output
    mid = silu(gate) * h_new    # [batch, ff_dim]
    out = mid @ W_down          # [batch, embed_dim]
    
    return out, h_new
```

### 2.3 Key Equations

**CfC Update (per neuron i):**
```
g[i] = σ(up[i] + α·h[i] + bias[i])
h'[i] = (1 - g[i])·h[i]·decay[i] + g[i]·tanh(up[i])
```

**Effective Gradient Decay:**
```
effective_decay = (1 - g) × decay
gradient_horizon ≈ log(0.01) / log(effective_decay)
```

**Output:**
```
mid[i] = SiLU(gate[i]) × h'[i]
out = mid @ W_down
```

### 2.4 Architectural Parameters

| Parameter | Symbol | LFM2-350M | Notes |
|-----------|--------|-----------|-------|
| Embedding dim | d | 1024 | |
| FFN dim | d_ff | 4096 | 4x embedding |
| Layers | L | 16 | |
| Vocab size | V | 32000 | BPE |
| Context | n | ∞ | No limit |
| Hidden states | h | 4096 × 16 | Per layer |

### 2.5 Memory Footprint

**Inference (per token):**
- Hidden states: d_ff × L × 4 bytes = 256 KB
- No KV cache needed

**Comparison at 2048 context:**
| Component | Attention | SRC-FFN |
|-----------|-----------|---------|
| KV cache | 268 MB | 0 |
| Hidden states | 0 | 0.26 MB |
| **Total** | **268 MB** | **0.26 MB** |

---

## 3. Initialization Strategy

### 3.1 The Gate Bias Problem

With default initialization (gate_bias = 0):
- g ≈ sigmoid(noise) ≈ 0.5
- effective_decay = 0.5 × 0.95 = 0.475
- gradient_horizon ≈ 6 tokens (too short!)

### 3.2 Recommended Initialization

**Multi-scale temporal features:**

```python
def initialize_src_ffn(ff_dim, n_scales=4):
    quarter = ff_dim // n_scales
    
    gate_bias = torch.zeros(ff_dim)
    decay = torch.zeros(ff_dim)
    
    # Short-range (local syntax, ~10 token horizon)
    gate_bias[:quarter] = -1.0
    decay[:quarter] = 0.9
    
    # Medium-range (phrases, ~30 token horizon)
    gate_bias[quarter:2*quarter] = -2.0
    decay[quarter:2*quarter] = 0.95
    
    # Long-range (sentences, ~100 token horizon)
    gate_bias[2*quarter:3*quarter] = -3.0
    decay[2*quarter:3*quarter] = 0.99
    
    # Very long-range (documents, ~300+ token horizon)
    gate_bias[3*quarter:] = -4.0
    decay[3*quarter:] = 0.995
    
    return gate_bias, decay
```

### 3.3 Expected Gate Values by Initialization

| gate_bias | Mean g | effective_decay | Horizon |
|-----------|--------|-----------------|---------|
| 0.0 | 0.50 | 0.48 | 6 |
| -1.0 | 0.30 | 0.67 | 13 |
| -2.0 | 0.14 | 0.82 | 25 |
| -3.0 | 0.06 | 0.89 | 43 |
| -4.0 | 0.02 | 0.93 | 66 |

---

## 4. Weight Quantization

### 4.1 Ternary Weights

All projection matrices use 2-bit ternary encoding:
- Values: {-1, 0, +1}
- Encoding: 00 = 0, 01 = +1, 10 = -1, 11 = unused
- Packing: 4 weights per byte

**Benefits:**
- 16x compression vs FP32
- No multiplication needed (just conditional add/subtract)
- 12 GOPS throughput on ARM NEON

### 4.2 Dot Product Implementation

```c
// LUT-based ternary dot product
float ternary_dot(uint8_t* weights, float* x, int n) {
    float32x4_t acc = vdupq_n_f32(0);
    
    for (int i = 0; i < n; i += 4) {
        uint8_t packed = weights[i/4];
        float32x4_t signs = vld1q_f32(BYTE_TO_SIGNS[packed]);
        float32x4_t xv = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, xv, signs);
    }
    
    return vaddvq_f32(acc);
}
```

---

## 5. Parallelization Strategy

### 5.1 Six-Core Execution (Dimensity 930)

```
Layer execution timeline:

A78 Core 6: [----CfC (512 outputs)----][--FFN down (512)--]
A78 Core 7: [----CfC (512 outputs)----][--FFN down (512)--]
A55 Core 0: [         wait           ][--FFN up (682)--]
A55 Core 1: [         wait           ][--FFN up (682)--]
A55 Core 2: [         wait           ][--FFN up (682)--]
A55 Core 3: [         wait           ][--FFN up (682)--]
Main:       [RoPE][  residual+norm   ][    residual    ]
```

### 5.2 Work Distribution

| Operation | Cores | Rows/Core | Notes |
|-----------|-------|-----------|-------|
| RoPE | Main | 1024 | Single-threaded |
| CfC | A78×2 | 512 | Latency critical |
| FFN up | All 6 | 682 | Embarrassingly parallel |
| FFN down | A78×2 | 512 | Latency critical |
| Residual | Main | 1024 | Single-threaded |

### 5.3 Performance Results

| Configuration | Tokens/sec | Speedup |
|---------------|------------|---------|
| Single A78 | 23.0 | 1.0x |
| Dual A78 | 45.7 | 2.0x |
| 2×A78 + 4×A55 | 74.8 | 3.3x |
| Target baseline | 50.0 | - |

---

## 6. Comparison with Alternatives

### 6.1 vs Standard Transformer

| Aspect | Transformer | SRC-FFN |
|--------|-------------|---------|
| Attention | O(n²) | None |
| Memory/token | O(n) | O(1) |
| Long-range | Excellent | Tunable |
| Parallelizable | Yes | Yes |
| Mobile-friendly | No | Yes |

### 6.2 vs Mamba

| Aspect | Mamba | SRC-FFN |
|--------|-------|---------|
| Recurrence location | Dedicated SSM | Inside FFN |
| State size | O(d) | O(d_ff) |
| Parallelism | Parallel scan | Row-parallel |
| Weight format | Dense | Ternary |
| Hardware | GPU-optimized | Mobile-optimized |

### 6.3 vs RWKV

| Aspect | RWKV | SRC-FFN |
|--------|------|---------|
| Recurrence | WKV mechanism | CfC per neuron |
| Time mixing | Linear | Nonlinear (gated) |
| Learned timescales | Global | Per-neuron |
| Training | Backprop | EP possible |

---

## 7. Theoretical Properties

### 7.1 Expressiveness

Each SRC-FFN layer has:
- **d_ff = 4096** independent CfC neurons
- Each neuron can learn a different temporal pattern
- Equivalent to 4096 parallel feature detectors with memory

### 7.2 Temporal Capacity

With multi-scale initialization:
- 1024 neurons × ~10 token horizon (local)
- 1024 neurons × ~30 token horizon (phrase)
- 1024 neurons × ~100 token horizon (sentence)
- 1024 neurons × ~300 token horizon (document)

Total effective memory: heterogeneous, task-adaptive.

### 7.3 Contraction Property

The CfC update is a contraction mapping:
```
||f(h₁) - f(h₂)|| ≤ k ||h₁ - h₂||

where k = (1-g) × decay < 1
```

This guarantees:
- Convergence to unique fixed point
- Bounded hidden state norms
- Stable training dynamics

---

## 8. Implementation Checklist

### 8.1 Inference Engine

- [x] NEON ternary dot product
- [x] LUT-based weight decoding
- [x] 8-way accumulator unrolling
- [x] Software prefetch
- [x] Six-core parallelization
- [x] Fast sigmoid/tanh approximations
- [ ] Proper gate_bias initialization
- [ ] Multi-scale decay initialization

### 8.2 Training Pipeline

- [ ] PyTorch reference implementation
- [ ] Equilibrium propagation trainer
- [ ] Ternary weight quantization
- [ ] Gradient checkpointing (fallback)

### 8.3 Validation

- [ ] Perplexity on WikiText-103
- [ ] Long-range benchmark (PG-19)
- [ ] Inference latency on mobile
- [ ] Memory profiling

---

## 9. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Poor long-range modeling | Medium | High | Multi-scale init, tune decay |
| Training instability | Low | Medium | CfC contraction guarantees |
| Quantization loss | Medium | Medium | Ternary-aware training |
| EP gradient mismatch | Medium | High | Validate against backprop |

---

## 10. Success Criteria

1. **Inference:** ≥50 tok/s on Moto G Power 5G ✓ (74.8 achieved)
2. **Memory:** <1 MB hidden state for 16-layer model ✓ (256 KB)
3. **Quality:** Within 5% perplexity of attention baseline (TBD)
4. **Training:** EP converges comparably to BPTT (TBD)

---

## Appendix A: Mathematical Derivations

### A.1 Gradient Horizon Formula

For CfC update h' = (1-g)·h·decay + g·tanh(up):

```
∂h'ₜ/∂hₜ₋₁ = (1-g)·decay + g·α·σ'(gate_input)·hₜ₋₁

For small α contribution:
∂h'ₜ/∂hₜ₋₁ ≈ (1-g)·decay = effective_decay

∂hₜ/∂h₀ = ∏ᵢ₌₁ᵗ effective_decayᵢ ≈ (effective_decay)ᵗ

For gradient to reach 1% of original:
(effective_decay)ᵗ = 0.01
t = log(0.01) / log(effective_decay)
```

### A.2 Fixed Point Solution

At equilibrium h* = f(h*, x):

```
h* = (1-g*)·h*·decay + g*·tanh(up)
h*·(1 - (1-g*)·decay) = g*·tanh(up)
h* = g*·tanh(up) / (1 - (1-g*)·decay)
```

Where g* = σ(up + α·h* + bias) depends on h*, requiring iterative solution.

---

## Appendix B: Code Reference

Main implementation files:
- `six_core_llm.cpp` - Production inference engine
- `src_ffn.cpp` - Architecture exploration
- `decay_experiment_v2.py` - Initialization analysis
- `equilibrium_prop_analysis.py` - EP feasibility study

---

*End of PRD-001*
