# PRD-004: Unified Breathing Architecture

**Version:** 1.0  
**Date:** February 5, 2026  
**Status:** Ready for Implementation  
**Authors:** Research Team  
**Depends on:** PRD-001, PRD-002, PRD-003

---

## Executive Summary

This PRD synthesizes learnings from PRDs 001-003 and today's experimental findings into a unified architecture. The key innovations:

1. **Breathing Rhythm**: FFN and Equilibrium work as one system (5 steps/token)
2. **Niche-Based Evolution**: Temporal diversity preserved by construction
3. **Kuramoto Monitoring**: Order parameter tracks population health
4. **Hybrid Training**: Gradients for weights, evolution for dynamics

This addresses the failures we discovered:
- Fitness signals that were 10^6 too small → **Gradient-magnitude fitness**
- Equilibrium that never equilibrated → **Breathing rhythm**
- Evolution that collapsed diversity → **Niche-based selection**

---

## 1. Architecture Overview

### 1.1 The Breathing CfC Layer

```
Token arrives
    ↓
INHALE: FFN computes target state
    target = tanh(W_up @ x)
    ↓
BREATHE: State relaxes toward target (5 iterations)
    for k in 1..5:
        g = σ(gate_input + α·h + gate_bias)
        h = (1-g)·h·decay + g·target
    ↓
EXHALE: Output from equilibrated state
    output = W_down @ (SiLU(gate) × h)
```

### 1.2 Parameter Separation

| Parameter | Training Method | Why |
|-----------|-----------------|-----|
| W_up, W_gate, W_down | Gradient descent | Precise credit assignment |
| Embeddings | Gradient descent | Token representations |
| decay | Niche evolution | Temporal scale (slow to explore) |
| gate_bias | Niche evolution | Memory/reactivity balance |

### 1.3 Memory Hierarchy

```
Niche 0: decay ∈ [0.50, 0.65] → Fast (reactive)
Niche 1: decay ∈ [0.65, 0.75] → Medium-fast  
Niche 2: decay ∈ [0.75, 0.85] → Medium-slow
Niche 3: decay ∈ [0.85, 0.95] → Slow (memory)
```

Each layer has neurons in ALL niches. Evolution happens within niches, preserving diversity.

---

## 2. Mathematical Foundations

### 2.1 Breathing Equilibrium

With balanced gates (g ≈ 0.5) and decay d:

```
After k steps: h_k ≈ (0.5d)^k·h_0 + (1-(0.5d)^k)·target

For d=0.7, k=5: 0.5% old + 99.5% target (practical equilibrium)
For d=0.9, k=5: 1.8% old + 98.2% target (still good)
```

### 2.2 Kuramoto Order Parameter

Measures temporal diversity:

```
r × e^(iψ) = (1/N) × Σ e^(iθ_j)

where:
    θ_j = ω_j × t + φ(h_j)
    ω_j = 1 - decay_j  (natural frequency)
```

**Interpretation:**
- r → 0: Diverse (good hierarchy)
- r → 1: Synchronized (redundant, bad)

**Target:** r(t=50) < 0.15

### 2.3 Gradient-Magnitude Fitness

```
fitness_j = E[|∂L/∂output_j|]

High fitness = neuron strongly influences loss
Low fitness = neuron irrelevant to task
```

### 2.4 Niche-Based Selection

```
For each niche:
    survivors = top 50% by fitness within niche
    offspring = reproduce(survivors)
    
Global diversity = Σ niche_diversities
```

---

## 3. Implementation Specification

### 3.1 BreathingPopulation

```python
class BreathingPopulation(nn.Module):
    def __init__(self, config):
        # Trainable weights
        self.w_up = nn.Parameter(...)    # [n_neurons, embed_dim]
        self.w_gate = nn.Parameter(...)  # [n_neurons, embed_dim]
        self.w_down = nn.Parameter(...)  # [n_neurons, embed_dim]
        
        # Evolvable dynamics (buffers, not parameters)
        self.decay = buffer(...)         # [n_neurons], in [0.5, 0.95]
        self.gate_bias = buffer(...)     # [n_neurons], in [-1, +1]
        
        # State
        self.h = buffer(zeros(n_neurons))
        
        # Fitness tracking
        self.grad_accum = buffer(zeros(n_neurons))
        self.grad_count = buffer(zeros(1))
        
        # Niche assignment (fixed at init)
        self.niche = buffer(...)  # [n_neurons], values 0-3
```

### 3.2 Forward Pass (Breathing)

```python
def forward(self, x):
    # INHALE
    up = self.w_up @ x
    target = tanh(up)
    gate_input = self.w_gate @ x
    
    # BREATHE (5 steps, detached for efficiency)
    h = self.h.detach()
    for _ in range(5):
        g = sigmoid(gate_input.detach() + α*h + self.gate_bias)
        h = (1-g) * h * self.decay + g * target.detach()
    
    self.h = h.detach()
    
    # EXHALE (with gradients for last step)
    g_final = sigmoid(gate_input + α*h + self.gate_bias)
    h_final = (1-g_final) * h * self.decay + g_final * target
    
    output = self.w_down.T @ (silu(gate_input) * h_final)
    return output
```

### 3.3 Niche Evolution

```python
def evolve_niche(self, niche_idx):
    # Get neurons in this niche
    mask = (self.niche == niche_idx)
    indices = mask.nonzero().squeeze()
    
    # Sort by fitness within niche
    niche_fitness = self.fitness[indices]
    sorted_idx = argsort(niche_fitness, descending=True)
    
    n_survive = len(indices) // 2
    survivors = indices[sorted_idx[:n_survive]]
    casualties = indices[sorted_idx[n_survive:]]
    
    # Reproduce within niche
    for dst in casualties:
        parent = random.choice(survivors)
        
        # Clone dynamics (stay in same niche!)
        self.decay[dst] = self.decay[parent] + randn() * 0.02
        self.gate_bias[dst] = self.gate_bias[parent] + randn() * 0.1
        
        # Clamp to niche bounds
        niche_low = 0.5 + niche_idx * 0.1125
        niche_high = niche_low + 0.1125
        self.decay[dst] = clamp(self.decay[dst], niche_low, niche_high)
```

### 3.4 Kuramoto Tracking

```python
def compute_kuramoto(self, t=50):
    omega = 1 - self.decay  # Natural frequencies
    theta = omega * t       # Phases at time t
    
    complex_phases = exp(1j * theta)
    r = abs(complex_phases.mean())
    
    return r.item()
```

---

## 4. Training Loop

```python
def train_step(model, token_id, target_id, optimizer):
    # Forward
    logits = model.forward(token_id)
    loss = cross_entropy(logits, target_id)
    
    # Backward (gradients flow to weights, not dynamics)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Maybe evolve (every N tokens)
    model.maybe_evolve()
    
    return loss

def maybe_evolve(self):
    if self.tokens_seen % evolve_every == 0:
        for layer in self.layers:
            # Record pre-evolution metrics
            r = layer.compute_kuramoto()
            
            # Evolve each niche independently
            for niche_idx in range(4):
                layer.evolve_niche(niche_idx)
            
            # Verify diversity preserved
            r_new = layer.compute_kuramoto()
            assert r_new < 0.3, "Diversity collapsed!"
```

---

## 5. Validation Plan

### 5.1 Baseline Comparison

**Test:** Does breathing CfC beat a simple GRU?

```
Model A: BreathingCfC (4 layers, 128 neurons/layer)
Model B: GRU (1 layer, 512 hidden)

Same parameter count (~500K)
Same training data (WikiText-2, 50K tokens)
Same optimizer (AdamW, lr=1e-3)

Success: Model A perplexity ≤ Model B perplexity
```

### 5.2 Ablation Studies

| Experiment | Change | Expected Result |
|------------|--------|-----------------|
| No breathing | 1 step instead of 5 | Higher PPL |
| No evolution | Fixed dynamics | Higher PPL |
| Standard evolution | Global, not niche | Diversity collapses |
| No niche constraint | Evolution can cross niches | Diversity collapses |

### 5.3 Kuramoto Monitoring

Track r(t=50) throughout training:

```
Expected trajectory:
  Init: r ≈ 0.06 (random, diverse)
  Training: r stays < 0.15 (niche evolution preserves)
  
Red flag:
  r > 0.3 → diversity collapsing
  r increasing over time → evolution broken
```

### 5.4 Success Criteria

| Metric | Target | Validation |
|--------|--------|------------|
| PPL on WikiText-2 | < 500 after 50K tokens | Beats baseline |
| Kuramoto r | < 0.15 | Diversity preserved |
| Generation quality | Coherent phrases | Manual inspection |
| Speed | > 30 tok/s | Practical training |

---

## 6. Connection to PRDs 001-003

### 6.1 From PRD-001 (SRC-FFN)

**Inherited:**
- Per-neuron CfC dynamics
- Ternary weight potential
- Multi-scale temporal features

**Modified:**
- Decay range: [0.8, 0.999] → [0.5, 0.95] (faster settling)
- Gate bias range: [-4, 0] → [-1, +1] (balanced gates)
- Single step → 5 breathing steps

### 6.2 From PRD-002 (Equilibrium Propagation)

**Inherited:**
- Equilibrium concept
- Contraction property
- Fixed-point dynamics

**Modified:**
- Full EP (50 steps) → Breathing (5 steps)
- Nudged phase → Gradient descent on weights
- Continuous dynamics → Discrete steps

### 6.3 From PRD-003 (Evolving Populations)

**Inherited:**
- Neurons as organisms
- Fitness-based selection
- Emergence through evolution

**Modified:**
- Fitness = contribution → Fitness = gradient magnitude
- Global selection → Niche-based selection
- Evolve weights → Evolve only dynamics

---

## 7. Implementation Checklist

### 7.1 Phase 1: Core Architecture

- [ ] Implement `BreathingPopulation` with niche assignment
- [ ] Implement 5-step breathing forward pass
- [ ] Implement gradient-magnitude fitness accumulation
- [ ] Implement niche-based evolution
- [ ] Add Kuramoto order tracking

### 7.2 Phase 2: Validation

- [ ] Create GRU baseline with same parameter count
- [ ] Run head-to-head comparison on WikiText-2
- [ ] Run ablation studies (breathing, evolution, niche)
- [ ] Track Kuramoto throughout training

### 7.3 Phase 3: Analysis

- [ ] Analyze evolved hierarchy per layer
- [ ] Compare niche distributions across layers
- [ ] Test generation quality
- [ ] Profile speed bottlenecks

### 7.4 Phase 4: Documentation

- [ ] Update PRD-003 with final results
- [ ] Document what worked and what didn't
- [ ] Create reproducibility guide

---

## 8. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Doesn't beat GRU | Medium | High | Simplify architecture, focus on what works |
| Still too slow | Medium | Medium | Reduce breathing steps to 3, optimize |
| Niche evolution doesn't help | Medium | Medium | Compare with fixed hierarchy |
| Kuramoto not predictive | Low | Low | Use other diversity metrics |

---

## 9. The Core Hypothesis

**Claim:** A breathing CfC architecture with niche-based evolution will:
1. Learn faster than pure evolution (gradient-trained weights)
2. Be more expressive than fixed RNNs (evolved dynamics)
3. Maintain temporal diversity (niche constraints)
4. Reach equilibrium efficiently (breathing rhythm)

**Test:** Beat a GRU baseline on language modeling.

If this fails, we learn that the complexity isn't justified.
If this succeeds, we have a new architecture paradigm.

---

## 10. Conclusion

PRD-004 represents the synthesis of three research directions:
- **SRC-FFN** (PRD-001): The per-neuron CfC concept
- **Equilibrium Propagation** (PRD-002): The settling dynamics
- **Evolving Populations** (PRD-003): The emergent hierarchy

Plus the fixes discovered on February 5, 2026:
- Gradient-magnitude fitness (not contribution-based)
- Breathing rhythm (not single-step)
- Niche-based evolution (not global selection)
- Kuramoto monitoring (diversity health check)

The architecture is ready. The validation plan is clear. 

Time to crush it.

---

*End of PRD-004*
