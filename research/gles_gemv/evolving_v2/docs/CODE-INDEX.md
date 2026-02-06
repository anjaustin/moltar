# Code Index: Evolving Neural Populations

**Last updated:** 2026-02-05

---

## Core Architecture Files

### Original Implementation (broken fitness)

| File | Description | Status |
|------|-------------|--------|
| `vectorized_population.py` | Batched CfC neurons with evolution | Deprecated (fitness bug) |
| `model.py` | Full model assembly | Deprecated |
| `train.py` | Simple data training | Deprecated |
| `train_wikitext.py` | WikiText-2 training | Deprecated |

### Hybrid Implementation (gradient + evolution)

| File | Description | Status |
|------|-------------|--------|
| `hybrid_layer.py` | Trainable weights + evolvable dynamics | Working |
| `hybrid_model.py` | Full hybrid model | Working |
| `train_hybrid.py` | Hybrid training script | Working |

### Breathing Implementation (equilibrium rhythm)

| File | Description | Status |
|------|-------------|--------|
| `breathing_layer.py` | CfC with 5-step breathing rhythm | Working |
| `breathing_model.py` | Full breathing model | Working |

---

## Analysis Tools

| File | Description |
|------|-------------|
| `train_trajectory.py` | Training with trajectory tracking |
| `analyze_trajectory.py` | Delta Observer-style trajectory analysis |
| `analyze_checkpoint.py` | Checkpoint analysis tool |
| `prepare_wikitext.py` | WikiText-2 data preparation |

---

## Data Files

| File | Description |
|------|-------------|
| `wikitext2_tokens.pkl` | Cached WikiText-2 tokens (GPT-2 tokenizer) |

---

## Experiment Outputs

```
experiments/
├── wikitext_real/           # Original WikiText experiments (broken)
│   ├── checkpoint_e0_t20000.pt
│   └── checkpoint_e0_t40000.pt
├── trajectory_analysis/     # Trajectory tracking experiments
│   └── trajectory_gen*.json
└── hybrid_v1/              # Hybrid model experiments
    ├── checkpoint_t10000.pt
    └── checkpoint_t20000.pt
```

---

## Key Equations

### CfC Dynamics

```
h' = (1-g) × h × decay + g × tanh(up)
g = σ(up + α×h + gate_bias)
```

### Kuramoto Order Parameter

```
r × e^(iψ) = (1/N) × Σ e^(iθ_j)

where θ_j = ω_j × t + φ(h_j)
      ω_j = 1 - decay_j
```

### Breathing Equilibrium

With g ≈ 0.5 and decay d:
```
After k steps: (0.5×d)^k old + (1-(0.5×d)^k) target

decay=0.7, k=5: 0.5% old, 99.5% target (practical equilibrium)
```

---

## Configuration Defaults

### BreathingModelConfig

```python
vocab_size: int = 50257      # GPT-2 tokenizer
embed_dim: int = 128
n_layers: int = 8
n_neurons_per_layer: int = 256
n_breathe: int = 5           # Steps per token
decay_min: float = 0.5       # Fast neurons
decay_max: float = 0.9       # Slow neurons
gate_bias_min: float = -1.0  # Balanced gates
gate_bias_max: float = 1.0
evolve_every: int = 100
```

---

## Architecture Evolution

```
v1: Vectorized populations, fitness = contribution to target
    Problem: Fitness signal 10^-6, evolution is noise
    
v2: Hybrid (gradient weights + evolved dynamics)
    Fix: Fitness = gradient magnitude
    Problem: FFN/equilibrium imbalance (1 step vs 30 needed)
    
v3: Breathing (5-step equilibrium per token)
    Fix: Lower decay + balanced gates + breathing rhythm
    Problem: Evolution collapses diversity (Kuramoto revealed)
    
v4: Niche-based evolution (proposed)
    Fix: Evolve within temporal niches, preserve diversity
    Status: Designed, not yet implemented
```

---

## Next Implementation Tasks

1. Add niche-based evolution to `breathing_layer.py`
2. Add Kuramoto order tracking to trajectory
3. Implement proper baseline comparison (GRU)
4. Clean WikiText-2 data (remove formatting)
