# Synthesis: LFM2 Inference Optimization — The Clean Cut

## Verdict

**CLOSED: Integer path optimization for activation quantization.**

We are at **92% of theoretical DRAM bandwidth utilization**. The bottleneck is memory bandwidth, not compute. KleidiAI already fuses F32→int8 activation quantization into its kernel with negligible overhead (0.02% of MUL_MAT time).

The original question — "can we use integer math to avoid multiplying" — misidentified the problem. SDOT provides 4 int8 MACs per cycle; multiplies are free. The constraint is fetching operands from LPDDR4X at 6.6 GB/s effective.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                 LFM2-350M INFERENCE PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  F32 Activations ──┬──> KleidiAI pack_func_ex ──> int8 packed   │
│                    │         (fused quantization)                │
│                    │                  │                          │
│                    │                  v                          │
│  Q4_0 Weights ─────┴──> KleidiAI NEON SDOT kernel ──> F32 out   │
│  (pre-packed at load)                                            │
│                                                                  │
│  Time breakdown per token:                                       │
│    - Weight load from DRAM:  ~15.7 ms (theoretical minimum)     │
│    - Actual measured:        ~17 ms                              │
│    - Utilization:            92%                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Decisions

### Decision 1: Do NOT pursue W4A8 optimization
**Because:** Activation quantization already happens inside KleidiAI at 0.02% overhead. There's nothing to optimize.

### Decision 2: Do NOT pursue custom integer kernels
**Because:** KleidiAI is Arm's production-quality implementation. We measured 92% bandwidth utilization. Custom kernels cannot beat this without changing the memory access pattern.

### Decision 3: Ternary/Yinsen kernels WAIT for sparse models
**Because:** Post-hoc ternarization of Q4_0 destroys quality (proven Phase 5). The 2.2x speedup requires natively sparse-trained models that don't exist yet.

### Decision 4: PIVOT to algorithmic optimizations
**Because:** The only remaining lever is smarter token generation — speculative decoding amortizes memory loads.

---

## Next Phase: Speculative Decoding Investigation

### Concept
Use a fast draft model to generate N candidate tokens, then verify them in batch with the target model. If acceptance rate is high, we amortize weight loads across N tokens.

### Candidates

| Approach | Draft Model | Target Model | Expected Gain |
|----------|-------------|--------------|---------------|
| Cross-model | LFM2-350M | LFM2-1.2B | 2-3x if >70% accept |
| Self-speculative | LFM2-350M early-exit | LFM2-350M full | 1.5-2x |
| Medusa-style | Trained heads | LFM2-350M | Requires training |

### Probe Required
1. Measure acceptance rate of 350M → 1.2B drafting
2. Test if llama.cpp supports speculative decoding for LFM2
3. Profile batch verification cost vs serial generation

---

## Alternative: Lower Quantization (IQ2/IQ3)

### Concept
Sub-4-bit quantization reduces bytes per weight → less DRAM traffic.

| Quant | BPW | Size (350M) | Expected tok/s | Quality Impact |
|-------|-----|-------------|----------------|----------------|
| Q4_0 | 4.9 | 206 MB | 56 | Baseline |
| Q3_K | 3.4 | ~143 MB | ~80? | Minor |
| IQ3_XXS | 3.0 | ~125 MB | ~90? | Moderate |
| IQ2_XXS | 2.3 | ~96 MB | ~120? | Significant |

### Probe Required
1. Check KleidiAI support for IQ2/IQ3 (may fallback to slow path)
2. Convert LFM2-350M to IQ3_XXS, measure quality via perplexity
3. Profile actual throughput on device

---

## Success Criteria

- [x] Understand the actual bottleneck (DRAM bandwidth, 92% utilized)
- [x] Rule out W4A8 as optimization path (already done by KleidiAI)
- [x] Identify remaining levers (speculative decoding, lower quant, native sparsity)
- [ ] Probe speculative decoding feasibility
- [ ] Probe IQ3 performance on KleidiAI

---

## Summary Table

| Optimization Path | Status | Reason |
|-------------------|--------|--------|
| W4A8 activation quant | CLOSED | Already fused in KleidiAI, 0.02% overhead |
| Custom integer kernels | CLOSED | Can't beat 92% bandwidth utilization |
| Ternary/Yinsen | WAITING | Needs natively sparse models |
| Speculative decoding | OPEN | Next investigation |
| IQ2/IQ3 quantization | OPEN | Probe required |
| Core/thread tuning | CLOSED | Already optimal (2 A78, taskset) |
| Memory prefaulting | CLOSED | Already in fabric daemon |

---

## The Clean Cut

The wood has been cut. The grain was **memory bandwidth**, not compute. KleidiAI is the sharpest axe available for Q4_0 on Cortex-A78. 

To go faster, we must either:
1. **Load fewer bytes** (smaller quantization)
2. **Load bytes less often** (speculative decoding)
3. **Load different bytes** (natively sparse models)

The integer multiply question was answered: **SDOT makes multiplies free. The constraint is getting operands to the ALU, not what the ALU does with them.**

---

*"Give me six hours to chop down a tree, and I will spend the first four sharpening the axe."*

The axe is sharp. Now we consider: is this the right tree?
