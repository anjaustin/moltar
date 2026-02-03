# Synthesis: GPU Verdict & Next Directions

## The Clean Cut

**The PowerVR BXM-8-256 GPU on Dimensity 930 has no viable role in LFM2 inference.** This is a conclusive, data-backed finding from 4 probes and a full pipeline prototype. The GPU investigation is closed.

### Evidence Summary

| Probe | Finding | Implication |
|-------|---------|-------------|
| Probe A: Dispatch latency | 160us minimum (timeline), 386us cold | Floor exceeds target operation cost |
| Probe B: CPU+GPU overlap | 93.9% hidden, overlapped=780us | Misleading — tested synthetic overlap, not real dependency graph |
| Probe C: GPU matmul | 19x slower than CPU at ALL batch sizes | GPU matmul is dead, no crossover exists |
| Prototype: FFN pipeline | GPU SwiGLU adds +14.3% overhead | Critical path dependency blocks all overlap |
| Prototype: CPU activations | SwiGLU=25.6us, RMSNorm=1.1us | Activations are 1.7% of layer time — nothing to offload |

### Root Cause

The BXM-8-256 shares the LPDDR4X memory bus with the CPU. It has no independent bandwidth path. For memory-bandwidth-limited workloads (which ALL of LFM2 inference is), adding GPU traffic to the shared bus causes contention (Probe B sequential: 2011us, expected 1126us). The GPU can only help workloads that are compute-bound with data already in GPU-local memory — neither condition applies here.

---

## Three Directions Forward

### Direction A: Sustained Performance Characterization (RECOMMENDED NEXT)

**What:** Measure tok/s degradation over 5-minute sustained runs for all 3 models. Characterize thermal throttling curve. Test daemon CfC controller's ability to maintain throughput.

**Why:** We've never measured sustained performance. Peak=44 tok/s, but real users run multi-minute conversations. If thermal throttling drops performance to 30 tok/s after 2 minutes, the daemon's adaptive scheduling becomes the primary value proposition.

**Probes needed:**
1. Thermal endurance probe: run 5-min continuous generation, sample tok/s every 10 seconds
2. Compare: daemon vs bare taskset under sustained load
3. Measure: CPU frequency over time (governor response to thermal)
4. Test: CfC controller predicting and reacting to thermal events

**Effort:** Low (reuse existing infrastructure, just longer runs)
**Potential impact:** Defines the daemon's real-world value proposition

### Direction B: Memory Bandwidth Optimization

**What:** Investigate why we achieve only 51% of theoretical LPDDR4X bandwidth during matvec. Identify and close the gap.

**Why:** Matvecs are 97.2% of compute. Even a 10% bandwidth improvement = 10% throughput improvement across ALL models.

**Probes needed:**
1. Software prefetch probe: add `prfm pldl1keep` N cache lines ahead of the matvec load stream. Measure impact.
2. Weight padding probe: pad Q4_0 blocks from 18→32 bytes for cache-line alignment. Measure bandwidth vs compute tradeoff.
3. KleidiAI disassembly: examine llama.cpp's actual SDOT loop to understand existing prefetch/scheduling.

**Effort:** Medium (requires careful NEON assembly work)
**Potential impact:** +10-20% throughput if bandwidth gap is closeable. Huge.

### Direction C: Multi-Model Adaptive Daemon

**What:** Extend the daemon to manage multiple model tiers, auto-selecting based on system state.

**Why:** Product differentiation. No other inference daemon dynamically switches models based on thermal/memory/battery state. This makes the fabric a system-level intelligence layer, not just a core pinner.

**Implementation:**
1. CfC controller observes: thermal trend, available RAM, battery state
2. Outputs: recommended model tier (350M/700M/1.2B)
3. Daemon pre-loads next-tier model when switch is predicted
4. Expose a simple API: `int trix_recommended_model(void)` → 0/1/2

**Effort:** Medium (CfC weight tuning, model preloading logic)
**Potential impact:** Unique product feature, but requires application integration

---

## Key Decisions

1. **GPU path: CLOSED.** No further GPU investigation for this device. All GPU probes and shaders preserved as reference.
2. **Direction A first** — it's the lowest effort with the most immediate insight about the daemon's real-world value.
3. **Direction B if A reveals thermal headroom** — if sustained performance stays near peak (no throttling), then bandwidth optimization is the next throughput lever.
4. **Direction C is a product decision** — depends on where Tripp wants the project to go.

## Success Criteria

- [ ] Sustained performance curve measured for all 3 models (5-min runs)
- [ ] Thermal throttling onset time identified
- [ ] Daemon vs bare taskset delta under sustained load quantified
- [ ] Decision: pursue bandwidth optimization or multi-model intelligence

## Revised Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TriX Fabric v0.4 (CURRENT)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  cpu6 (inference)          cpu7 (inference + daemon)             │
│  ┌────────────────┐       ┌────────────────┐                    │
│  │ llama.cpp -t 2 │◄─────►│ llama.cpp -t 2 │                    │
│  │ KleidiAI SDOT  │       │ KleidiAI SDOT  │                    │
│  └────────────────┘       └────────────────┘                    │
│         │                        │                               │
│         └────────┬───────────────┘                               │
│              LPDDR4X bus (~6.6 GB/s measured)                    │
│                  │                                               │
│         ┌────────┴────────┐                                      │
│         │  Model weights  │  ← mlock'd, pre-faulted             │
│         │  (209-663 MB)   │                                      │
│         └─────────────────┘                                      │
│                                                                  │
│  ┌─────────────────────────────────┐                             │
│  │  CfC Q15 Controller (10-100Hz)  │                             │
│  │  • Core pinning                 │                             │
│  │  • Memory management            │                             │
│  │  • Adaptive tick rate           │                             │
│  │  • [NEW: thermal prediction]    │                             │
│  │  • [NEW: model tier selection]  │                             │
│  └─────────────────────────────────┘                             │
│                                                                  │
│  GPU: PowerVR BXM-8-256                                         │
│  └── IDLE (conclusively unviable for this workload)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Files Produced in GPU Investigation (for reference)

| File | Status |
|------|--------|
| `probes/vk_persistent_dispatch.c` | Probe A — DONE, 160us timeline |
| `probes/gpu_pipeline_probe.c` | Probe B — DONE, 93.9% hidden (misleading) |
| `probes/gpu_matmul_probe.c` | Probe C — DONE, GPU 19x slower |
| `probes/ffn_pipeline_proto.c` | Prototype — DONE, GPU adds +14.3% |
| `probes/swiglu.comp` / `.spv` | SwiGLU shader — works, not useful |
| `probes/matmul_f32.comp` / `.spv` | FP32 matmul shader — works, not useful |
| `probes/q4_matmul.comp` / `.spv` | Q4_0 matmul shader — works, not useful |
| `probes/rmsnorm.comp` / `.spv` | RMSNorm shader — works, not useful |
