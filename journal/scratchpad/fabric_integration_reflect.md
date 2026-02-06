# REFLECT: Patterns in Fabric Integration

## Pattern A: Threshold Effects
Multiple nodes point to thresholds that change behavior:
- Node 1: 16 MB size threshold for uncached benefit
- Node 7: Context length scaling changes KV cache size
- Node 15: Crossover point question

**Insight**: Performance optimization isn't linear. There are regime changes where different strategies win. Finding these thresholds is key.

## Pattern B: Implementation-Hypothesis Gap
- Node 2: Correct implementation, wrong test
- Node 5: Allocation done, prefetch not integrated
- Node 13: Phase 2 requirements list

**Insight**: Successfully building something doesn't validate the hypothesis. The test conditions must match the hypothesis conditions.

## Pattern C: Hidden Dominants
- Node 3: SIMD flags dominate (47% impact)
- Node 11: KleidiAI GEMM optimization dominates
- Node 9: Memory bound but sequential access

**Insight**: The obvious optimization target (memory) might be secondary to less obvious factors (instruction selection, access pattern).

## Pattern D: Assumption Failures
- Node 4: offload flag behavior mismatch
- Node 8: Sparse attention in hybrid model
- Node 12: Unmeasured variance

**Insight**: Every assumption needs verification. The codebase behavior differs from documentation/expectation.

## Pattern E: Tradeoff Awareness
- Node 6: Variance vs throughput
- Node 7: Context length vs RAM
- Node 15: Validity depends on workload

**Insight**: There's no universal "better" - only "better for this use case". Optimization choices depend on requirements.

## Pattern F: Infrastructure Readiness
- Node 10: dma_heap integration clean
- Node 14: Defensive design complete
- Node 2: Implementation correct

**Insight**: The infrastructure work is done well. The issue is workload mismatch, not implementation bugs.

## Cross-Pattern Synthesis

### The Regime Model
Performance behavior exists in regimes:
1. **Small KV regime** (<16 MB): Cached wins, hardware prefetch excellent
2. **Medium KV regime** (16-256 MB): Uncached may win, depends on access pattern
3. **Large KV regime** (>256 MB): Memory pressure dominates, swap/OOM issues

Current test is in regime 1. Need to test regime 2.

### The Hierarchy of Impacts
1. **Instruction selection** (SIMD flags): 47% impact
2. **Algorithm choice** (KleidiAI GEMM): ~30% impact (estimated)
3. **Memory type** (cached vs uncached): 5% impact (measured)
4. **Prefetch strategy** (LITTLE cores): 4% impact (from probes)

Memory optimization is real but lower priority than instruction optimization.

### The Validity Boundary
The fabric hypothesis is valid within boundaries:
- KV cache size > 16 MB
- Random access pattern (not sequential)
- Variance matters for application
- LITTLE cores available for prefetch

Current test violates first three boundaries.

## Key Question Emerging
Should we pivot to testing with larger context, or recognize that for typical mobile use cases (short conversations), the fabric approach doesn't help?

The answer depends on target use case:
- **Chat assistant** (short context): Fabric not needed
- **Document analysis** (long context): Fabric might help
- **Real-time systems** (variance sensitive): Fabric might help despite speed cost
