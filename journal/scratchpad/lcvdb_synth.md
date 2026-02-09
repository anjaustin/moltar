# Synthesis: LCVDB — From Search Engine to Semantic Memory

## Reframing

LCVDB is not a document retrieval engine. ColBERT handles that. LCVDB is a **semantic register file** — a cache-resident data structure for sub-25 us associative lookup over small, dynamic collections. Its natural use cases are:

- **Conversation memory**: Embed each turn, recall related prior context during generation
- **Entity tracking**: "Have we discussed this entity before?" at 436K QPS
- **Agent working memory**: Index intermediate reasoning steps for self-referential retrieval
- **Semantic deduplication**: Detect near-duplicate inputs in real-time

These workloads share properties that match LCVDB's architecture: small N (64-512), continuous inserts, instant lookups, coarse matching sufficient.

## Architecture Decision: Conversation Memory Layer

```
User input
    │
    ├─── embed to 48D int8 ─────► LCVDB insert (working memory)
    │                                    │
    │                                    ▼
    ├─── embed to 128D (ColBERT) ──► MaxSim search (knowledge base)
    │                                    │
    │                                    ▼
    │                              Merge context:
    │                              - LCVDB: recent related turns
    │                              - ColBERT: relevant knowledge
    │                                    │
    │                                    ▼
    └───────────────────────────► LFM2-1.2B generate with merged context
```

**Key insight**: LCVDB and ColBERT serve different memory timescales. LCVDB is working memory (seconds to minutes, microsecond access). ColBERT is long-term memory (persistent knowledge, millisecond access). Both feed into the generation prompt.

## Immediate Actions (No Design Uncertainty)

### 1. Fix Documentation Numbers
The topo struct is 64 bytes/node (M=16), not 32 bytes. All memory budget tables are 2x too low. Fix in:
- PRD.md: LCVDB section
- PERFORMANCE.md: memory budget table
- ARCHITECTURE.md: split storage section
- lcvdb.h: header comments (already correct in struct definition, wrong in some comments)

### 2. Re-Benchmark Recall
Build, push, and run `test_recall` on device with the current build_ref.c (which has beam search). Compare with documented numbers. This resolves Node 5 — are the benchmarks stale?

### 3. NEON Dot Product in Build
In build_ref.c, replace:
```c
static int32_t dot_i8(const int8_t *a, const int8_t *b) {
    int32_t sum = 0;
    for (int i = 0; i < LCVDB_VEC_DIM; i++)
        sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}
```
With:
```c
#define dot_i8 lcvdb_dot_i8
```
One line. Free 3x speedup on build path. Zero risk (NEON and scalar produce identical int8 dot products).

## Deferred Actions (Require Design Decisions)

### 4. Layer Probability Fix (if recall is still low after re-benchmark)
Change PRNG layer assignment from:
```c
// Current: P(layer>=1) = 1/8 = 12.5%
if ((s & 0x7) == 0) { layer++; ... }
```
To standard HNSW:
```c
// Standard: P(layer>=1) = 1/ln(M) ≈ 1/2.77 ≈ 36% for M=16
// Approximate with: P(layer>=1) = 1/3
if ((s % 3) == 0) { layer++; ... }
```
Or use the exact formula: `layer = floor(-log(uniform) * mL)` where `mL = 1/ln(M)`.

### 5. Diversity Threshold Relaxation (if layer fix isn't enough)
Change `sc >= cscore` to `sc > cscore` in both forward diversity selection (Phase 3) and reverse edge replacement (Phase 5) of build_ref.c. Allows ties to pass through, producing denser graphs.

### 6. Embedding Pipeline for Conversation Memory
The open question: how to produce 48D int8 vectors from conversation turns without loading a model. Options:
- **Random projection from LFM2 hidden states**: During generation, LFM2-1.2B already computes hidden states (1536D float). Apply a fixed random matrix (1536x48, stored as int8) to project to 48D. Quantize to int8. Cost: one matrix multiply per turn, using the hidden state that's already computed. Zero additional model loading.
- **Mean-pool ColBERT embeddings**: If ColBERT already embedded the query, mean-pool its 128D token embeddings to 1x128D, then project to 48D. But this requires ColBERT to be loaded, which defeats the purpose.
- **Bag-of-words hash**: No model at all. Hash token IDs to 48D int8 via locality-sensitive hashing. Fast but terrible quality.

**Recommendation**: Random projection from LFM2 hidden states. It's free (the hidden states already exist during generation), requires no additional model, and captures the same semantic information the LLM is using. The projection matrix is 1536*48 = 73 KB — negligible.

## Success Criteria

### For the re-benchmark (immediate):
- [ ] `test_recall` runs on device with current code
- [ ] Numbers compared with PERFORMANCE.md
- [ ] Decision: are numbers stale or is recall genuinely degraded?

### For conversation memory integration (future):
- [ ] 48D int8 embeddings produced from LFM2 hidden states during generation
- [ ] Conversation turns inserted into LCVDB in <50 us
- [ ] Top-3 related prior turns retrieved in <25 us
- [ ] Retrieved context improves generation quality on multi-turn conversations
- [ ] Total memory overhead: <64 KB for 256 turns (fits L1D)

## What Surprised Me

The simplicity of the answer. LCVDB isn't broken and doesn't need to be fixed. It needs to be repositioned. The code is solid, the performance is extraordinary, and the architecture is correct. The problem was always the use case — trying to make it a document retrieval engine when it's a semantic register file. Once you see it as working memory, everything clicks: the small N, the cache residency, the coarse 48D embeddings, the microsecond latency. It was always a memory layer. We just called it a search engine.
