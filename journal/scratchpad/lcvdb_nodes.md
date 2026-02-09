# Nodes of Interest: LCVDB

## Node 1: The Identity Crisis
LCVDB was built as the retrieval engine for on-device RAG. ColBERT replaced it. The PRD demoted LCVDB from P0 to P2. It currently has no consumer.
**Why it matters**: A component without a use case accumulates tech debt. Either find a purpose or archive it.

## Node 2: Speed Asymmetry
LCVDB searches 256 nodes in 19 us. ColBERT MaxSim searches 10 chunks in 15 ms. That's ~800x faster. But ColBERT retrieves better because of per-token resolution.
**Why it matters**: Speed without quality is useless, but 800x is not a rounding error. There's a design space between "impossibly fast but coarse" and "good but slow."

## Node 3: The Two-Stage Architecture
Standard IR systems use coarse-then-fine retrieval. LCVDB could be the L1 filter (microseconds, thousands of documents) feeding ColBERT as the L2 reranker (milliseconds, dozens of documents). This is The Laundry Method applied to retrieval.
**Why it matters**: This gives LCVDB a clear architectural role and makes ColBERT scale beyond brute-force.

## Node 4: The Embedding Gap
LCVDB needs single-vector 48D int8 per document. ColBERT produces multi-vector 128D float per token. There's no bridge between them. Mean-pooling ColBERT token embeddings and projecting to 48D would lose information but might preserve coarse topic similarity.
**Why it matters**: Without a way to produce 48D int8 document embeddings, two-stage retrieval is theoretical.

## Node 5: The Recall Question
PERFORMANCE.md reports 84.8% recall@5 at N=512 and 94.4% at N=256. But build_ref.c already contains the beam search fix that was supposed to improve recall. Either: (a) these numbers are post-fix and the fix wasn't enough, (b) the numbers are stale pre-fix data, or (c) something else limits recall.
**Why it matters**: We don't know if the code matches the reported metrics. A re-benchmark is needed before any further work.

## Node 6: Layer Probability Distribution
The PRNG assigns layer >= 1 with probability 1/8 (~12.5%). Standard HNSW uses 1/ln(M) which for M=16 is ~36%. Our upper layers are 3x sparser than standard HNSW. This means fewer skip connections, longer greedy descent, potentially worse connectivity.
**Why it matters**: Could be a root cause of recall degradation at large N. The graph's highway system is underdeveloped.
**Tension with Node 5**: If the beam search fix is in and recall is still dropping, sparse upper layers could be the missing piece.

## Node 7: The Conversation Memory Use Case
At N=256, LCVDB fits entirely in L1D (32 KB). Perfect for: conversation turn memory, entity tracking, agent working memory, semantic deduplication. These are sub-second, sub-kilobyte workloads where ColBERT's model loading overhead is absurd.
**Why it matters**: This might be LCVDB's natural habitat — not document retrieval, but ephemeral semantic indexing.

## Node 8: Dimensionality Choice (48D)
48D was chosen to fit in a 64-byte cache line with 16 bytes padding. But 64D would also fit (64 bytes, zero padding). Or 128D across two cache lines (128 bytes per vec slot). Higher dimensions = better recall = slower distance computation.
**Why it matters**: The 48D choice was a hardware constraint, not a quality optimization. If the distance function is fast enough, we could increase dimensions and improve retrieval quality.
**Tension with Node 2**: More dimensions means slower search, reducing the speed advantage over ColBERT.

## Node 9: Scalar Build vs NEON Search
build_ref.c uses `dot_i8()` (scalar C). search_ref.c uses `lcvdb_dot_i8` (NEON from distance.S). Both should produce identical results for int8 arithmetic. But the build is ~3x slower than it needs to be for every distance computation during insert.
**Why it matters**: Build is currently called once, so speed doesn't matter for correctness. But if LCVDB becomes a dynamic index (inserts during conversation), build speed matters. And the dead ends list says "NEON assembly port of LCVDB search — constraint costs more than preloaded query saves" — but that was about search, not build. Using NEON in build (just calling lcvdb_dot_i8 instead of the scalar function) is trivial and untried.

## Node 10: M=16 for 48D
M=16 neighbors in a 48D space is generous. Standard HNSW guidance suggests M=12-16 for hundreds of dimensions. For 48D, M=8-12 might suffice, using less memory and fewer distance computations during build/search.
**Why it matters**: Reducing M from 16 to 8 would halve topology size (32B/node instead of 64B/node), fitting 2x more nodes in L1D. But it would hurt connectivity.
**Tension with Node 6**: Reducing M while also having sparse upper layers could compound connectivity problems.

## Node 11: The Diversity Heuristic Threshold
The diversity check uses `sc >= cscore` (reject if any selected neighbor is at least as close). This is strict. Standard implementations often use `sc > cscore` (reject only if strictly closer). The >= variant is more aggressive, which could over-prune edges.
**Why it matters**: At N=512+, if the graph is losing connectivity, overly aggressive pruning during both insert forward edges (Phase 3) and reverse edge replacement (Phase 5) could be the cause.
**Tension with Node 5**: Another potential contributor to recall degradation.

## Node 12: Stack Memory Pressure
The visited bitset is 8192 bytes (64K bits for LCVDB_MAX_NODES=65535). This is allocated on stack in both build_ref.c and search_ref.c. Combined with candidate/result arrays, each insert or search uses ~10-12 KB of stack. On Android, default thread stack is 1 MB, so this is fine. But it's wasteful when N << 65535.
**Why it matters**: Minor. Only relevant if LCVDB runs on a microcontroller or in a thread with small stack.

## Node 13: The Documentation Says 32B/node Topo, Code Says 64B/node
lcvdb.h defines `lcvdb_topo_t` as 64 bytes (1 cache line). But the older PRD text and some comments reference "32 bytes/node" topology. The struct has M=16 neighbors * 2 bytes = 32 bytes of IDs, plus 32 bytes of metadata/padding, totaling 64 bytes.
**Why it matters**: The documentation is inconsistent. PERFORMANCE.md says "Topo 8KB" for N=256 (which is 32B/node), but the actual struct is 64B/node, so it should be 16 KB. The earlier docs were written when M=8 and topo was actually 32B. This discrepancy propagated through all docs.
