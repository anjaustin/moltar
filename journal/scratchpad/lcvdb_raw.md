# Raw Thoughts: LCVDB — What Is This Thing, Really?

## Stream of Consciousness

LCVDB is a vector database that fits in CPU cache. 48D int8 vectors, HNSW graph, NEON distance functions. It works — 436K QPS at N=32, 51K QPS at N=256 with 99.6% recall@5. The split storage design is clever: topology (graph edges) and vectors in separate arrays so graph traversal only touches the topology cache lines, vectors loaded on-demand for distance computation.

But here's what's nagging me: **we don't use it for anything.**

The ColBERT RAG pipeline uses brute-force MaxSim scoring over per-token 128D embeddings. LCVDB uses single-vector 48D int8 HNSW. These are architecturally incompatible. ColBERT won. The PRD demoted LCVDB from P0 to P2. The HNSW recall fix at large N is "not currently blocking."

So what is LCVDB for? Is it:
1. A component waiting for a use case that ColBERT replaced?
2. A research artifact that proved the cache-residency thesis?
3. A future component for a different retrieval path we haven't designed yet?
4. Dead weight?

Let me think about what it CAN do that ColBERT can't:
- Sub-microsecond search at small N. ColBERT's MaxSim is ~15ms for 10 chunks. LCVDB does 256 nodes in 19 us. That's 800x faster.
- The 48D int8 vectors fit in L1D. At N=256, the entire working set is 32 KB. ColBERT's 128D per-token embeddings are much larger per document.
- LCVDB scales logarithmically (HNSW). ColBERT's brute-force MaxSim scales linearly.

But ColBERT retrieves better because it has per-token resolution. A single 48D embedding per document loses massive amounts of semantic information compared to 15-30 128D token embeddings per document.

What if the two were combined? LCVDB as a **coarse filter** — search 10,000 documents in microseconds to get top-100, then ColBERT MaxSim over the top-100 for precise ranking. That's a standard two-stage retrieval architecture. LCVDB handles the "partition first" (The Laundry Method!), ColBERT handles "search within."

But then we need single-vector document embeddings. Where do those come from? ColBERT produces per-token embeddings. Could we mean-pool ColBERT's token embeddings into a single 128D vector, then PCA/random-project down to 48D, then quantize to int8? That's a lot of information loss but it might be good enough for coarse filtering.

Or we could increase LCVDB's vector dimension. 48D was chosen to fit in a cache line with padding. But we have 64 bytes per vec slot and only use 48. What if we went to 64D? Or restructured to use 128D across two cache lines? The distance function would need updating.

Wait, there's a more fundamental question: **do we even need 10,000+ documents on a phone?** The device has 6 GB RAM. After LFM2-1.2B (661 MB) + KV cache, we have maybe 4-5 GB free. That's a lot of document embeddings but is anyone actually going to have a 10,000-document knowledge base on a Moto G Power? The realistic use case is probably 50-500 documents. At that scale, ColBERT brute-force is fine.

So maybe LCVDB's real value is different. Not as a retrieval engine for RAG, but as a general-purpose semantic cache or memory layer. Think: conversation memory. Each turn of a conversation gets embedded and stored in LCVDB. When generating a new response, search the last 256 conversation turns for relevant context. That's exactly the scale where LCVDB shines — N=256, 19 us, fits in L1D.

Or: entity memory. Each entity mentioned in conversation gets an embedding. Quick lookup: "has the user mentioned this before?" at 436K QPS.

Or: the VDB could be the working memory of an agent. Fast semantic indexing of intermediate reasoning steps. The agent embeds each thought, stores it, and can retrieve related thoughts in microseconds during the next reasoning step.

The 48D int8 constraint is interesting here. It forces severe dimensionality reduction, which means the embedding must capture only the coarsest semantic features. That might actually be a feature, not a bug — coarse features are exactly what you want for broad topic matching, deduplication, or "have I seen something like this before?" queries.

What about the build path? Build is still O(N^2) brute-force for candidate collection. The beam search fix in build_ref.c uses HNSW search during insert, but... wait, I need to re-read the code. Let me check if the beam search fix was already applied.

*reads build_ref.c*

YES — build_ref.c already has beam search during insert (Phase 2 in the code). It's the proper HNSW Algorithm 2 with ef_construction=64 beam width. The comment says "Key fix over previous version: Phase 5 uses graph beam search (O(log N)) instead of brute-force O(N) scan." So this fix IS already in the code.

But the performance numbers in PERFORMANCE.md still show recall drop at N>=512. Did we re-measure after this fix? The test_recall.c exists. Have we run it with the new build_ref.c? The git log shows the beam search was added... but I'm not sure the benchmarks were re-run.

Actually wait — the PRD says "recall drops from 99.6% at N=256 to 84.8% at N=512." But build_ref.c has the beam search fix. So either:
1. The beam search fix didn't help enough, or
2. The benchmarks are stale (pre-fix numbers), or
3. The fix is in the code but something else is limiting recall

This is a significant question. If the beam search is already in and recall is still 84.8% at N=512, then the problem is deeper than candidate collection.

The scalar dot product in build_ref.c (`dot_i8`) vs the NEON dot product in search_ref.c (`lcvdb_dot_i8`) — could there be a discrepancy? The scalar and NEON should produce identical results for int8 dot products (no floating point, exact arithmetic). But worth checking.

The diversity heuristic in Phase 3 of build_ref.c — keep candidate C only if dot(query, C) > dot(S, C) for all selected S. This is the standard HNSW extended heuristic. But the threshold is >=, not >. If dot(S, C) == dot(query, C), the candidate is REJECTED. In a 48D int8 space with random vectors, ties are rare but not impossible. Is the heuristic too aggressive?

The backfill in Phase 5 (reverse edge connection) — when a neighbor is full, we collect M+1 candidates and diversity-select back to M. There's a backfill clause: "if diversity was too aggressive, backfill with remaining candidates." Good. But the diversity selection on reverse edges uses the NEIGHBOR'S vector as the center, not the query vector. That's correct for maintaining the neighbor's edge quality, but it means the new node might not get a reverse edge if it's in a dense region.

What about the visited bitset? It's stack-allocated: `uint8_t visited[(LCVDB_MAX_NODES + 8) / 8]`. LCVDB_MAX_NODES is 65535, so that's 8192 bytes on the stack. For N=1024, we only memset (new_id + 7) / 8 = 128 bytes. That's fine. But the stack allocation of 8 KB is a bit much for a microcontroller context.

The EFC (ef_construction) is 64. The BEAM_MAX is 256. The ef_search is also 64. These might be worth tuning. At N=1024, ef_construction=64 might not explore enough of the graph during insert to find good candidates. Standard HNSW papers recommend ef_construction >= 2*M. We have M=16, so ef_construction=64 is 4*M. That should be fine.

Actually, M=16 is large for 48 dimensions. Typical HNSW uses M=12-16 for hundreds of dimensions. At 48D, the space is less complex and M=8-12 might be sufficient. But larger M means more edges, better connectivity, more memory. With the cache-line budget, M=16 fills exactly 32 bytes of neighbor IDs (16 * 2 bytes). Clean.

One more thing: the PRNG. xorshift32 with `prng_state = 0x12345678`. Deterministic. The layer assignment uses bit masking: layer 0 with prob 7/8, layer 1 with prob ~1/64, layer 2 with prob ~1/512. That's a very flat distribution — few nodes in upper layers. Standard HNSW uses 1/ln(M) probability, which for M=16 is about 0.36. Our probability of layer >= 1 is 1/8 = 0.125. That means sparser upper layers, which means longer greedy descent paths. Might affect build quality at large N.

## Questions Arising

1. Is LCVDB still a live component or a research artifact?
2. Were recall benchmarks re-run after the beam search fix in build_ref.c?
3. What is the actual use case for LCVDB given ColBERT handles RAG?
4. Could LCVDB serve as a coarse first-stage filter in a two-stage retrieval pipeline?
5. Is the layer assignment probability too conservative (1/8 vs ~1/3)?
6. Is the diversity heuristic too aggressive at >= threshold?
7. Should M be reduced for 48D vectors?
8. What happens to the build quality when search uses NEON and build uses scalar — any divergence?
9. Is the 48D dimensionality fundamental or could we go to 64D or 128D?

## First Instincts

- LCVDB's future is as a semantic memory/cache layer, not a document retrieval engine
- The recall numbers in the docs might be stale — need to re-benchmark
- The layer probability distribution feels wrong for this graph density
- Two-stage retrieval (LCVDB -> ColBERT) is the architecturally clean integration path
- The code quality is good but the algorithm might be subtly misconfigured for 48D int8
