# Architecture

Moltar has four integrated systems — LLM inference, ColBERT retrieval, vector search, and a memory integration layer — all optimized for the MT6855V's memory hierarchy.

## Target Hardware

```
MediaTek MT6855V (Dimensity 930)
├── CPU Cluster
│   ├── 2x Cortex-A78 @ 2.2 GHz (big)     ← all inference runs here
│   │   ├── L1D: 64 KB, L1I: 64 KB
│   │   ├── L2: 256 KB per core
│   │   └── Cache line: 64 bytes
│   ├── 6x Cortex-A55 (little)              ← unused for inference
│   └── L3 (DSU): ~1 MB shared
├── GPU: PowerVR BXM-8-256
│   ├── OpenCL 3.0, 128-wide SIMD
│   ├── 28 KB local memory
│   └── 950 MHz max
└── DRAM: LPDDR4X @ 4266 MHz
    ├── 6 GB total
    └── 15.5 GB/s sustained (Android stopped)
```

ISA: ARMv8.2-a with DOTPROD (`vdotq_laneq_s32`), FP16, NEON. No I8MM, no SVE.

## LLM Inference Engine

Located in `research/llama.cpp/`. A fork of llama.cpp with custom kernels in `ggml/src/ggml-cpu/`.

### Quantization Formats

**Row-Scaled Q4_0** (`block_q4_0x4_rs`, 64 bytes = 1 cache line):
- 4 interleaved Q4_0 rows per block
- Pure-integer SDOT accumulation (no float until final reduction)
- Power-of-2 shift activation quantization via IEEE754 bit manipulation

**Row-Scaled Q8_0** (`block_q8_0x4_rs`, 128 bytes = 2 cache lines):
- 4 interleaved Q8_0 rows per block
- Same integer accumulation path

### GEMV Kernel (Token Generation)

Token generation is DRAM-bandwidth-bound. Each token reads the full weight matrix once.

```
Inner loop (NEON DOTPROD):
  vdotq_laneq_s32  v_acc, v_weights, v_activations, lane
```

- Processes 4 output elements per iteration (4 rows interleaved in block)
- Activation quantized once, cached across all weight blocks in a row
- Single `addv` horizontal reduction per output element

### GEMM Kernel (Prompt Processing)

Prompt processing has activation reuse across multiple tokens.

- 4 activation rows processed simultaneously
- Same weight block read serves all 4 activations
- 4x reduction in weight memory traffic vs sequential GEMV

### Dispatch Optimizations

- **Barrier-skip**: skip pthread barriers when only 1 thread has work
- **Graph dispatch fast path**: bypass scheduler overhead for simple graphs
- **Thread 1 fast-forward**: main thread starts compute before worker is signaled

### Modified Files

```
ggml/src/ggml-cpu/
├── repack.h              # block_q4_0x4_rs, block_q8_0x4_rs structs
├── repack.cpp            # repack functions, tensor_traits, forward_mul_mat dispatch
└── arch/arm/
    └── repack.cpp        # NEON DOTPROD GEMV/GEMM implementations
```

## ColBERT RAG Pipeline

Located in `research/colbert/`. On-device retrieval-augmented generation using ColBERT late-interaction embeddings.

### Pipeline Architecture

```
User query
    │
    ▼
┌──────────────────────┐
│  LFM2-ColBERT-350M   │  Embed query → per-token 128D float vectors
│  (Q4_0, 209 MB)      │  ~66 ms for 8-15 tokens
└──────────┬───────────┘
           │ quantize to int8
           ▼
┌──────────────────────┐
│  MaxSim Search       │  Brute-force over all document chunks
│  (NEON SDOT)         │  ~15 ms for 10 chunks
└──────────┬───────────┘
           │ top-k chunk text
           ▼
┌──────────────────────┐
│  LFM2-1.2B           │  Generate answer with retrieved context
│  (Q4_0, 661 MB)      │  ~21 tok/s
└──────────────────────┘
```

**Key constraint**: The two models run sequentially. ColBERT (209 MB) + LFM2-1.2B (661 MB) + KV cache would exceed available memory, so `moltar_rag.sh` runs embedding and generation as separate subprocess calls.

### ColBERT Late Interaction

Unlike single-vector embedding models, ColBERT produces **one 128D vector per token**. Standard scoring uses MaxSim:

```
score(Q, D) = Σ_q max_d (q · d)
```

For each query token embedding `q`, find the maximum dot product against all document token embeddings `d`, then sum across query tokens.

### Variance-Weighted Mean-Centered MaxSim

Standard MaxSim fails on small corpora because common tokens ("Moltar", "project", BOS) match well in every chunk, drowning out distinctive tokens. `moltar_rag.c` implements an improved scoring algorithm:

```
per_token[q][c] = max_j dot(q_i, chunk_c_j)       # Phase 1: decompose MaxSim
mean_q          = avg_c(per_token[q][c])            # Phase 2: per-token statistics
stdev_q         = stdev_c(per_token[q][c])
weight_q        = stdev_q / max(stdev)              # high variance = distinctive token
score(c)        = Σ_q weight_q * (per_token[q][c] - mean_q) / sqrt(n_doc_tokens_c)
```

1. **Mean-centering**: Common tokens that score similarly across all chunks contribute ~0 after subtraction.
2. **Variance weighting**: Distinctive tokens (high cross-chunk variance) get full weight; common tokens and BOS get near-zero weight.
3. **Length normalization**: `sqrt(n_doc_tokens)` compensates for longer documents having more matching opportunities.

This is equivalent to learned IDF weighting applied to late interaction — tokens with uniform match patterns are implicitly downweighted.

### MaxSim Implementation

**Index structure** (`colbert.h`):
- Per-document: array of 128D int8 token embeddings + token count
- Global index: array of document entries, manifest file mapping chunk IDs to text files

**NEON kernel** (`maxsim_neon.S`):
- `colbert_dot_i8`: 128D int8 dot product using 8x SDOT instructions
- `colbert_maxsim_i8`: full MaxSim scoring — iterates query tokens, finds max dot per document token, accumulates sum

**Quantization**: float32 embeddings → int8 via per-vector max-abs scaling (in `colbert.c`). Preserves relative ordering for ranking.

### Orchestration (`moltar_rag.sh`)

Shell script handling the full pipeline on device:
1. **`ingest`**: Chunk text files → embed with `llama-embedding --embd-output-format raw` → store `.txt` + `.emb` pairs
2. **`query`**: Embed query → `moltar_rag` MaxSim search → extract top-k chunk text → construct prompt → `llama-cli --single-turn` generation
3. **`demo`**: Run predefined queries to validate pipeline

### Files

```
research/colbert/
├── colbert.h           # Index structs, MaxSim API
├── colbert.c           # Init, quantize, add_doc, search
├── maxsim_neon.S       # NEON SDOT: colbert_dot_i8, colbert_maxsim_i8
├── test_colbert.c      # 4 correctness tests + benchmark
├── moltar_rag.c        # Variance-weighted MaxSim search tool
├── moltar_rag.sh       # Shell orchestrator (ingest, query, demo)
├── knowledge/
│   └── moltar.txt      # Sample knowledge base
└── Makefile            # Cross-compile with aarch64-linux-gnu-gcc -static
```

## L-Cache Vector Database

Located in `research/lcvdb/`. A cache-resident HNSW vector database designed for sub-25 us associative lookup. **Serves as the semantic working memory layer** — conversation turn recall, entity tracking, agent state. ColBERT handles long-term knowledge retrieval (different architecture, different timescale).

### Split Storage Layout

Three separate arrays, each with different access patterns:

```
Topology Array (hot during graph traversal):
  lcvdb_topo_t, 64 bytes per node (1 cache line), M=16
  ┌───────────────────────────────┬────┬────┬───────┬──────────┬──────────┐
  │   neighbors (u16 x 16)       │ nc │ ml │ flags │ payload  │ reserved │
  │          32 bytes             │ 1  │ 1  │  2    │    4     │   24     │
  └───────────────────────────────┴────┴────┴───────┴──────────┴──────────┘

Vector Array (touched only for distance computation):
  lcvdb_vec_t, 64 bytes per node (1 cache line)
  ┌──────────────────────────────┬──────────┐
  │ int8 vector (48 dimensions)  │ padding  │
  │          48 bytes            │ 16 bytes │
  └──────────────────────────────┴──────────┘

DB Header (1 cache line):
  lcvdb_t, 64 bytes
  ┌───────┬────┬────┬───┬───────────┬──────────┬───────────┬──────┬──────────┐
  │ count │ ep │ ml │ M │ topo_ptr  │ vec_ptr  │ max_nodes │ prng │ reserved │
  │  4B   │ 2B │ 1B │1B │    8B     │    8B    │    4B     │  4B  │   32B    │
  └───────┴────┴────┴───┴───────────┴──────────┴───────────┴──────┴──────────┘
```

### Why Split Storage

With the old unified layout (64 bytes/node with M=8), traversing the graph loaded full vectors even when only checking neighbor lists. With split storage:

- **Graph traversal** reads only topology: neighbor IDs + metadata, no vector data
- **Distance computation** loads vectors on-demand: only for candidates being scored
- At N=256: topology = 16 KB, vectors = 16 KB, total = 32 KB → fits L1D (64 KB)
- At N=1024: topology = 64 KB, vectors = 64 KB, total = 128 KB → fits L2 (256 KB)

### HNSW Algorithm

**Insert** (build_ref.c):
1. Assign random layer via geometric distribution (xorshift32 PRNG)
2. Greedy descent through upper layers to target layer
3. Beam search (ef_construction=64) on layer 0 to find candidates (O(log N))
4. Diversity selection: keep candidate C only if `dot(query, C) > dot(S, C)` for all selected S
5. Connect forward edges (new node -> selected neighbors)
6. Connect reverse edges with replacement when full

**Search** (search_ref.c):
1. Greedy descent through upper layers
2. Beam search on layer 0 with ef_search=64 beam width
3. Two-list approach: candidate pool (unordered, pick best) + result list (sorted)
4. Early termination: stop when best candidate worse than worst result
5. Skip deleted nodes (tombstone flag check)

**Distance** (distance.S):
- NEON int8 dot product for 48 dimensions
- 3 chunks of 16 elements: SMULL -> SADDLP/SADALP (widen to int32 per chunk to avoid int16 overflow)
- ADDV horizontal reduction
- ~12 cycles fully pipelined

### Files

```
research/lcvdb/
├── lcvdb.h           # Structs (lcvdb_t, lcvdb_topo_t, lcvdb_vec_t) + API
├── distance.S        # NEON: lcvdb_dot_i8, lcvdb_dot_i8_preloaded, lcvdb_dot_i8_batch4
├── init_ref.c        # C: lcvdb_init()
├── build_ref.c       # C: lcvdb_insert(), lcvdb_delete()
├── search_ref.c      # C: lcvdb_search() — uses NEON dot products from distance.S
├── test_lcvdb.c      # Correctness: distance, insert, vector integrity, payload, connectivity, search, delete
├── test_recall.c     # Recall@k at N=32..1024
├── test_bench.c      # Latency + throughput at N=32..1024
└── Makefile          # Cross-compile with aarch64-linux-gnu-gcc
```

Old assembly files (`init.S`, `build.S`, `search.S`) exist but are not used — they target the pre-split-storage layout.

## Memory Integration Layer

Located in `research/memory/`. Bridges LFM2 inference with LCVDB working memory via random projection of hidden states. This is the "glue" that makes the three-layer memory architecture work as a unified system.

### Data Flow (per conversation turn)

```
User input
    │
    ├──────────────────────────────┐
    │                              │ (if --rag enabled)
    ▼                              ▼
┌─────────────────────┐   ┌─────────────────────┐
│  LCVDB Search       │   │  ColBERT RAG        │  Shell out to llama-embedding
│  (23 us)            │   │  (~0.9 s cold)      │  + moltar_rag search
└──────────┬──────────┘   └──────────┬──────────┘
           │ context text            │ knowledge text
           ▼                         ▼
┌──────────────────────────────────────────────┐
│  ChatML Prompt Construction                   │
│  System + knowledge base + working memory     │
│  + recent turn + user input                   │
│  <|im_start|>system ... <|im_end|>            │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌─────────────────────┐
│  LFM2-1.2B Decode   │  embeddings=true → logits + hidden state
│  (21 tok/s)         │  llama_get_embeddings_ith(-1) → 2048D float
└──────────┬──────────┘
           │ 2048D float hidden state
           ▼
┌─────────────────────┐
│  Random Projection  │  Rademacher {-1,+1} matrix: 2048×48, 96 KB
│  (90 us)            │  max-abs quantization → 48D int8
└──────────┬──────────┘
           │ 48D int8
           ▼
┌─────────────────────┐
│  LCVDB Insert       │  Store vector + turn text for future retrieval
│  (~5 us)            │
└─────────────────────┘
```

### ColBERT RAG Subprocess Architecture

The ColBERT model (209 MB) and LFM2-1.2B (661 MB) cannot coexist in RAM. `moltar-agent` shells out via `system()` to run embedding and search as subprocesses:

1. **Write query to temp file** — user input is written to `/data/local/tmp/rag_query_agent.txt`, never passed on the command line. This prevents shell injection from metacharacters in user queries.
2. **`llama-embedding -f <file>`** loads ColBERT model via mmap, embeds query from file, writes `.emb` file, exits
3. **Background prefetch** — immediately after embedding exits, `cat model.gguf > /dev/null &` starts faulting LFM2 pages back into the page cache while MaxSim runs. Overlaps I/O with computation.
4. **`moltar_rag search`** reads query `.emb` + chunk `.emb` files, runs variance-weighted MaxSim, writes ranked results
5. **`moltar-agent`** reads chunk `.txt` files for top-K results, injects as knowledge context

The OS handles memory pressure: when ColBERT loads, LFM2-1.2B pages get evicted from page cache. When generation starts, LFM2 pages fault back in (~4-5s reload, partially hidden by prefetch). Total RAG overhead: ~8s per query (dominated by model swap I/O).

**End-to-end latency** (measured on device):

| Step | Time |
|------|------|
| ColBERT embedding (cold) | ~0.9 s |
| MaxSim search (10 chunks) | ~15 ms |
| LFM2-1.2B reload (mmap fault-in) | ~4-5 s |
| Prompt processing (~220 tokens) | ~2.4 s |
| Generation (~60 tokens) | ~2.9 s |
| **Total** | **~10.5 s** |

### Random Projection (`project.h`, `project.c`)

Johnson-Lindenstrauss projection from LFM2 hidden states to LCVDB vectors.

- **Matrix**: `int8[48][2048]`, Rademacher {-1,+1} entries, 96 KB
- **Deterministic**: xorshift32 PRNG with seed `0x4D4F4C54` ("MOLT")
- **Apply**: float×int8 matrix multiply → float[48] → max-abs quantize → int8[48]
- **Latency**: 90 us on A78 @ 2.2 GHz (negligible vs. LLM decode time)
- **Quality**: 100% cluster ordering preservation on structured data (JL guarantee)

### `moltar-agent` (`moltar-agent.c`)

The main integration binary. ~31 KB, dynamically linked against `libllama.so`, LCVDB compiled in statically.

- **Loads LFM2-1.2B** with `embeddings=true` via `llama_model_load_from_file` + `llama_init_from_model`
- **Sampling**: configurable temperature, top-k, top-p (default: greedy)
- **ChatML format**: `<|im_start|>system/user/assistant<|im_end|>` — matches LFM2 tokenizer chat template
- **Working memory**: LCVDB search injects top-3 related prior turns into system message
- **Knowledge retrieval**: optional ColBERT RAG via subprocess (`--rag` flag, `/rag` toggle). Variance-weighted MaxSim for retrieval quality. Query sanitization via temp file (no user input in shell commands).
- **Session persistence**: `--session FILE` saves and restores LCVDB graph state + turn text across restarts. Auto-save on exit, auto-load on startup. `/save` and `/load` interactive commands. Binary format: `[magic "MOLT"][version 1][n_turns][vdb scalars 32B][topo array][vec array][turns array]`.
- **Model prefetch**: background `cat` of LFM2 model file after ColBERT embedding exits, overlapping I/O with MaxSim computation
- **Prompt structure**: system message + knowledge base (RAG) + relevant prior conversation (LCVDB) + recent turn + current user input
- **Hidden state extraction**: `llama_get_embeddings_ith(ctx, -1)` after final token decode
- **Memory storage**: projects hidden state → inserts into LCVDB → stores turn text for retrieval

### Files

```
research/memory/
├── project.h           # moltar_proj_t struct, init/apply API
├── project.c           # Rademacher projection + max-abs quantization
├── moltar-agent.c      # Three-layer memory agent: LFM2 + LCVDB + ColBERT RAG
│                       #   session persistence, query sanitization, model prefetch
├── test_project.c      # 6 tests: init, determinism, JL, LCVDB roundtrip, latency
└── Makefile            # test_project (GNU static) + moltar-agent (NDK dynamic)
```

## Device Setup

### Boot Script

`/data/adb/service.d/moltar_perf.sh` (installed via Magisk):
- Stops Android framework on boot (`stop`)
- Sets big core governors to `performance`
- Frees ~4 GB/s DRAM bandwidth for inference

### Runtime

```bash
# Set governors (if not done by boot script)
su -c 'echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor'
su -c 'echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor'

# Stop Android framework
su -c 'stop'

# Pin to big cores
taskset c0 /data/local/tmp/test_lcvdb
```

## GPU

PowerVR BXM-8-256, confirmed working via OpenCL probe (`gpu_probe.c`). Int8 dot product kernel written and benchmarked (`gpu_bench.c` + `gpu_dot.cl`). Correct at all N.

GPU dispatch overhead is ~40 us, which makes it slower than CPU for standalone VDB queries. Useful for async search while CPU does LLM inference (dispatch overhead is free when overlapped with 12-50 ms token generation).
