# Changelog

## 44ac2ab — Knowledge base expansion (2026-02-09)

### Added
- **4 new knowledge documents** in `research/colbert/knowledge/`:
  - `arm_architecture.txt` — Cortex-A78, NEON, SDOT, cache hierarchy, LPDDR4X, DSU, Dimensity 930, DVFS (10 paragraphs)
  - `llm_quantization.txt` — Q4_0, Q8_0, GGUF, mixed quant, activation quant, speculative decoding, KV cache, perplexity (10 paragraphs)
  - `vector_search.txt` — HNSW, cosine similarity, ANN, ColBERT, product quantization, embeddings, RAG, late interaction, IVF (10 paragraphs)
  - `android_linux.txt` — Android kernel, mmap, page cache, zygote, taskset, Magisk, CPU governors, OOM killer, ADB, Termux (10 paragraphs)
- **50 total chunks** across 5 knowledge domains (up from 10 chunks in 1 document)

### Changed
- **RAG_TOP_K bumped from 2 to 3** in both `moltar-agent.c` and `moltar-server.c` for better coverage with larger corpus

### Measured
- MaxSim search at 50 chunks: **~92 ms** (under 100 ms target)
- Cross-domain retrieval verified: distinctive queries retrieve correct domain chunks

## 7cda8d5 — HTTP server + web chat UI (2026-02-09)

### Added
- **`moltar-server`** (`research/memory/moltar-server.c`) — single-threaded HTTP server embedding the full three-layer memory agent (LFM2 inference + LCVDB working memory + ColBERT RAG + session persistence) in one 37 KB binary
  - `GET /` — embedded dark-theme web chat UI (HTML/CSS/JS inline in C string, 3050 bytes)
  - `POST /api/chat` — `{"message":"..."}` → `{"response":"...","turn":N}`
  - `POST /api/ingest` — `{"file":"..."}` or `{"text":"..."}` → `{"chunks":N}`
  - `GET /api/status` — `{"turns":N,"rag_chunks":N,"rag":bool,"memory":bool}`
  - `POST /api/rag` — toggle RAG on/off
  - `POST /api/save` / `POST /api/load` — session persistence
  - CORS headers (`Access-Control-Allow-Origin: *`), Content-Length-aware body reading, SIGINT/SIGTERM clean shutdown
- **Termux boot script** (`moltar_boot.sh`) — auto-starts server on device boot with performance governors, RAG, and session persistence
- **Makefile `server` target** — `make server` builds `moltar-server` with NDK (same build flags as `moltar-agent`)

### Verified on Device
- `GET /api/status` → `{"turns":0,"rag_chunks":11,"rag":true,"memory":true}`
- `POST /api/chat` with RAG retrieval generates coherent response about Moltar, turn counter increments
- `POST /api/save` → `{"success":true,"turns":2}`
- `GET /` serves 3050 bytes of web UI HTML
- Follow-up chat demonstrates working memory (references prior turn)

## d7a6320 — Knowledge ingestion pipeline (2026-02-09)

### Added
- **`moltar_rag ingest`** command in `moltar_rag.c` — reads text file, splits by paragraph (double newline), min 80 chars, max 1600 chars with word-boundary splitting, appends to existing index, updates manifest
- **`/ingest <file>`** and **`/learn <text>`** commands in `moltar-agent.c` — ingest from file or inline text, auto-enable RAG, prefetch LLM model back

### Verified on Device
- Standalone ingest: 3 paragraphs → 3 chunks
- Append mode preserves existing index
- `/learn` from agent adds chunk 10, RAG search finds it ranked #1 for distinctive query

## 5f31b55 — Code hardening (2026-02-09)

### Fixed
- **Heap-allocated `per_token` array** in `moltar_rag.c` — was 64 KB on stack, now `malloc`'d with error handling + `free`
- **`strncpy` → `snprintf`** in `moltar-agent.c` — 5 call sites now guarantee null-termination
- **`posix_memalign` error checking** — both calls in `moltar-agent.c` now bail cleanly on failure

## 65810c4 — Comprehensive documentation update (2026-02-09)

### Changed
- Updated all 5 core docs (README.md, PRD.md, ARCHITECTURE.md, PERFORMANCE.md, CHANGELOG.md)
- Added three-layer memory diagram, session persistence, variance-weighted MaxSim, prefetch, query sanitization
- Corrected RAG latency tables (standalone ~2s vs agent ~10-14s with model swap)
- 285 insertions across 5 files

## 7445fc1 — Persistent session memory (2026-02-09)

### Added
- **Persistent session memory** (`moltar-agent.c`) — save/load LCVDB state + turn text across agent restarts
  - `--session FILE` CLI flag: auto-load on startup, auto-save on exit
  - `/save` and `/load` interactive commands for manual session management
  - Binary format: `[magic "MOLT"][version 1][n_turns][vdb scalars 32B][topo array][vec array][turns array]`
  - `session_save()` and `session_load()` functions with magic/version validation

### Verified on Device
- Two-session test: conversation with name/activity, quit, restart, agent correctly recalled "I'm Austin, and I'm building Moltar" from prior session
- Turn counter continues across sessions (1→2 in session 1, 3 in session 2)
- Auto-save on exit confirmed: `[moltar] Session saved: 2 turns -> /data/local/tmp/session.bin`

## 2623788 — Background LFM2 model prefetch (2026-02-09)

### Added
- **Background prefetch** in `rag_retrieve()` — after ColBERT embedding subprocess exits, starts `cat model.gguf > /dev/null &` to fault LFM2 pages back into page cache while MaxSim search runs
- Overlaps I/O (model reload) with computation (MaxSim search)

### Changed
- `rag_retrieve()` signature now accepts `const char *llm_model_path` parameter

### Measured
- RAG-enabled query latency: ~13.6 s with Android running (model swap I/O partially hidden)
- Without RAG: ~5.6 s

## b35b22c — Query sanitization (2026-02-09)

### Fixed
- **Shell injection vulnerability** in `moltar-agent.c` — user query was passed directly into `system()` via `-p "%s"`. Shell metacharacters (`"`, `$`, `` ` ``, `\`) could break the command or cause injection.
- **Fix**: Query written to temp file (`/data/local/tmp/rag_query_agent.txt`), passed via `-f <file>` to `llama-embedding`. No user input ever touches the shell command string.
- Tested with `"`, `$HOME`, `` `whoami` `` in query — works correctly.

## 224a215 — Variance-weighted mean-centered MaxSim (2026-02-09)

### Added
- **Variance-weighted MaxSim** in `moltar_rag.c` — fixes RAG retrieval quality on small corpora
  - Phase 1: Decompose MaxSim into per-token per-chunk max-dot scores
  - Phase 2: Compute per-token mean and stdev across chunks
  - Phase 3: Score = `Σ (stdev/max_stdev) * (score - mean) / sqrt(n_doc_tokens)`
  - Common tokens (uniform scores) contribute ~0; distinctive tokens get full weight
  - Length normalization via `sqrt(n_doc_tokens)` compensates for longer documents

### Measured
- Speed query (want chunk 1): #3 → **#1**
- ColBERT query (want chunk 5): #1 → #1 (bigger margin)
- GPU query (want chunk 6): #4 → #3
- Agent now answers "21 tokens per second" (correct) instead of hallucinating

## af767fd — Documentation update for ColBERT RAG (2026-02-09)

### Changed
- **PRD.md** — marked ColBERT RAG integration as DONE, added end-to-end latency breakdown table (10.5s total), updated completed milestones
- **ARCHITECTURE.md** — added subprocess architecture section, ColBERT RAG pipeline diagram, model prefetch description, `--rag` flag documentation

## e65a47d — ColBERT RAG integration in moltar-agent (2026-02-09)

### Added
- **ColBERT RAG in moltar-agent** — knowledge retrieval via subprocess calls to `llama-embedding` and `moltar_rag`
  - `--rag` / `--no-rag` CLI flags
  - `/rag` toggle and `/rag status` interactive commands
  - `rag_retrieve()` function: embed query → MaxSim search → read chunk text → inject into prompt
  - `rag_config_t` struct with configurable ColBERT model, embedding binary, search binary, index directory, top-K
  - Manifest loading (`rag_load_manifest`) for chunk count

### Verified on Device
- RAG-grounded factual responses (e.g. "128-dimensional per-token embedding model released by Liquid AI")
- Working memory + RAG coexistence confirmed
- End-to-end latency: ~10.5 s (ColBERT embed ~0.9s + MaxSim ~15ms + model swap ~4-5s + prompt ~2.4s + generation ~2.9s)

## 763ee59 — moltar-agent: three-layer memory integration (2026-02-09)

### Added
- **`moltar-agent`** (`research/memory/moltar-agent.c`) — multi-turn LLM with three-layer memory
  - Links against `libllama.so` (LFM2 inference) + LCVDB (semantic working memory, compiled statically)
  - Random projection of LFM2 hidden states (2048D float → 48D int8) via Rademacher matrix
  - Per-turn cycle: LCVDB search → build ChatML prompt → LFM2 decode → extract hidden state → project → insert
  - `/memory` interactive command for working memory status
  - Configurable: `-t` threads, `-c` context, `-n` max tokens, `--temp`, `--top-k`, `--top-p`, `--no-memory`
- **Projection layer** (`project.h`, `project.c`) — Johnson-Lindenstrauss random projection
  - Rademacher {-1,+1} matrix, 96 KB (2048×48), deterministic seed 0x4D4F4C54
  - 90 us latency on A78 @ 2.2 GHz
  - 100% cluster ordering preservation in JL tests
- **Projection tests** (`test_project.c`) — 6 tests: init, determinism, discrimination, JL distance, LCVDB roundtrip, latency
- **Makefile** — `make agent` (NDK dynamic) + `make all` (GNU static test binary)

### Verified on Device
- 3-turn conversation with name recall, activity recall, and contextual responses
- Total memory overhead per turn: ~113 us (90 us projection + 23 us search)

---

## [Previous — ColBERT RAG Pipeline + Cache Fix]

### Added
- **ColBERT RAG pipeline** (`research/colbert/`) — on-device retrieval-augmented generation using LFM2-ColBERT-350M for late-interaction embeddings and LFM2-1.2B for generation
  - `colbert.h`, `colbert.c` — ColBERT index: 128D int8 token embeddings, MaxSim scoring, top-k heap search
  - `maxsim_neon.S` — NEON SDOT assembly for 128D int8 dot products and full MaxSim scoring
  - `test_colbert.c` — 4 correctness tests + benchmark (all pass on device)
  - `moltar_rag.c` — Search binary: parses raw embeddings, runs MaxSim, outputs ranked results
  - `moltar_rag.sh` — Shell orchestrator: `ingest`, `query`, `demo` commands
  - `knowledge/moltar.txt` — Sample knowledge base
- **LFM2-ColBERT-350M model conversion** — HuggingFace download, GGUF F16 conversion, Q4_0 quantization (209 MB)

### Fixed
- **Activation quantization cache bug** — cache in `repack.cpp` keyed on `src1->data` pointer. The llama.cpp allocator reuses buffer addresses for different tensors, causing stale quantized activations to be used. All models produced gibberish output despite correct speed measurements. Fixed by removing the cache entirely.
  - Corrected benchmarks: LFM2-350M Q4_0 77.1 (was 80.1), Q8_0 40.9 (was 41.7), LFM2-1.2B Q4_0 21.3 (was 21.7)
  - 2-4% throughput cost — previous numbers were inflated by the bug
- **`--no-repack` workaround** removed from RAG pipeline after cache fix

### Changed
- **LCVDB reframing** — Lincoln Manifold analysis (journal/scratchpad/lcvdb_{raw,nodes,reflect,synth}.md) identified LCVDB as a semantic working memory layer, not a document retrieval engine. ColBERT handles knowledge retrieval; LCVDB handles conversation memory, entity tracking, agent state.
- **Memory budget correction** — all docs referenced 32B/node topology (from M=8 era). Actual struct is 64B/node (M=16). All memory tables corrected: N=256 is 32 KB total (was 24 KB), N=1024 is 128 KB (was 96 KB). Conclusions unchanged — still fits in cache.

### Verified on Device
- ColBERT RAG end-to-end: embed query (~66 ms) + MaxSim search (~15 ms) + LLM generate (~2 s for short answer)
- 3 demo queries producing coherent, context-grounded answers
- All 4 ColBERT correctness tests pass
- LLM inference produces correct (non-gibberish) output after cache fix

---

## Previous — L-Cache VDB Split Storage

### Added
- **L-Cache VDB split storage redesign** — separate topology (64B/node, M=16) and vector (64B/node) arrays, uint16 IDs (max 65534 nodes), tombstone delete, payload IDs
- **C reference implementations** — `init_ref.c`, `build_ref.c`, `search_ref.c` replace old assembly for split storage layout
- **VDB test suite** — `test_lcvdb.c` (7 correctness tests), `test_recall.c` (N=32..1024), `test_bench.c` (latency + throughput)
- **PRD.md** — product requirements with current state, prioritized next steps, acceptance criteria

### Changed
- **lcvdb.h** — rewritten for split storage: `lcvdb_topo_t` (32B), `lcvdb_vec_t` (64B), new `lcvdb_t` struct with separate array pointers
- **search_ref.c** — rewritten for uint16 IDs, split array access, skip deleted nodes, return count
- **Makefile** — updated for C reference files (init_ref.c, build_ref.c, search_ref.c), builds all three test binaries

### Verified on Device
- All 7 VDB correctness tests pass (distance, insert, vector integrity, payload IDs, connectivity, search, delete)
- 100% recall@5 at N<=128, 99.6% at N=256
- 19.4 us search latency at N=256, 24.3 us at N=1024
- 51K QPS at N=256, 41K QPS at N=1024

---

## dea7fb2 — GPU probe + OpenCL benchmark (2026-02-08)

### Added
- `gpu_probe.c` — OpenCL device query for PowerVR BXM-8-256
- `gpu_bench.c` — int8 dot product benchmark on GPU
- `gpu_dot.cl` — OpenCL kernel for int8 dot product
- Confirmed: 128-wide SIMD, 28 KB local mem, OpenCL 3.0, 950 MHz

### Measured
- GPU dispatch overhead: ~40 us (too slow for standalone VDB, useful for async)

## befeaa5 — int16 overflow fix + NEON search (2026-02-07)

### Fixed
- **int16 overflow in NEON dot products** — SMULL+SMLAL accumulating 3 chunks into int16 can exceed +/-32767. Fixed all 7 dot product sites in `distance.S` and `build.S` by widening to int32 via SADDLP/SADALP after each chunk.

### Changed
- **Search uses NEON dot product** — replaced scalar C `dot_i8_ref` with `lcvdb_dot_i8` from `distance.S`. 26% speedup (24 us -> 19 us at N=256).
- **Search candidate addition** — add ALL unvisited neighbors to candidate pool (was: only those beating worst result). MAX_CAND increased from 64 to 256.

### Measured
- N=256: 18,870 ns/query, 53K QPS, 99.6% recall@5

## 3184ced — HNSW diversity heuristic + search fixes (2026-02-07)

### Added
- **HNSW diversity heuristic** in `build.S` — Algorithm 4 extended heuristic: collect 2*M candidates, sort by score, keep candidate C only if `dot(query, C) > dot(S, C)` for all selected S
- **Upper layer support** — greedy descent through layers > 0 before layer-0 beam search

### Fixed
- Search under-exploration at larger N

## 75829b8 — Q8_0 RS kernel + VDB foundation (2026-02-06)

### Added
- **Q8_0 Row-Scaled kernel** — `block_q8_0x4_rs` (128 bytes), NEON DOTPROD GEMV
- **L-Cache VDB initial implementation** — `lcvdb.h`, `init.S`, `distance.S`, `build.S`, `search_ref.c`
- HNSW graph with 48D int8 vectors, M=8

### Measured
- LFM2-350M Q8_0: 40.9 tok/s (corrected after cache fix)

## 0db4425 — Graph dispatch + fast-forward (2026-02-06)

### Added
- **Graph dispatch fast path** — bypass scheduler overhead for simple graphs
- **Thread 1 fast-forward** — main thread starts compute before worker is signaled
- **TG fast path** — inlined GEMV for token generation
- ~~**Activation quant caching**~~ — REMOVED in later commit (caused stale data due to pointer reuse)

### Measured
- LFM2-350M Q4_0: 56.8 -> 58.3 tok/s (+2.6%)

## 89b555e — Barrier-skip optimization (2026-02-05)

### Added
- Skip pthread barriers when only 1 thread has work

### Measured
- LFM2-350M Q4_0: 56.0 -> 56.8 tok/s (+1.4%)

## bd3dad3 — GEMM dispatch (2026-02-05)

### Added
- **GEMM kernel** — 4 activation rows processed simultaneously with weight reuse
- Automatic dispatch: GEMV for tg (1 token), GEMM for pp (batch)

### Measured
- LFM2-350M Q4_0 pp32: 304 tok/s (3x over sequential GEMV)

## 0614203 — Row-Scaled Q4_0 (2026-02-04)

### Added
- **Row-Scaled Q4_0 format** (`block_q4_0x4_rs`) — 64-byte cache-line-aligned blocks
- **Pure-integer SDOT GEMV** — `vdotq_laneq_s32` inner loop, no float until final reduction
- **Power-of-2 shift activation quantization** — IEEE754 bit manipulation
- Custom `repack.h`, `repack.cpp`, `arch/arm/repack.cpp`

### Measured
- LFM2-350M Q4_0: 55.8 tok/s (baseline for optimization series)

## 69c0a5b — DRAM bandwidth discovery (2026-02-04)

### Discovered
- **Android framework consumes ~4 GB/s DRAM bandwidth** — SurfaceFlinger, SystemUI, etc.
- Running `su -c 'stop'` frees bandwidth, eliminates "thermal throttling"
- Magisk boot script at `/data/adb/service.d/moltar_perf.sh`

### Measured
- LFM2-350M Q4_0: 58 -> 77 tok/s with Android stopped (+32%, corrected after cache fix)
- DRAM bandwidth: 11 GB/s -> 15.5 GB/s (2 threads, Android stopped)
