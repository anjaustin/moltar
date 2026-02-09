# Changelog

## [Unreleased]

### Added
- **L-Cache VDB split storage redesign** — separate topology (32B/node) and vector (64B/node) arrays, uint16 IDs (max 65534 nodes), tombstone delete, payload IDs
- **C reference implementations** — `init_ref.c`, `build_ref.c`, `search_ref.c` replace old assembly for split storage layout
- **VDB test suite** — `test_lcvdb.c` (7 correctness tests), `test_recall.c` (N=32..1024), `test_bench.c` (latency + throughput)
- **PRD.md** — product requirements with current state, prioritized next steps, acceptance criteria

### Changed
- **lcvdb.h** — rewritten for split storage: `lcvdb_topo_t` (32B), `lcvdb_vec_t` (64B), new `lcvdb_t` struct with separate array pointers
- **search_ref.c** — rewritten for uint16 IDs, split array access, skip deleted nodes, return count
- **Makefile** — updated for C reference files (init_ref.c, build_ref.c, search_ref.c), builds all three test binaries
- **README.md** — rewritten with actual performance numbers and repo structure
- **PERFORMANCE.md** — rewritten with measured benchmarks (LLM + VDB)
- **ARCHITECTURE.md** — rewritten with actual system design
- **CHANGELOG.md** — rewritten with real commit history

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
- LFM2-350M Q8_0: 41.69 tok/s

## 0db4425 — Graph dispatch + fast-forward + quant caching (2026-02-06)

### Added
- **Graph dispatch fast path** — bypass scheduler overhead for simple graphs
- **Thread 1 fast-forward** — main thread starts compute before worker is signaled
- **Activation quant caching** — quantize once, reuse across GEMV calls
- **TG fast path** — inlined GEMV for token generation

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
- LFM2-350M Q4_0: 58 -> 80 tok/s with Android stopped (+37%)
- DRAM bandwidth: 11 GB/s -> 15.5 GB/s (2 threads, Android stopped)
