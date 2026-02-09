# ColBERT RAG Pipeline

On-device retrieval-augmented generation for the Motorola Moto G Power 5G (2023). Uses ColBERT late-interaction embeddings for retrieval and LFM2 for generation.

## How It Works

1. **Ingest**: Chunk documents, embed each chunk with LFM2-ColBERT-350M (128D float per token), quantize to int8, store index
2. **Query**: Embed query tokens, run MaxSim scoring against all document chunks, retrieve top-k
3. **Generate**: Inject retrieved context into prompt, generate answer with LFM2-1.2B

## Performance (on device)

| Step | Time |
|------|------|
| Embed query (8-15 tokens) | ~66 ms |
| MaxSim search (10 chunks) | ~15 ms |
| LLM generate (short answer) | ~1-2 s |
| **Total** | **~2 s** |

## Models

| Model | File | Size | Purpose |
|-------|------|------|---------|
| LFM2-ColBERT-350M | `LFM2-ColBERT-350M-Q4_0.gguf` | 209 MB | Embedding (128D per token) |
| LFM2-1.2B | `LFM2-1.2B-Q4_0.gguf` | 661 MB | Generation (~21 tok/s) |

Models run sequentially (not enough RAM for both + KV cache simultaneously).

## Files

| File | Description |
|------|-------------|
| `colbert.h` | Index structs: 128D int8 token embeddings, document entries, MaxSim API |
| `colbert.c` | C implementation: init, quantize f32->i8, add_doc, search with top-k heap |
| `maxsim_neon.S` | NEON assembly: `colbert_dot_i8` (128D SDOT), `colbert_maxsim_i8` (full MaxSim) |
| `test_colbert.c` | 4 correctness tests + benchmark |
| `moltar_rag.c` | Search binary: parses raw embedding files, runs MaxSim, outputs ranked results |
| `moltar_rag.sh` | Shell orchestrator: `ingest`, `query`, `demo` commands |
| `knowledge/moltar.txt` | Sample knowledge base about the Moltar project |
| `Makefile` | Builds `test_colbert` and `moltar_rag` with `aarch64-linux-gnu-gcc -static` |

## Build

```bash
# From x86_64 host
make clean all    # builds test_colbert, moltar_rag

# Push to device
adb push test_colbert moltar_rag moltar_rag.sh /data/local/tmp/
```

## Usage

### Run Tests

```bash
adb shell "su -c 'taskset c0 /data/local/tmp/test_colbert'"
```

All 4 tests should pass: dot product, MaxSim, quantization, search ranking.

### Full RAG Demo

```bash
# Requires models + llama-cli + libs already on device at /data/local/tmp/
adb shell "su -c 'sh /data/local/tmp/moltar_rag.sh demo'"
```

### Ingest a Knowledge Base

```bash
# Push knowledge base to device
adb push knowledge/moltar.txt /data/local/tmp/rag_knowledge/moltar.txt

# Ingest (chunks text, embeds, builds index)
adb shell "su -c 'sh /data/local/tmp/moltar_rag.sh ingest /data/local/tmp/rag_knowledge/moltar.txt'"
```

### Query

```bash
adb shell "su -c 'sh /data/local/tmp/moltar_rag.sh query \"How fast is LLM inference?\"'"
```

## ColBERT Scoring

ColBERT uses **late interaction** — per-token embeddings instead of a single vector per document.

**MaxSim**:
```
score(Q, D) = sum over query tokens q of: max over document tokens d of: dot(q, d)
```

Each query token finds its best-matching document token, then scores are summed. This captures fine-grained token-level semantic matching that single-vector models miss.

**Quantization**: float32 embeddings from `llama-embedding --embd-output-format raw` are quantized to int8 via per-vector max-abs scaling. This preserves relative ordering for ranking while enabling integer NEON dot products.

## Dependencies

- `llama-embedding` — for embedding queries and documents (from `research/llama.cpp/`)
- `llama-cli` — for text generation (from `research/llama.cpp/`)
- Shared libraries: `libggml-base.so`, `libggml-cpu.so`, `libggml.so`, `libllama.so`, `libmtmd.so`
- All must be at `/data/local/tmp/` on device

## Known Limitations

- **Sequential model loading**: Can't run embedding + generation simultaneously
- **Brute-force search**: MaxSim scans all chunks — O(N) in document count. Fine for <100 chunks, may need optimization for larger indices
- **`llama-cli` output parsing**: Conversation mode is default; uses `--single-turn` + `--no-display-prompt` with banner filtering
- **Android shell**: Device uses mksh/toybox, no bash-isms. Script avoids arrays, `[[ ]]`, process substitution
