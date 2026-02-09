#!/system/bin/sh
#
# moltar_rag.sh — On-device RAG pipeline using ColBERT + LFM2
#
# Usage:
#   moltar_rag.sh ingest <text_dir>     — Index all .txt files in directory
#   moltar_rag.sh query "your question" — Answer using RAG
#   moltar_rag.sh demo                  — Run demo with built-in questions
#
# Paths (all on device at /data/local/tmp/)

BASE=/data/local/tmp
COLBERT_MODEL=$BASE/LFM2-ColBERT-350M-Q4_0.gguf
LLM_MODEL=$BASE/LFM2-1.2B-Q4_0.gguf
EMBEDDING_BIN=$BASE/llama-embedding
LLM_BIN=$BASE/llama-cli
RAG_BIN=$BASE/moltar_rag
INDEX_DIR=$BASE/rag_index
export LD_LIBRARY_PATH=$BASE

# ColBERT settings
COLBERT_CTX=1024
COLBERT_THREADS=2

# LLM settings
LLM_THREADS=2
LLM_TOKENS=256

# ── Helpers ──

log() { echo "[moltar] $*"; }
die() { echo "[moltar] ERROR: $*" >&2; exit 1; }

get_embeddings() {
    # Run llama-embedding and capture the raw embedding output
    # Input: text on stdin or as argument
    # Output: embedding floats to stdout (parsed)
    local text="$1"
    local tmpfile=$BASE/rag_emb_tmp.txt

    # Run embedding with pooling=none for per-token embeddings
    # Use raw format: one line per token, 128 space-separated floats
    taskset c0 $EMBEDDING_BIN \
        -m $COLBERT_MODEL \
        -c $COLBERT_CTX \
        -p "$text" \
        --pooling none \
        --embd-normalize -1 \
        --embd-output-format raw \
        -t $COLBERT_THREADS \
        --no-warmup \
        --no-repack \
        2>/dev/null
}

# ── Ingest Command ──

cmd_ingest() {
    local text_dir="$1"
    [ -d "$text_dir" ] || die "Directory not found: $text_dir"

    mkdir -p $INDEX_DIR

    log "Ingesting documents from $text_dir"

    # Process each text file
    local doc_id=0
    local chunk_id=0

    for f in "$text_dir"/*.txt; do
        [ -f "$f" ] || continue
        local fname=$(basename "$f")
        log "  Processing: $fname"

        # Read file and split into chunks (by paragraph / double newline)
        # Each chunk: max ~400 chars to stay under 512 tokens
        local chunk=""
        local chunk_len=0

        while IFS= read -r line || [ -n "$line" ]; do
            if [ -z "$line" ] && [ $chunk_len -gt 0 ]; then
                # Empty line = paragraph break, flush chunk if big enough
                if [ $chunk_len -gt 100 ]; then
                    # Save chunk text
                    echo "$chunk" > "$INDEX_DIR/chunk_${chunk_id}.txt"

                    # Get embeddings
                    log "    Chunk $chunk_id: ${chunk_len} chars"
                    get_embeddings "$chunk" > "$INDEX_DIR/chunk_${chunk_id}.emb"

                    # Count tokens
                    local n_tok=$(wc -l < "$INDEX_DIR/chunk_${chunk_id}.emb")
                    log "    -> $n_tok token embeddings"

                    chunk_id=$((chunk_id + 1))
                    chunk=""
                    chunk_len=0
                fi
            else
                if [ $chunk_len -gt 0 ]; then
                    chunk="$chunk $line"
                else
                    chunk="$line"
                fi
                chunk_len=$((chunk_len + ${#line}))
            fi
        done < "$f"

        # Flush last chunk
        if [ $chunk_len -gt 50 ]; then
            echo "$chunk" > "$INDEX_DIR/chunk_${chunk_id}.txt"
            log "    Chunk $chunk_id: ${chunk_len} chars"
            get_embeddings "$chunk" > "$INDEX_DIR/chunk_${chunk_id}.emb"
            local n_tok=$(wc -l < "$INDEX_DIR/chunk_${chunk_id}.emb")
            log "    -> $n_tok token embeddings"
            chunk_id=$((chunk_id + 1))
        fi

        doc_id=$((doc_id + 1))
    done

    # Write manifest
    echo "$chunk_id" > "$INDEX_DIR/manifest.txt"
    log "Ingestion complete: $chunk_id chunks from $doc_id documents"
}

# ── Query Command ──

cmd_query() {
    local query="$1"
    [ -n "$query" ] || die "No query provided"
    [ -f "$INDEX_DIR/manifest.txt" ] || die "No index found. Run: moltar_rag.sh ingest <dir>"

    local n_chunks=$(cat "$INDEX_DIR/manifest.txt")
    log "Query: $query"
    log "Searching $n_chunks chunks..."

    # Step 1: Embed query
    local t0=$(date +%s%N 2>/dev/null || date +%s)
    get_embeddings "$query" > "$BASE/rag_query.emb"
    local n_qtok=$(wc -l < "$BASE/rag_query.emb")
    local t1=$(date +%s%N 2>/dev/null || date +%s)
    log "Query embedded: $n_qtok tokens"

    # Step 2: MaxSim search — use the C binary
    taskset c0 $RAG_BIN search \
        "$BASE/rag_query.emb" \
        "$INDEX_DIR" \
        "$n_chunks" \
        3 > "$BASE/rag_results.txt"

    log "Search results:"
    cat "$BASE/rag_results.txt"

    # Step 3: Collect context from top results
    local context=""
    while IFS='|' read -r rank score cid; do
        # Trim whitespace
        cid=$(echo "$cid" | tr -d ' ')
        local chunk_text=$(cat "$INDEX_DIR/chunk_${cid}.txt")
        context="${context}${chunk_text}\n\n"
        log "  Rank $rank (score=$score): chunk $cid"
    done < "$BASE/rag_results.txt"

    # Step 4: Build prompt and generate
    # Use -sys for system prompt and -p for user query
    # llama-cli applies chat template automatically (ChatML for LFM2)
    local sys_prompt="Answer the question using only the provided context. Be concise and accurate. Context: ${context}"

    log "Generating response..."
    local t2=$(date +%s%N 2>/dev/null || date +%s)

    local llm_out=$BASE/rag_llm_out.txt
    taskset c0 $LLM_BIN \
        -m $LLM_MODEL \
        -c 2048 \
        -sys "$sys_prompt" \
        -p "$query" \
        -n $LLM_TOKENS \
        -t $LLM_THREADS \
        --no-warmup \
        --no-repack \
        --no-display-prompt \
        --single-turn \
        --simple-io \
        2>/dev/null > "$llm_out"

    # Extract generated text: skip banner/UI lines, stop before stats
    local in_response=0
    while IFS= read -r line; do
        case "$line" in
            "> "*) in_response=1; continue ;;
            "[ Prompt:"*) break ;;
            "Loading model"*|"build "*|"model "*|"modalities"*) continue ;;
            "available commands:"*|"  /"*) continue ;;
            *"▄▄"*|*"██"*|*"▀▀"*|*"████"*) continue ;;
            "Exiting..."*) break ;;
            "")
                if [ $in_response -eq 1 ]; then echo ""; fi
                continue ;;
            *)
                if [ $in_response -eq 1 ]; then echo "$line"; fi
                ;;
        esac
    done < "$llm_out"

    echo ""
    local t3=$(date +%s%N 2>/dev/null || date +%s)
    log "Done."
}

# ── Demo Command ──

cmd_demo() {
    log "=== Moltar RAG Demo ==="
    log ""

    # Check prerequisites
    [ -f "$COLBERT_MODEL" ] || die "ColBERT model not found: $COLBERT_MODEL"
    [ -f "$LLM_MODEL" ] || die "LLM model not found: $LLM_MODEL"
    [ -f "$EMBEDDING_BIN" ] || die "llama-embedding not found: $EMBEDDING_BIN"
    [ -f "$LLM_BIN" ] || die "llama-cli not found: $LLM_BIN"

    # Check for index
    if [ ! -f "$INDEX_DIR/manifest.txt" ]; then
        log "No index found. Ingesting knowledge base..."
        cmd_ingest "$BASE/rag_knowledge"
    fi

    log ""
    log "=== Query 1 ==="
    cmd_query "How fast is LLM inference on the Moto G Power?"

    log ""
    log "=== Query 2 ==="
    cmd_query "What is the ColBERT embedding model?"

    log ""
    log "=== Query 3 ==="
    cmd_query "Why did stopping Android framework improve performance?"
}

# ── Main ──

case "$1" in
    ingest) cmd_ingest "$2" ;;
    query)  cmd_query "$2" ;;
    demo)   cmd_demo ;;
    *)
        echo "Usage: $0 {ingest <dir>|query \"question\"|demo}"
        exit 1
        ;;
esac
