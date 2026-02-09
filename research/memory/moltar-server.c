/* ==========================================================================
 * Moltar Server — HTTP API for Three-Layer Memory Agent
 * ==========================================================================
 * Single-threaded HTTP server embedding the full moltar-agent:
 *   - LFM2-1.2B inference via libllama.so
 *   - LCVDB working memory (semantic turn recall)
 *   - ColBERT RAG knowledge retrieval (via subprocess)
 *   - Session persistence (save/load LCVDB + turns)
 *
 * API:
 *   GET  /              → embedded web chat UI
 *   POST /api/chat      → {"message":"..."} → {"response":"...","turn":N}
 *   POST /api/ingest    → {"file":"..."} or {"text":"..."} → {"chunks":N}
 *   GET  /api/status    → {"turns":N,"rag_chunks":N,"rag":bool}
 *   POST /api/rag       → {"enabled":bool}
 *   POST /api/save      → save session
 *   POST /api/load      → load session
 *
 * Build: make server (NDK dynamic, links libllama.so)
 * Run:   moltar-server <model.gguf> [options] [-p PORT]
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>

#include "llama.h"
#include "ggml-backend.h"

#include "project.h"
#include "../lcvdb/lcvdb.h"

/* ---------- Agent Configuration (same as moltar-agent.c) ---------- */

#define MAX_TURNS       256
#define MAX_CONTEXT     3
#undef  MAX_INPUT
#define MAX_INPUT       4096     /* larger for HTTP payloads */
#define MAX_RESPONSE    512
#define MAX_PROMPT_TOKENS 2048
#define MAX_TURN_TEXT   2048
#define MAX_KNOWLEDGE   4096
#define MAX_PROMPT_BUF  16384
#define PROJ_SEED       0x4D4F4C54

#define RAG_BASE        "/data/local/tmp"
#define RAG_COLBERT     RAG_BASE "/LFM2-ColBERT-350M-Q4_0.gguf"
#define RAG_EMBEDDING   RAG_BASE "/llama-embedding"
#define RAG_SEARCH      RAG_BASE "/moltar_rag"
#define RAG_INDEX       RAG_BASE "/rag_index"
#define RAG_TOP_K       3

/* ---------- HTTP Configuration ---------- */

#define DEFAULT_PORT    8080
#define HTTP_BUF_SIZE   65536
#define MAX_HEADERS     64

/* ---------- Turn storage ---------- */

typedef struct {
    char text[MAX_TURN_TEXT];
    int  valid;
} turn_t;

static turn_t     turns[MAX_TURNS];
static int        n_turns = 0;

/* ---------- LCVDB working memory ---------- */

static lcvdb_t    vdb __attribute__((aligned(64)));
static void      *topo_buf;
static void      *vec_buf;

static moltar_proj_t proj;

/* ---------- Session Persistence ---------- */

#define SESSION_MAGIC   0x4D4F4C54
#define SESSION_VERSION 1

static int session_save(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t magic = SESSION_MAGIC, version = SESSION_VERSION;
    uint32_t nt = (uint32_t)n_turns;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&nt, 4, 1, f);
    fwrite(&vdb, 32, 1, f);
    fwrite(topo_buf, sizeof(lcvdb_topo_t), nt, f);
    fwrite(vec_buf, sizeof(lcvdb_vec_t), nt, f);
    fwrite(turns, sizeof(turn_t), nt, f);
    fclose(f);
    return 0;
}

static int session_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic, version, nt;
    if (fread(&magic, 4, 1, f) != 1 || magic != SESSION_MAGIC) { fclose(f); return -1; }
    if (fread(&version, 4, 1, f) != 1 || version != SESSION_VERSION) { fclose(f); return -1; }
    if (fread(&nt, 4, 1, f) != 1 || nt > MAX_TURNS) { fclose(f); return -1; }
    fread(&vdb, 32, 1, f);
    vdb.topo_array = (lcvdb_topo_t *)topo_buf;
    vdb.vec_array  = (lcvdb_vec_t *)vec_buf;
    vdb.max_nodes  = MAX_TURNS;
    fread(topo_buf, sizeof(lcvdb_topo_t), nt, f);
    fread(vec_buf, sizeof(lcvdb_vec_t), nt, f);
    fread(turns, sizeof(turn_t), nt, f);
    n_turns = (int)nt;
    fclose(f);
    return 0;
}

/* ---------- RAG Configuration ---------- */

typedef struct {
    int   enabled;
    char  colbert_model[256];
    char  embedding_bin[256];
    char  search_bin[256];
    char  index_dir[256];
    int   top_k;
    int   n_chunks;
} rag_config_t;

static rag_config_t rag_cfg;

static int rag_load_manifest(rag_config_t *cfg) {
    char path[512];
    snprintf(path, sizeof(path), "%s/manifest.txt", cfg->index_dir);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int n = 0;
    if (fscanf(f, "%d", &n) != 1) n = 0;
    fclose(f);
    return n;
}

static int rag_retrieve(const rag_config_t *cfg, const char *query,
                        char *out_buf, int out_max,
                        const char *llm_model_path) {
    if (!cfg->enabled || cfg->n_chunks <= 0) return 0;
    char cmd[2048], emb_path[256], results_path[256], query_path[256];
    snprintf(emb_path, sizeof(emb_path), "%s/rag_query_server.emb", RAG_BASE);
    snprintf(results_path, sizeof(results_path), "%s/rag_results_server.txt", RAG_BASE);
    snprintf(query_path, sizeof(query_path), "%s/rag_query_server.txt", RAG_BASE);

    FILE *qf = fopen(query_path, "w");
    if (!qf) return 0;
    fputs(query, qf);
    fclose(qf);

    snprintf(cmd, sizeof(cmd),
        "export LD_LIBRARY_PATH=%s && taskset c0 %s -m %s -c 1024 -f %s "
        "--pooling none --embd-normalize -1 --embd-output-format raw "
        "-t 2 --no-warmup 2>/dev/null > %s",
        RAG_BASE, cfg->embedding_bin, cfg->colbert_model, query_path, emb_path);
    if (system(cmd) != 0) return 0;

    if (llm_model_path) {
        snprintf(cmd, sizeof(cmd), "cat %s > /dev/null 2>/dev/null &", llm_model_path);
        system(cmd);
    }

    snprintf(cmd, sizeof(cmd),
        "taskset c0 %s search %s %s %d %d > %s 2>/dev/null",
        cfg->search_bin, emb_path, cfg->index_dir,
        cfg->n_chunks, cfg->top_k, results_path);
    if (system(cmd) != 0) return 0;

    FILE *f = fopen(results_path, "r");
    if (!f) return 0;
    int pos = 0;
    char line[256];
    int n_results = 0;
    while (fgets(line, sizeof(line), f) && n_results < cfg->top_k) {
        int rank, score, chunk_id;
        if (sscanf(line, "%d|%d|%d", &rank, &score, &chunk_id) != 3) continue;
        char chunk_path[512];
        snprintf(chunk_path, sizeof(chunk_path), "%s/chunk_%d.txt", cfg->index_dir, chunk_id);
        FILE *cf = fopen(chunk_path, "r");
        if (!cf) continue;
        char chunk_text[2048];
        int chunk_len = (int)fread(chunk_text, 1, sizeof(chunk_text) - 1, cf);
        fclose(cf);
        chunk_text[chunk_len] = '\0';
        while (chunk_len > 0 && (chunk_text[chunk_len-1] == '\n' || chunk_text[chunk_len-1] == ' '))
            chunk_text[--chunk_len] = '\0';
        if (pos + chunk_len + 10 < out_max) {
            pos += snprintf(out_buf + pos, out_max - pos, "- %s\n", chunk_text);
            n_results++;
        }
    }
    fclose(f);
    return pos;
}

/* ---------- Prompt Builder ---------- */

static int build_prompt(char *prompt, int prompt_max,
                        const char *user_input, int use_memory,
                        const char *knowledge_ctx) {
    int pos = 0;
    pos += snprintf(prompt + pos, prompt_max - pos,
        "<|im_start|>system\n"
        "You are Moltar, a helpful AI assistant running on a Motorola phone. "
        "Be concise and direct.");
    if (knowledge_ctx && knowledge_ctx[0])
        pos += snprintf(prompt + pos, prompt_max - pos,
            "\n\nKnowledge base:\n%s", knowledge_ctx);
    if (use_memory && n_turns > 0 && n_turns >= 1) {
        uint16_t last_id = (uint16_t)(n_turns - 1);
        uint16_t result_ids[MAX_CONTEXT + 1];
        int32_t  result_scores[MAX_CONTEXT + 1];
        int nresults = lcvdb_search(&vdb,
            (const int8_t *)((char *)vec_buf + last_id * sizeof(lcvdb_vec_t)),
            MAX_CONTEXT + 1, result_ids, result_scores);
        int context_count = 0;
        char context_buf[MAX_TURN_TEXT * MAX_CONTEXT];
        int ctx_pos = 0;
        for (int i = 0; i < nresults && context_count < MAX_CONTEXT; i++) {
            uint16_t tid = result_ids[i];
            if (tid == last_id) continue;
            if (tid < MAX_TURNS && turns[tid].valid) {
                ctx_pos += snprintf(context_buf + ctx_pos,
                    sizeof(context_buf) - ctx_pos, "- %s\n", turns[tid].text);
                context_count++;
            }
        }
        if (context_count > 0)
            pos += snprintf(prompt + pos, prompt_max - pos,
                "\n\nRelevant prior conversation:\n%s", context_buf);
    }
    pos += snprintf(prompt + pos, prompt_max - pos, "<|im_end|>\n");
    if (use_memory && n_turns >= 1 && turns[n_turns - 1].valid) {
        const char *turn_text = turns[n_turns - 1].text;
        const char *asst_part = strstr(turn_text, "\nAssistant: ");
        if (asst_part) {
            const char *user_part = turn_text;
            if (strncmp(user_part, "User: ", 6) == 0) user_part += 6;
            int user_len = (int)(asst_part - user_part);
            pos += snprintf(prompt + pos, prompt_max - pos,
                "<|im_start|>user\n%.*s<|im_end|>\n"
                "<|im_start|>assistant\n%s<|im_end|>\n",
                user_len, user_part, asst_part + 12);
        }
    }
    pos += snprintf(prompt + pos, prompt_max - pos,
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_input);
    return pos;
}

/* ---------- Global LLM state ---------- */

static struct llama_model   *g_model;
static struct llama_context *g_ctx;
static struct llama_sampler *g_smpl;
static const struct llama_vocab *g_vocab;
static llama_token          *g_prompt_tokens;
static char                 *g_prompt_buf;
static int                   g_n_embd;
static int                   g_use_memory = 1;
static const char           *g_model_path;
static const char           *g_session_path;
static int                   g_n_predict = MAX_RESPONSE;

/* ---------- Chat turn (core logic) ---------- */

static int do_chat(const char *user_input, char *response_out, int response_max) {
    char knowledge_buf[MAX_KNOWLEDGE];
    knowledge_buf[0] = '\0';

    if (rag_cfg.enabled)
        rag_retrieve(&rag_cfg, user_input, knowledge_buf, MAX_KNOWLEDGE, g_model_path);

    int prompt_len = build_prompt(g_prompt_buf, MAX_PROMPT_BUF,
                                   user_input, g_use_memory, knowledge_buf);

    int n_tokens = -llama_tokenize(g_vocab, g_prompt_buf, prompt_len, NULL, 0, 1, 1);
    if (n_tokens > MAX_PROMPT_TOKENS) n_tokens = MAX_PROMPT_TOKENS;
    llama_tokenize(g_vocab, g_prompt_buf, prompt_len, g_prompt_tokens, n_tokens, 1, 1);

    llama_memory_clear(llama_get_memory(g_ctx), 1);

    struct llama_batch batch = llama_batch_get_one(g_prompt_tokens, n_tokens);
    if (llama_decode(g_ctx, batch) != 0) return -1;

    int response_len = 0, n_generated = 0;
    for (int i = 0; i < g_n_predict; i++) {
        llama_token new_token = llama_sampler_sample(g_smpl, g_ctx, -1);
        if (llama_vocab_is_eog(g_vocab, new_token)) break;

        char piece[128];
        int piece_len = llama_token_to_piece(g_vocab, new_token, piece, sizeof(piece) - 1, 0, 1);
        if (piece_len > 0 && response_len + piece_len < response_max - 1) {
            memcpy(response_out + response_len, piece, piece_len);
            response_len += piece_len;
        }
        n_generated++;

        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_ctx, batch) != 0) break;
    }
    response_out[response_len] = '\0';

    /* Store in working memory */
    if (g_use_memory && n_turns < MAX_TURNS) {
        float *embd = llama_get_embeddings_ith(g_ctx, -1);
        if (embd) {
            int8_t vec48[48];
            moltar_proj_apply(&proj, embd, vec48);
            lcvdb_insert(&vdb, vec48, (uint32_t)n_turns);
            snprintf(turns[n_turns].text, MAX_TURN_TEXT,
                     "User: %s\nAssistant: %s", user_input, response_out);
            turns[n_turns].valid = 1;
            n_turns++;
        }
    }

    return n_turns;
}

/* ---------- JSON helpers (minimal, no deps) ---------- */

/* Extract a string value for a key from a JSON object. Returns ptr into buf. */
static const char *json_get_string(const char *json, const char *key,
                                    char *out, int out_max) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return NULL;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < out_max - 1) {
        if (*p == '\\' && p[1]) { p++; } /* skip escapes simply */
        out[i++] = *p++;
    }
    out[i] = '\0';
    return out;
}

static int json_get_bool(const char *json, const char *key, int default_val) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return default_val;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return default_val;
}

/* Escape a string for JSON output. Returns length written. */
static int json_escape(const char *src, char *dst, int dst_max) {
    int j = 0;
    for (int i = 0; src[i] && j < dst_max - 2; i++) {
        char c = src[i];
        if (c == '"' || c == '\\') { dst[j++] = '\\'; dst[j++] = c; }
        else if (c == '\n') { dst[j++] = '\\'; dst[j++] = 'n'; }
        else if (c == '\r') { dst[j++] = '\\'; dst[j++] = 'r'; }
        else if (c == '\t') { dst[j++] = '\\'; dst[j++] = 't'; }
        else if ((unsigned char)c < 0x20) { /* skip control chars */ }
        else { dst[j++] = c; }
    }
    dst[j] = '\0';
    return j;
}

/* ---------- Embedded Web UI ---------- */

static const char *WEB_UI =
"<!DOCTYPE html>\n"
"<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
"<title>Moltar</title>\n"
"<style>\n"
"*{box-sizing:border-box;margin:0;padding:0}\n"
"body{font-family:-apple-system,system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}\n"
"#header{padding:12px 16px;background:#16213e;border-bottom:1px solid #0f3460;display:flex;align-items:center;gap:12px}\n"
"#header h1{font-size:18px;color:#e94560;font-weight:700}\n"
"#header .status{font-size:12px;color:#888}\n"
"#chat{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}\n"
".msg{max-width:85%;padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}\n"
".user{align-self:flex-end;background:#0f3460;color:#e0e0e0;border-bottom-right-radius:4px}\n"
".bot{align-self:flex-start;background:#16213e;color:#e0e0e0;border-bottom-left-radius:4px;border:1px solid #0f3460}\n"
".thinking{align-self:flex-start;color:#888;font-style:italic;font-size:13px}\n"
"#input-bar{padding:12px 16px;background:#16213e;border-top:1px solid #0f3460;display:flex;gap:8px}\n"
"#msg-input{flex:1;padding:10px 14px;border:1px solid #0f3460;border-radius:8px;background:#1a1a2e;color:#e0e0e0;font-size:14px;outline:none}\n"
"#msg-input:focus{border-color:#e94560}\n"
"#send-btn{padding:10px 20px;background:#e94560;color:#fff;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer}\n"
"#send-btn:hover{background:#c73e54}\n"
"#send-btn:disabled{background:#555;cursor:not-allowed}\n"
"</style></head><body>\n"
"<div id=\"header\"><h1>Moltar</h1><span class=\"status\" id=\"status\">Loading...</span></div>\n"
"<div id=\"chat\"></div>\n"
"<div id=\"input-bar\">\n"
"<input id=\"msg-input\" placeholder=\"Type a message...\" autocomplete=\"off\">\n"
"<button id=\"send-btn\">Send</button>\n"
"</div>\n"
"<script>\n"
"const chat=document.getElementById('chat'),input=document.getElementById('msg-input'),btn=document.getElementById('send-btn'),status=document.getElementById('status');\n"
"function addMsg(text,cls){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d;}\n"
"async function refreshStatus(){try{const r=await fetch('/api/status');const s=await r.json();status.textContent='Turns: '+s.turns+' | RAG: '+(s.rag?'ON ('+s.rag_chunks+')':'OFF');}catch(e){status.textContent='Disconnected';}}\n"
"async function send(){\n"
"  const text=input.value.trim();if(!text)return;\n"
"  input.value='';btn.disabled=true;\n"
"  addMsg(text,'user');\n"
"  const thinking=addMsg('Thinking...','thinking');\n"
"  try{\n"
"    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});\n"
"    const j=await r.json();\n"
"    thinking.remove();\n"
"    addMsg(j.response,'bot');\n"
"    refreshStatus();\n"
"  }catch(e){thinking.textContent='Error: '+e.message;}\n"
"  btn.disabled=false;input.focus();\n"
"}\n"
"btn.onclick=send;\n"
"input.onkeydown=e=>{if(e.key==='Enter'&&!btn.disabled)send();};\n"
"refreshStatus();input.focus();\n"
"</script></body></html>\n";

/* ---------- HTTP Server ---------- */

static void http_respond(int fd, int code, const char *content_type,
                         const char *body, int body_len) {
    const char *status_text = (code == 200) ? "OK" : (code == 400) ? "Bad Request" : "Not Found";
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        code, status_text, content_type, body_len);
    write(fd, header, hlen);
    if (body_len > 0) write(fd, body, body_len);
}

static void http_json(int fd, int code, const char *json) {
    http_respond(fd, code, "application/json", json, (int)strlen(json));
}

static volatile int g_running = 1;

static void handle_signal(int sig) {
    (void)sig;
    g_running = 0;
}

static void handle_request(int client_fd, const char *method, const char *path,
                           const char *body) {
    char json_buf[MAX_TURN_TEXT * 2];
    char escaped[MAX_TURN_TEXT * 2];

    /* OPTIONS (CORS preflight) */
    if (strcmp(method, "OPTIONS") == 0) {
        http_respond(client_fd, 200, "text/plain", "", 0);
        return;
    }

    /* GET / — Web UI */
    if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
        http_respond(client_fd, 200, "text/html; charset=utf-8",
                     WEB_UI, (int)strlen(WEB_UI));
        return;
    }

    /* GET /api/status */
    if (strcmp(method, "GET") == 0 && strcmp(path, "/api/status") == 0) {
        snprintf(json_buf, sizeof(json_buf),
            "{\"turns\":%d,\"rag_chunks\":%d,\"rag\":%s,\"memory\":%s}",
            n_turns, rag_cfg.n_chunks,
            rag_cfg.enabled ? "true" : "false",
            g_use_memory ? "true" : "false");
        http_json(client_fd, 200, json_buf);
        return;
    }

    /* POST /api/chat */
    if (strcmp(method, "POST") == 0 && strcmp(path, "/api/chat") == 0) {
        char message[MAX_INPUT];
        if (!json_get_string(body, "message", message, sizeof(message))) {
            http_json(client_fd, 400, "{\"error\":\"missing 'message' field\"}");
            return;
        }

        fprintf(stderr, "[server] Chat: %.60s%s\n", message,
                (int)strlen(message) > 60 ? "..." : "");

        char response[MAX_TURN_TEXT];
        int turn = do_chat(message, response, sizeof(response));

        if (turn < 0) {
            http_json(client_fd, 500, "{\"error\":\"inference failed\"}");
            return;
        }

        json_escape(response, escaped, sizeof(escaped));
        snprintf(json_buf, sizeof(json_buf),
            "{\"response\":\"%s\",\"turn\":%d}", escaped, turn);
        http_json(client_fd, 200, json_buf);
        return;
    }

    /* POST /api/rag */
    if (strcmp(method, "POST") == 0 && strcmp(path, "/api/rag") == 0) {
        rag_cfg.enabled = json_get_bool(body, "enabled", !rag_cfg.enabled);
        if (rag_cfg.enabled && rag_cfg.n_chunks == 0)
            rag_cfg.n_chunks = rag_load_manifest(&rag_cfg);
        snprintf(json_buf, sizeof(json_buf),
            "{\"rag\":%s,\"rag_chunks\":%d}",
            rag_cfg.enabled ? "true" : "false", rag_cfg.n_chunks);
        http_json(client_fd, 200, json_buf);
        return;
    }

    /* POST /api/ingest */
    if (strcmp(method, "POST") == 0 && strcmp(path, "/api/ingest") == 0) {
        char file_path[512], text_buf[MAX_INPUT];
        int have_file = (json_get_string(body, "file", file_path, sizeof(file_path)) != NULL);
        int have_text = (json_get_string(body, "text", text_buf, sizeof(text_buf)) != NULL);

        if (!have_file && !have_text) {
            http_json(client_fd, 400, "{\"error\":\"need 'file' or 'text'\"}");
            return;
        }

        const char *ingest_path = file_path;
        if (have_text) {
            ingest_path = RAG_BASE "/learn_server_tmp.txt";
            FILE *tf = fopen(ingest_path, "w");
            if (!tf) {
                http_json(client_fd, 500, "{\"error\":\"cannot write temp file\"}");
                return;
            }
            fputs(text_buf, tf);
            fclose(tf);
        }

        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "taskset c0 %s ingest %s %s %s %s 2>&1",
            rag_cfg.search_bin, ingest_path, rag_cfg.index_dir,
            rag_cfg.embedding_bin, rag_cfg.colbert_model);
        int ret = system(cmd);

        if (have_text) remove(ingest_path);

        if (ret == 0) {
            rag_cfg.n_chunks = rag_load_manifest(&rag_cfg);
            if (!rag_cfg.enabled) rag_cfg.enabled = 1;
            /* Prefetch LLM model */
            snprintf(cmd, sizeof(cmd), "cat %s > /dev/null 2>/dev/null &", g_model_path);
            system(cmd);
        }

        snprintf(json_buf, sizeof(json_buf),
            "{\"success\":%s,\"chunks\":%d}",
            ret == 0 ? "true" : "false", rag_cfg.n_chunks);
        http_json(client_fd, 200, json_buf);
        return;
    }

    /* POST /api/save */
    if (strcmp(method, "POST") == 0 && strcmp(path, "/api/save") == 0) {
        if (!g_session_path) {
            http_json(client_fd, 400, "{\"error\":\"no session file configured\"}");
            return;
        }
        int ok = (session_save(g_session_path) == 0);
        snprintf(json_buf, sizeof(json_buf),
            "{\"success\":%s,\"turns\":%d}", ok ? "true" : "false", n_turns);
        http_json(client_fd, 200, json_buf);
        return;
    }

    /* POST /api/load */
    if (strcmp(method, "POST") == 0 && strcmp(path, "/api/load") == 0) {
        if (!g_session_path) {
            http_json(client_fd, 400, "{\"error\":\"no session file configured\"}");
            return;
        }
        int ok = (session_load(g_session_path) == 0);
        snprintf(json_buf, sizeof(json_buf),
            "{\"success\":%s,\"turns\":%d}", ok ? "true" : "false", n_turns);
        http_json(client_fd, 200, json_buf);
        return;
    }

    /* 404 */
    http_json(client_fd, 404, "{\"error\":\"not found\"}");
}

/* ---------- Main ---------- */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: moltar-server <model.gguf> [options]\n\n"
            "Options:\n"
            "  -t N         threads (default: 2)\n"
            "  -c N         context size (default: 4096)\n"
            "  -n N         max tokens per response (default: %d)\n"
            "  -p PORT      HTTP port (default: %d)\n"
            "  --temp F     temperature (default: 0.7)\n"
            "  --rag        enable ColBERT RAG\n"
            "  --session F  session file for persistence\n"
            "  --no-memory  disable working memory\n",
            MAX_RESPONSE, DEFAULT_PORT);
        return 1;
    }

    g_model_path = argv[1];
    int n_threads = 2, n_ctx = 4096, port = DEFAULT_PORT;
    float temp = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    g_session_path = NULL;

    rag_cfg.enabled = 0;
    snprintf(rag_cfg.colbert_model, sizeof(rag_cfg.colbert_model), "%s", RAG_COLBERT);
    snprintf(rag_cfg.embedding_bin, sizeof(rag_cfg.embedding_bin), "%s", RAG_EMBEDDING);
    snprintf(rag_cfg.search_bin, sizeof(rag_cfg.search_bin), "%s", RAG_SEARCH);
    snprintf(rag_cfg.index_dir, sizeof(rag_cfg.index_dir), "%s", RAG_INDEX);
    rag_cfg.top_k = RAG_TOP_K;
    rag_cfg.n_chunks = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) g_n_predict = atoi(argv[++i]);
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--rag") == 0) rag_cfg.enabled = 1;
        else if (strcmp(argv[i], "--session") == 0 && i + 1 < argc) g_session_path = argv[++i];
        else if (strcmp(argv[i], "--no-memory") == 0) g_use_memory = 0;
        else if (strcmp(argv[i], "--rag-index") == 0 && i + 1 < argc)
            snprintf(rag_cfg.index_dir, sizeof(rag_cfg.index_dir), "%s", argv[++i]);
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }

    /* Initialize RAG */
    if (rag_cfg.enabled) {
        rag_cfg.n_chunks = rag_load_manifest(&rag_cfg);
        if (rag_cfg.n_chunks > 0)
            fprintf(stderr, "[server] RAG: %d chunks in %s\n", rag_cfg.n_chunks, rag_cfg.index_dir);
        else { fprintf(stderr, "[server] WARN: RAG enabled but no index\n"); rag_cfg.enabled = 0; }
    }

    /* Initialize LLM */
    fprintf(stderr, "[server] Loading backends...\n");
    ggml_backend_load_all();

    fprintf(stderr, "[server] Loading model: %s\n", g_model_path);
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(g_model_path, mp);
    if (!g_model) { fprintf(stderr, "[server] ERROR: model load failed\n"); return 1; }

    g_vocab = llama_model_get_vocab(g_model);
    g_n_embd = llama_model_n_embd(g_model);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx;
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    cp.embeddings = 1; cp.no_perf = 1;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) { fprintf(stderr, "[server] ERROR: context init failed\n"); return 1; }

    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = 1;
    g_smpl = llama_sampler_chain_init(sp);
    if (temp < 0.01f) {
        llama_sampler_chain_add(g_smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(g_smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(g_smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(g_smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(g_smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    /* Initialize memory */
    if (g_use_memory) {
        moltar_proj_init(&proj, g_n_embd, PROJ_SEED);
        if (posix_memalign(&topo_buf, 64, MAX_TURNS * sizeof(lcvdb_topo_t)) != 0 ||
            posix_memalign(&vec_buf, 64, MAX_TURNS * sizeof(lcvdb_vec_t)) != 0) {
            fprintf(stderr, "[server] ERROR: LCVDB alloc failed\n"); return 1;
        }
        lcvdb_init(&vdb, topo_buf, vec_buf, MAX_TURNS);
        if (g_session_path && session_load(g_session_path) == 0)
            fprintf(stderr, "[server] Session restored: %d turns\n", n_turns);
    }

    g_prompt_tokens = (llama_token *)malloc(MAX_PROMPT_TOKENS * sizeof(llama_token));
    g_prompt_buf = (char *)malloc(MAX_PROMPT_BUF);

    /* Start HTTP server */
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    signal(SIGPIPE, SIG_IGN);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(server_fd); return 1;
    }
    if (listen(server_fd, 4) < 0) {
        perror("listen"); close(server_fd); return 1;
    }

    fprintf(stderr, "\n[server] Moltar listening on http://0.0.0.0:%d\n", port);
    fprintf(stderr, "[server] Memory: %s | RAG: %s | Threads: %d | Temp: %.1f\n\n",
            g_use_memory ? "ON" : "OFF", rag_cfg.enabled ? "ON" : "OFF", n_threads, temp);

    /* Accept loop */
    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) continue;

        /* Read request */
        char *buf = (char *)malloc(HTTP_BUF_SIZE);
        if (!buf) { close(client_fd); continue; }

        int total = 0;
        while (total < HTTP_BUF_SIZE - 1) {
            int n = (int)read(client_fd, buf + total, HTTP_BUF_SIZE - 1 - total);
            if (n <= 0) break;
            total += n;
            buf[total] = '\0';
            /* Check if we have the full request */
            char *hdr_end = strstr(buf, "\r\n\r\n");
            if (hdr_end) {
                int hdr_len = (int)(hdr_end - buf) + 4;
                /* Check Content-Length for body */
                char *cl = strstr(buf, "Content-Length:");
                if (!cl) cl = strstr(buf, "content-length:");
                if (cl) {
                    int content_len = atoi(cl + 15);
                    if (total >= hdr_len + content_len) break;
                } else {
                    break; /* no body expected */
                }
            }
        }

        /* Parse request line */
        char method[16] = "", path[256] = "";
        sscanf(buf, "%15s %255s", method, path);

        /* Find body */
        const char *body = "";
        char *hdr_end = strstr(buf, "\r\n\r\n");
        if (hdr_end) body = hdr_end + 4;

        fprintf(stderr, "[server] %s %s (%d bytes)\n", method, path, total);
        handle_request(client_fd, method, path, body);

        close(client_fd);
        free(buf);
    }

    /* Cleanup */
    fprintf(stderr, "\n[server] Shutting down...\n");
    if (g_session_path && g_use_memory && n_turns > 0) {
        if (session_save(g_session_path) == 0)
            fprintf(stderr, "[server] Session saved: %d turns\n", n_turns);
    }

    close(server_fd);
    free(g_prompt_tokens);
    free(g_prompt_buf);
    llama_sampler_free(g_smpl);
    llama_free(g_ctx);
    llama_model_free(g_model);
    llama_backend_free();
    if (g_use_memory) { free(topo_buf); free(vec_buf); }

    return 0;
}
