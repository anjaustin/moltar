/*
 * Cognitive Router - Heterogeneous Inference Architecture
 * 
 * A78-0: Executive (routing, coordination)
 * A78-1: Responder (fast user-facing generation)
 * A55 Pool: Memory system (graphs, vectors, background)
 * GPU: Vector operations (embeddings, similarity)
 * 
 * Build:
 *   aarch64-linux-android30-clang -O3 -o cognitive_router cognitive_router.c -lpthread
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <stdatomic.h>
#ifdef __ANDROID__
#include <sched.h>
#else
/* For non-Android builds (macOS dev), stub out affinity */
#define CPU_SETSIZE 1024
typedef struct { unsigned long __bits[CPU_SETSIZE / (8 * sizeof(long))]; } cpu_set_t;
#define CPU_ZERO(set) memset(set, 0, sizeof(cpu_set_t))
#define CPU_SET(cpu, set) ((set)->__bits[(cpu) / (8 * sizeof(long))] |= (1UL << ((cpu) % (8 * sizeof(long)))))
static int sched_setaffinity(pid_t pid, size_t size, const cpu_set_t* set) { (void)pid; (void)size; (void)set; return 0; }
#endif

/*============================================================================
 * Configuration
 *============================================================================*/

#define CPU_EXECUTIVE    6    /* A78-0: Coordinator */
#define CPU_RESPONDER    7    /* A78-1: Fast generation */
#define CPU_MEMORY_0     0    /* A55: Graph ops */
#define CPU_MEMORY_1     1    /* A55: Graph ops */
#define CPU_MEMORY_2     2    /* A55: Vector DB */
#define CPU_MEMORY_3     3    /* A55: Vector DB */
#define CPU_MEMORY_4     4    /* A55: Summarization */
#define CPU_MEMORY_5     5    /* A55: Summarization */

#define MAX_QUERY_LEN    1024
#define MAX_CONTEXT_LEN  4096

/*============================================================================
 * Query Classification
 *============================================================================*/

typedef enum {
    QUERY_REACTIVE,      /* Simple, immediate response needed */
    QUERY_INTERACTIVE,   /* Short answer, conversational */
    QUERY_MEMORY,        /* Needs memory retrieval */
    QUERY_GENERATIVE,    /* Long-form generation */
    QUERY_BACKGROUND     /* Can be processed async */
} query_type_t;

typedef struct {
    char text[MAX_QUERY_LEN];
    query_type_t type;
    int expected_tokens;
    int needs_memory;
    int priority;  /* 0=highest */
} query_t;

/* Simple keyword-based classifier (would be ML in production) */
query_type_t classify_query(const char* text) {
    /* REACTIVE: Tool calls, simple questions */
    if (strstr(text, "what time") || strstr(text, "what's the time") ||
        strstr(text, "call ") || strstr(text, "send ") ||
        strstr(text, "open ") || strstr(text, "set ") ||
        strstr(text, "2+2") || strstr(text, "calculate")) {
        return QUERY_REACTIVE;
    }
    
    /* MEMORY: References to past conversations */
    if (strstr(text, "remember") || strstr(text, "last time") ||
        strstr(text, "we discussed") || strstr(text, "you said") ||
        strstr(text, "earlier") || strstr(text, "yesterday") ||
        strstr(text, "what did")) {
        return QUERY_MEMORY;
    }
    
    /* GENERATIVE: Long-form requests */
    if (strstr(text, "explain") || strstr(text, "describe") ||
        strstr(text, "write a") || strstr(text, "tell me about") ||
        strstr(text, "how does") || strstr(text, "why do") ||
        strstr(text, "in detail")) {
        return QUERY_GENERATIVE;
    }
    
    /* BACKGROUND: Async tasks */
    if (strstr(text, "summarize") || strstr(text, "analyze") ||
        strstr(text, "when you have time") || strstr(text, "later")) {
        return QUERY_BACKGROUND;
    }
    
    /* Default: INTERACTIVE */
    return QUERY_INTERACTIVE;
}

const char* query_type_str(query_type_t t) {
    switch (t) {
        case QUERY_REACTIVE:    return "REACTIVE";
        case QUERY_INTERACTIVE: return "INTERACTIVE";
        case QUERY_MEMORY:      return "MEMORY";
        case QUERY_GENERATIVE:  return "GENERATIVE";
        case QUERY_BACKGROUND:  return "BACKGROUND";
        default:                return "UNKNOWN";
    }
}

/*============================================================================
 * Timing Utilities
 *============================================================================*/

static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

/*============================================================================
 * CPU Affinity
 *============================================================================*/

static void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
    }
}

static const char* cpu_name(int cpu) {
    if (cpu == CPU_EXECUTIVE) return "A78-EXEC";
    if (cpu == CPU_RESPONDER) return "A78-RESP";
    if (cpu >= 0 && cpu <= 5) return "A55-MEM";
    return "UNKNOWN";
}

/*============================================================================
 * Simulated Subsystems
 *============================================================================*/

/* Simulated memory retrieval on A55 cores */
typedef struct {
    const char* query;
    char result[MAX_CONTEXT_LEN];
    int done;
    uint64_t latency_us;
} memory_request_t;

void* memory_worker(void* arg) {
    memory_request_t* req = (memory_request_t*)arg;
    pin_to_cpu(CPU_MEMORY_2);  /* Vector DB cores */
    
    uint64_t t0 = get_time_us();
    
    /* Simulate vector search + graph traversal */
    usleep(50000);  /* 50ms - realistic for vector search */
    
    /* "Retrieved" context */
    snprintf(req->result, MAX_CONTEXT_LEN,
             "[Memory: Found 3 relevant chunks about '%s'. "
             "Last discussed 2 days ago. Related entities: neural networks, backprop, gradients.]",
             req->query);
    
    req->latency_us = get_time_us() - t0;
    req->done = 1;
    
    return NULL;
}

/* Simulated LLM generation */
typedef struct {
    const char* prompt;
    const char* context;
    int max_tokens;
    int cpu;
    char response[MAX_CONTEXT_LEN];
    int tokens_generated;
    uint64_t first_token_us;
    uint64_t total_us;
    float tok_per_sec;
} generate_request_t;

void* generate_worker(void* arg) {
    generate_request_t* req = (generate_request_t*)arg;
    pin_to_cpu(req->cpu);
    
    uint64_t t0 = get_time_us();
    
    /* Simulate first token latency based on core type */
    if (req->cpu == CPU_RESPONDER) {
        usleep(23000);  /* 23ms - A78 measured */
    } else {
        usleep(590000); /* 590ms - A55 measured */
    }
    
    req->first_token_us = get_time_us() - t0;
    
    /* Simulate token generation */
    float tok_s = (req->cpu == CPU_RESPONDER) ? 48.0f : 18.0f;
    int tokens = req->max_tokens;
    
    /* Simulate generation time */
    int gen_time_us = (int)(tokens / tok_s * 1000000);
    usleep(gen_time_us);
    
    req->tokens_generated = tokens;
    req->total_us = get_time_us() - t0;
    req->tok_per_sec = (float)tokens / (req->total_us / 1000000.0f);
    
    /* Generate fake response */
    snprintf(req->response, MAX_CONTEXT_LEN,
             "[Generated %d tokens on %s in %.1fms @ %.1f tok/s]",
             tokens, cpu_name(req->cpu), req->total_us / 1000.0f, req->tok_per_sec);
    
    return NULL;
}

/*============================================================================
 * Executive Router
 *============================================================================*/

typedef struct {
    /* Stats */
    int queries_processed;
    uint64_t total_latency_us;
    int reactive_count;
    int memory_count;
    int generative_count;
} executive_state_t;

void executive_process_query(executive_state_t* state, query_t* query) {
    uint64_t t0 = get_time_us();
    
    printf("\n");
    printf("=== EXECUTIVE (A78-0) ===\n");
    printf("Query: \"%s\"\n", query->text);
    printf("Classification: %s\n", query_type_str(query->type));
    
    switch (query->type) {
        case QUERY_REACTIVE:
        case QUERY_INTERACTIVE: {
            /* Fast path: Direct to A78-1 Responder */
            printf("Route: FAST PATH -> A78-1 Responder\n");
            
            generate_request_t gen = {
                .prompt = query->text,
                .context = NULL,
                .max_tokens = (query->type == QUERY_REACTIVE) ? 10 : 50,
                .cpu = CPU_RESPONDER
            };
            
            pthread_t t;
            pthread_create(&t, NULL, generate_worker, &gen);
            pthread_join(t, NULL);
            
            printf("\nResponse: %s\n", gen.response);
            printf("First token: %.1f ms\n", gen.first_token_us / 1000.0f);
            
            state->reactive_count++;
            break;
        }
        
        case QUERY_MEMORY: {
            /* Memory path: A55 retrieval, then A78-1 generation */
            printf("Route: MEMORY PATH -> A55 Pool -> A78-1 Responder\n");
            
            /* Step 1: Memory retrieval on A55 */
            printf("\n[1] Memory retrieval on A55 pool...\n");
            memory_request_t mem = { .query = query->text, .done = 0 };
            
            pthread_t mem_t;
            pthread_create(&mem_t, NULL, memory_worker, &mem);
            pthread_join(mem_t, NULL);
            
            printf("    Retrieved in %.1f ms\n", mem.latency_us / 1000.0f);
            printf("    Context: %s\n", mem.result);
            
            /* Step 2: Generate with context on A78 */
            printf("\n[2] Generation on A78-1...\n");
            generate_request_t gen = {
                .prompt = query->text,
                .context = mem.result,
                .max_tokens = 100,
                .cpu = CPU_RESPONDER
            };
            
            pthread_t gen_t;
            pthread_create(&gen_t, NULL, generate_worker, &gen);
            pthread_join(gen_t, NULL);
            
            printf("    %s\n", gen.response);
            printf("    First token: %.1f ms\n", gen.first_token_us / 1000.0f);
            
            state->memory_count++;
            break;
        }
        
        case QUERY_GENERATIVE: {
            /* Generative path: A78-1 starts fast, could handoff to A55 */
            printf("Route: GENERATIVE PATH -> A78-1 (fast start)\n");
            
            /* For demo: just use A78-1 for full generation */
            /* In production: handoff to A55 after first ~20 tokens */
            generate_request_t gen = {
                .prompt = query->text,
                .context = NULL,
                .max_tokens = 200,
                .cpu = CPU_RESPONDER
            };
            
            pthread_t t;
            pthread_create(&t, NULL, generate_worker, &gen);
            pthread_join(t, NULL);
            
            printf("\nResponse: %s\n", gen.response);
            printf("First token: %.1f ms (user sees response immediately)\n", 
                   gen.first_token_us / 1000.0f);
            
            state->generative_count++;
            break;
        }
        
        case QUERY_BACKGROUND: {
            /* Background path: Queue for A55, return immediately */
            printf("Route: BACKGROUND PATH -> A55 Pool (async)\n");
            printf("Response: \"I'll work on that and let you know.\"\n");
            printf("(Task queued for A55 processing)\n");
            break;
        }
    }
    
    uint64_t total = get_time_us() - t0;
    state->queries_processed++;
    state->total_latency_us += total;
    
    printf("\nTotal query time: %.1f ms\n", total / 1000.0f);
    printf("================================================\n");
}

/*============================================================================
 * Demo
 *============================================================================*/

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║         COGNITIVE ARCHITECTURE - Proof of Concept             ║\n");
    printf("║                                                               ║\n");
    printf("║  A78-0: Executive    A78-1: Responder    A55: Memory Pool    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    /* Pin executive to A78-0 */
    pin_to_cpu(CPU_EXECUTIVE);
    printf("\nExecutive pinned to CPU %d (%s)\n", CPU_EXECUTIVE, cpu_name(CPU_EXECUTIVE));
    
    executive_state_t state = {0};
    
    /* Test queries demonstrating different paths */
    const char* test_queries[] = {
        "what's 2+2?",                                    /* REACTIVE */
        "what did we discuss about neural networks?",     /* MEMORY */
        "explain how transformers work in detail",        /* GENERATIVE */
        "hello, how are you?",                            /* INTERACTIVE */
        "summarize our conversation when you have time",  /* BACKGROUND */
    };
    
    int n_queries = sizeof(test_queries) / sizeof(test_queries[0]);
    
    for (int i = 0; i < n_queries; i++) {
        query_t q;
        strncpy(q.text, test_queries[i], MAX_QUERY_LEN - 1);
        q.text[MAX_QUERY_LEN - 1] = '\0';
        q.type = classify_query(q.text);
        
        executive_process_query(&state, &q);
        
        usleep(500000);  /* Pause between queries for readability */
    }
    
    /* Summary */
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                         SUMMARY                               ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Queries processed: %-5d                                     ║\n", state.queries_processed);
    printf("║  Avg latency:       %-6.1f ms                                 ║\n", 
           (float)state.total_latency_us / state.queries_processed / 1000.0f);
    printf("║                                                               ║\n");
    printf("║  Reactive/Interactive: %-3d  (fast path, <25ms first token)   ║\n", state.reactive_count);
    printf("║  Memory queries:       %-3d  (A55 retrieval + A78 gen)        ║\n", state.memory_count);
    printf("║  Generative:           %-3d  (A78 fast start)                 ║\n", state.generative_count);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
