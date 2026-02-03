/*
 * Three Thinkers Architecture
 * 
 * A55 0-1: CREATIVE  - Divergent thinking, novel approaches
 * A55 2-3: ANALYTIC  - Logical verification, structured reasoning  
 * A55 4-5: SYNTHESIS - Integration, coherence, memory alignment
 *
 * A78-0: Executive (routing)
 * A78-1: Responder (final output)
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
#define CPU_SETSIZE 1024
typedef struct { unsigned long __bits[CPU_SETSIZE / (8 * sizeof(long))]; } cpu_set_t;
#define CPU_ZERO(set) memset(set, 0, sizeof(cpu_set_t))
#define CPU_SET(cpu, set) ((set)->__bits[(cpu) / (8 * sizeof(long))] |= (1UL << ((cpu) % (8 * sizeof(long)))))
static int sched_setaffinity(pid_t pid, size_t size, const cpu_set_t* set) { (void)pid; (void)size; (void)set; return 0; }
#endif

/*============================================================================
 * Core Assignments
 *============================================================================*/

#define CPU_EXECUTIVE    6
#define CPU_RESPONDER    7
#define CPU_CREATIVE_0   0
#define CPU_CREATIVE_1   1
#define CPU_ANALYTIC_0   2
#define CPU_ANALYTIC_1   3
#define CPU_SYNTHESIS_0  4
#define CPU_SYNTHESIS_1  5

/*============================================================================
 * Data Structures
 *============================================================================*/

#define MAX_THOUGHT_LEN 512
#define MAX_MEMORY_LEN 256

typedef struct {
    char text[MAX_THOUGHT_LEN];
    float confidence;
    uint64_t generation_time_us;
} thought_t;

typedef struct {
    char query[MAX_THOUGHT_LEN];
    
    thought_t creative;
    thought_t analytic;
    thought_t synthesis;
    
    /* Memory context */
    char user_traits[MAX_MEMORY_LEN];
    char past_interactions[MAX_MEMORY_LEN];
    
    /* Coordination */
    atomic_int creative_done;
    atomic_int analytic_done;
    atomic_int synthesis_done;
    
    /* Coherence check */
    int coherence_pass;
    char coherence_note[MAX_THOUGHT_LEN];
    
} reasoning_context_t;

/*============================================================================
 * Utilities
 *============================================================================*/

static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

static void pin_to_cpus(int cpu0, int cpu1) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu0, &set);
    CPU_SET(cpu1, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

static void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

/*============================================================================
 * CREATIVE Thinker (A55 0-1)
 * 
 * Divergent thinking: What novel approaches exist?
 * Lateral connections: What's a different way to see this?
 * Exploration: What possibilities haven't been considered?
 *============================================================================*/

void* creative_thinker(void* arg) {
    reasoning_context_t* ctx = (reasoning_context_t*)arg;
    pin_to_cpus(CPU_CREATIVE_0, CPU_CREATIVE_1);
    
    uint64_t t0 = get_time_us();
    
    /* Simulate creative reasoning (would be actual LLM in production) */
    usleep(80000);  /* 80ms - thinking time */
    
    /* Generate creative perspective */
    if (strstr(ctx->query, "conversation") || strstr(ctx->query, "talk")) {
        snprintf(ctx->creative.text, MAX_THOUGHT_LEN,
            "CREATIVE PERSPECTIVE: What if you reframe this not as a 'difficult' "
            "conversation but as an opportunity to deepen the relationship? "
            "Consider approaching with curiosity rather than defensiveness. "
            "What would happen if you asked for their mentorship on this issue?");
        ctx->creative.confidence = 0.75f;
    } else if (strstr(ctx->query, "decision") || strstr(ctx->query, "choose")) {
        snprintf(ctx->creative.text, MAX_THOUGHT_LEN,
            "CREATIVE PERSPECTIVE: Instead of choosing between options, "
            "explore: Is there a third path that combines the best elements? "
            "What would you do if failure wasn't possible? "
            "What would your future self advise?");
        ctx->creative.confidence = 0.80f;
    } else {
        snprintf(ctx->creative.text, MAX_THOUGHT_LEN,
            "CREATIVE PERSPECTIVE: Let's explore unexpected angles. "
            "What assumptions are we making that might not be true? "
            "How would someone from a completely different background approach this?");
        ctx->creative.confidence = 0.70f;
    }
    
    ctx->creative.generation_time_us = get_time_us() - t0;
    atomic_store(&ctx->creative_done, 1);
    
    return NULL;
}

/*============================================================================
 * ANALYTIC Thinker (A55 2-3)
 *
 * Logical verification: What are the facts?
 * Risk assessment: What could go wrong?
 * Structure: What's the step-by-step approach?
 *============================================================================*/

void* analytic_thinker(void* arg) {
    reasoning_context_t* ctx = (reasoning_context_t*)arg;
    pin_to_cpus(CPU_ANALYTIC_0, CPU_ANALYTIC_1);
    
    uint64_t t0 = get_time_us();
    
    /* Simulate analytical reasoning */
    usleep(100000);  /* 100ms - more methodical */
    
    /* Generate analytical perspective */
    if (strstr(ctx->query, "conversation") || strstr(ctx->query, "talk")) {
        snprintf(ctx->analytic.text, MAX_THOUGHT_LEN,
            "ANALYTICAL ASSESSMENT:\n"
            "1. TIMING: Consider their schedule and stress level\n"
            "2. EVIDENCE: Prepare specific examples, not generalizations\n"
            "3. PERSPECTIVE: Anticipate their viewpoint and concerns\n"
            "4. RISKS: Worst case is relationship damage; mitigate with respect\n"
            "5. STRUCTURE: State intent, share perspective, ask for theirs");
        ctx->analytic.confidence = 0.85f;
    } else if (strstr(ctx->query, "decision") || strstr(ctx->query, "choose")) {
        snprintf(ctx->analytic.text, MAX_THOUGHT_LEN,
            "ANALYTICAL ASSESSMENT:\n"
            "1. CRITERIA: List what matters most (rank by importance)\n"
            "2. OPTIONS: Enumerate all choices including 'do nothing'\n"
            "3. EVIDENCE: What data supports each option?\n"
            "4. REVERSIBILITY: Which choices can be undone?\n"
            "5. TIMELINE: When must you decide? What's the cost of waiting?");
        ctx->analytic.confidence = 0.90f;
    } else {
        snprintf(ctx->analytic.text, MAX_THOUGHT_LEN,
            "ANALYTICAL ASSESSMENT:\n"
            "1. Define the core problem precisely\n"
            "2. Identify constraints and resources\n"
            "3. List potential solutions\n"
            "4. Evaluate trade-offs\n"
            "5. Recommend based on evidence");
        ctx->analytic.confidence = 0.75f;
    }
    
    ctx->analytic.generation_time_us = get_time_us() - t0;
    atomic_store(&ctx->analytic_done, 1);
    
    return NULL;
}

/*============================================================================
 * SYNTHESIS Thinker (A55 4-5)
 *
 * Integration: Merge creative and analytical
 * Memory check: Is this coherent with user's identity/values?
 * Delta analysis: Does this response align with past behavior?
 *============================================================================*/

void* synthesis_thinker(void* arg) {
    reasoning_context_t* ctx = (reasoning_context_t*)arg;
    pin_to_cpus(CPU_SYNTHESIS_0, CPU_SYNTHESIS_1);
    
    /* Wait for both thinkers to complete */
    while (!atomic_load(&ctx->creative_done) || !atomic_load(&ctx->analytic_done)) {
        usleep(5000);  /* Check every 5ms */
    }
    
    uint64_t t0 = get_time_us();
    
    /* Simulate synthesis */
    usleep(60000);  /* 60ms - integration */
    
    /* Merge perspectives */
    snprintf(ctx->synthesis.text, MAX_THOUGHT_LEN,
        "SYNTHESIZED RESPONSE:\n\n"
        "Combining creative reframing with analytical structure:\n\n"
        "Approach this as a growth opportunity (creative) while being prepared "
        "with specifics (analytical). Lead with curiosity - ask for their "
        "perspective before stating yours. Have concrete examples ready, "
        "but present them as seeking guidance rather than making accusations.\n\n"
        "This balances relationship-building with practical effectiveness.");
    ctx->synthesis.confidence = (ctx->creative.confidence + ctx->analytic.confidence) / 2.0f;
    
    /* Memory coherence check */
    usleep(30000);  /* 30ms - memory lookup */
    
    /* Check against user traits (simulated) */
    if (strlen(ctx->user_traits) > 0) {
        if (strstr(ctx->user_traits, "direct")) {
            ctx->coherence_pass = 1;
            snprintf(ctx->coherence_note, MAX_THOUGHT_LEN,
                "COHERENCE CHECK: User values directness. "
                "Response maintains honesty while adding tact. ALIGNED.");
        } else if (strstr(ctx->user_traits, "avoidant")) {
            ctx->coherence_pass = 0;
            snprintf(ctx->coherence_note, MAX_THOUGHT_LEN,
                "COHERENCE WARNING: User tends to avoid conflict. "
                "Consider if they need encouragement to engage, or permission to wait.");
        }
    } else {
        ctx->coherence_pass = 1;
        snprintf(ctx->coherence_note, MAX_THOUGHT_LEN,
            "COHERENCE CHECK: No conflicting user traits detected. ALIGNED.");
    }
    
    ctx->synthesis.generation_time_us = get_time_us() - t0;
    atomic_store(&ctx->synthesis_done, 1);
    
    return NULL;
}

/*============================================================================
 * Executive Router (A78-0)
 *============================================================================*/

void executive_process(reasoning_context_t* ctx) {
    pin_to_cpu(CPU_EXECUTIVE);
    
    uint64_t t0 = get_time_us();
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                    THREE THINKERS ARCHITECTURE                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Query: %-58s ║\n", ctx->query);
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    /* Dispatch to all three thinkers in parallel */
    printf("[EXECUTIVE] Dispatching to Creative, Analytic, and Synthesis...\n\n");
    
    pthread_t creative_t, analytic_t, synthesis_t;
    
    pthread_create(&creative_t, NULL, creative_thinker, ctx);
    pthread_create(&analytic_t, NULL, analytic_thinker, ctx);
    pthread_create(&synthesis_t, NULL, synthesis_thinker, ctx);
    
    /* Wait for all to complete */
    pthread_join(creative_t, NULL);
    pthread_join(analytic_t, NULL);
    pthread_join(synthesis_t, NULL);
    
    uint64_t total_time = get_time_us() - t0;
    
    /* Display results */
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ CREATIVE (A55 0-1) - %.1fms, confidence: %.0f%%                      \n", 
           ctx->creative.generation_time_us / 1000.0f, ctx->creative.confidence * 100);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ %s\n", ctx->creative.text);
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ ANALYTIC (A55 2-3) - %.1fms, confidence: %.0f%%                      \n",
           ctx->analytic.generation_time_us / 1000.0f, ctx->analytic.confidence * 100);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ %s\n", ctx->analytic.text);
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ SYNTHESIS (A55 4-5) - %.1fms, confidence: %.0f%%                     \n",
           ctx->synthesis.generation_time_us / 1000.0f, ctx->synthesis.confidence * 100);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ %s\n", ctx->synthesis.text);
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ %s\n", ctx->coherence_note);
    printf("│ Coherence: %s\n", ctx->coherence_pass ? "✓ PASS" : "⚠ WARNING");
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("Total thinking time: %.1fms (parallel execution)\n", total_time / 1000.0f);
    printf("Creative: %.1fms | Analytic: %.1fms | Synthesis: %.1fms (waited + processed)\n",
           ctx->creative.generation_time_us / 1000.0f,
           ctx->analytic.generation_time_us / 1000.0f,
           ctx->synthesis.generation_time_us / 1000.0f);
    printf("═══════════════════════════════════════════════════════════════════════\n");
}

/*============================================================================
 * Main
 *============================================================================*/

int main(int argc, char** argv) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║            THREE THINKERS - Cognitive Architecture v2             ║\n");
    printf("║                                                                   ║\n");
    printf("║   A55 0-1: CREATIVE    (Right Brain - Divergent Thinking)        ║\n");
    printf("║   A55 2-3: ANALYTIC    (Left Brain - Logical Reasoning)          ║\n");
    printf("║   A55 4-5: SYNTHESIS   (Corpus Callosum - Integration)           ║\n");
    printf("║                                                                   ║\n");
    printf("║   A78-0: Executive     A78-1: Responder                          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    /* Test case 1: Difficult conversation */
    {
        reasoning_context_t ctx = {0};
        strncpy(ctx.query, "How should I approach a difficult conversation with my boss?", MAX_THOUGHT_LEN);
        strncpy(ctx.user_traits, "direct, values honesty, good past mentorship", MAX_MEMORY_LEN);
        executive_process(&ctx);
    }
    
    usleep(500000);
    
    /* Test case 2: Decision making */
    {
        reasoning_context_t ctx = {0};
        strncpy(ctx.query, "I need to decide between two job offers", MAX_THOUGHT_LEN);
        strncpy(ctx.user_traits, "analytical, risk-averse", MAX_MEMORY_LEN);
        executive_process(&ctx);
    }
    
    return 0;
}
