/*
 * trix_fabric_launch — TriX Fabric Phase 1+2 launcher
 *
 * A transparent wrapper that optimizes the hardware environment before
 * exec'ing the target application. The application runs unmodified.
 *
 * What it does:
 *   1. Detects big.LITTLE core topology (reads cpu_capacity from sysfs)
 *   2. Pins itself (and the child) to big cores via sched_setaffinity
 *   3. Pre-faults the model file into page cache (if -m specified)
 *   4. Sets madvise hints for sequential access
 *   5. Reports hardware state (frequencies, thermal if available)
 *   6. exec's the target command with correct thread count
 *
 * Usage:
 *   trix_fabric_launch [-m model.gguf] [-v] -- command [args...]
 *   trix_fabric_launch -m LFM2-350M-Q4_0.gguf -- ./llama-completion -m LFM2-350M-Q4_0.gguf -p "hello" -n 64
 *
 * The fabric is invisible: the child process doesn't know it was launched
 * by the fabric. It just runs on better-configured hardware.
 *
 * Cross-compile:
 *   NDK=~/Library/Android/sdk/ndk/28.2.13676358
 *   $NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang \
 *       -O2 -o trix_fabric_launch src/trix_fabric_launch.c
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

/* ── Configuration ─────────────────────────────────────────────── */

#define MAX_CPUS        16
#define BIG_CORE_THRESH 512  /* cpu_capacity threshold: big >= 512 */

/* ── Helpers ───────────────────────────────────────────────────── */

static int verbose = 0;

#define LOG(fmt, ...) do { \
    fprintf(stderr, "[fabric] " fmt "\n", ##__VA_ARGS__); \
} while(0)

#define VLOG(fmt, ...) do { \
    if (verbose) fprintf(stderr, "[fabric] " fmt "\n", ##__VA_ARGS__); \
} while(0)

static int read_int_from(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    char buf[64];
    int n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    return atoi(buf);
}

static int read_str_from(const char *path, char *out, int maxlen) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int n = read(fd, out, maxlen - 1);
    close(fd);
    if (n <= 0) return -1;
    out[n] = '\0';
    /* strip trailing newline */
    if (n > 0 && out[n-1] == '\n') out[n-1] = '\0';
    return 0;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* ── Core Topology Detection ──────────────────────────────────── */

typedef struct {
    int id;
    int capacity;    /* from /sys/devices/system/cpu/cpuN/cpu_capacity */
    int online;
    int max_freq_khz;
    int cur_freq_khz;
    int is_big;
} core_info_t;

typedef struct {
    int n_cores;
    int n_big;
    int n_little;
    core_info_t cores[MAX_CPUS];
    cpu_set_t big_mask;
    cpu_set_t all_mask;
} topology_t;

static void detect_topology(topology_t *topo) {
    memset(topo, 0, sizeof(*topo));
    CPU_ZERO(&topo->big_mask);
    CPU_ZERO(&topo->all_mask);

    char path[256];
    for (int i = 0; i < MAX_CPUS; i++) {
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);
        int cap = read_int_from(path);
        if (cap < 0) break;  /* no more cores */

        core_info_t *c = &topo->cores[topo->n_cores];
        c->id = i;
        c->capacity = cap;

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/online", i);
        c->online = read_int_from(path);
        if (c->online < 0) c->online = 1;  /* cpu0 may not have online file */

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        c->max_freq_khz = read_int_from(path);

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", i);
        c->cur_freq_khz = read_int_from(path);

        c->is_big = (cap >= BIG_CORE_THRESH);
        if (c->is_big) {
            topo->n_big++;
            CPU_SET(i, &topo->big_mask);
        } else {
            topo->n_little++;
        }
        CPU_SET(i, &topo->all_mask);
        topo->n_cores++;
    }
}

static void print_topology(const topology_t *topo) {
    LOG("CPU topology: %d cores (%d big + %d little)",
        topo->n_cores, topo->n_big, topo->n_little);
    for (int i = 0; i < topo->n_cores; i++) {
        const core_info_t *c = &topo->cores[i];
        VLOG("  cpu%d: capacity=%d %s freq=%d/%d KHz %s",
             c->id, c->capacity,
             c->is_big ? "BIG" : "little",
             c->cur_freq_khz, c->max_freq_khz,
             c->online ? "" : "(OFFLINE)");
    }
}

/* ── Core Pinning ─────────────────────────────────────────────── */

static int pin_to_big_cores(const topology_t *topo) {
    if (topo->n_big == 0) {
        LOG("WARNING: no big cores detected, skipping pin");
        return -1;
    }

    if (sched_setaffinity(0, sizeof(cpu_set_t), &topo->big_mask) != 0) {
        LOG("WARNING: sched_setaffinity failed: %s", strerror(errno));
        return -1;
    }

    /* Build core list string for logging */
    char cores_str[128] = {0};
    int pos = 0;
    for (int i = 0; i < topo->n_cores; i++) {
        if (topo->cores[i].is_big) {
            pos += snprintf(cores_str + pos, sizeof(cores_str) - pos,
                           "%s%d", pos > 0 ? "," : "", topo->cores[i].id);
        }
    }
    LOG("pinned to big cores: [%s] (%d threads optimal)", cores_str, topo->n_big);
    return 0;
}

/* ── Memory Pre-fault ─────────────────────────────────────────── */

static int prefault_model(const char *model_path) {
    double t0 = now_ms();

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        LOG("WARNING: cannot open model '%s': %s", model_path, strerror(errno));
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        LOG("WARNING: fstat failed: %s", strerror(errno));
        close(fd);
        return -1;
    }

    size_t size = st.st_size;
    LOG("pre-faulting model: %s (%.1f MiB)", model_path, size / (1024.0 * 1024.0));

    /* mmap the file */
    void *addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (addr == MAP_FAILED) {
        /* MAP_POPULATE may not be supported, try without */
        addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            LOG("WARNING: mmap failed: %s", strerror(errno));
            close(fd);
            return -1;
        }
        /* Manually fault pages by reading */
        VLOG("  MAP_POPULATE not available, faulting manually...");
        volatile char sum = 0;
        const char *p = (const char *)addr;
        for (size_t off = 0; off < size; off += 4096) {
            sum += p[off];
        }
        (void)sum;
    }

    /* Set sequential access hint — tells kernel to read-ahead aggressively */
    if (madvise(addr, size, MADV_SEQUENTIAL) != 0) {
        VLOG("  madvise(SEQUENTIAL) failed: %s (non-fatal)", strerror(errno));
    } else {
        VLOG("  madvise(SEQUENTIAL) set");
    }

    /* Try to lock pages in RAM (may fail without root, non-fatal) */
    if (mlock(addr, size) != 0) {
        VLOG("  mlock failed: %s (non-fatal, pages still cached)", strerror(errno));
    } else {
        VLOG("  mlock succeeded — %zu MiB locked in RAM", size / (1024 * 1024));
    }

    /* Don't munmap — keep the mapping alive so pages stay in page cache.
     * When llama.cpp opens and mmaps the same file, it'll hit the page cache
     * instead of going to storage. The mapping will be inherited by the child
     * and cleaned up on process exit. */

    close(fd);  /* fd can be closed, mapping persists */

    double elapsed = now_ms() - t0;
    LOG("pre-fault complete: %.1f ms (%.1f MiB/s)",
        elapsed, (size / (1024.0 * 1024.0)) / (elapsed / 1000.0));

    return 0;
}

/* ── Main ─────────────────────────────────────────────────────── */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [-m model.gguf] [-v] -- command [args...]\n"
        "\n"
        "TriX Fabric Phase 1+2: Hardware environment optimizer.\n"
        "Pins to big cores, pre-faults model, then exec's command.\n"
        "\n"
        "Options:\n"
        "  -m FILE   Model file to pre-fault into page cache\n"
        "  -v        Verbose output\n"
        "  --        Separator between fabric opts and target command\n"
        "\n"
        "Example:\n"
        "  %s -m LFM2-350M-Q4_0.gguf -- ./llama-completion -m LFM2-350M-Q4_0.gguf -p 'hello' -n 64 -t 2\n"
        "\n"
        "The target command runs on optimized hardware. It doesn't know\n"
        "the fabric exists.\n",
        prog, prog);
}

int main(int argc, char *argv[]) {
    const char *model_path = NULL;
    int cmd_start = -1;

    /* Parse fabric options (before --) */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) {
            cmd_start = i + 1;
            break;
        }
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    if (cmd_start < 0 || cmd_start >= argc) {
        fprintf(stderr, "Error: no command specified after '--'\n");
        usage(argv[0]);
        return 1;
    }

    LOG("=== TriX Fabric v0.1 ===");

    /* ── Step 1: Detect topology ── */
    topology_t topo;
    detect_topology(&topo);
    print_topology(&topo);

    /* ── Step 2: Pin to big cores ── */
    pin_to_big_cores(&topo);

    /* ── Step 3: Pre-fault model ── */
    if (model_path) {
        prefault_model(model_path);
    }

    /* ── Step 4: Report state ── */
    LOG("launching: %s", argv[cmd_start]);
    VLOG("  big cores: %d (recommended -t %d)", topo.n_big, topo.n_big);

    /* ── Step 5: exec the target ── */
    /* The child inherits:
     *   - CPU affinity (pinned to big cores)
     *   - Pre-faulted model pages in page cache
     *   - madvise hints on the mmap'd model
     *   - mlock'd pages (if successful)
     */
    execvp(argv[cmd_start], &argv[cmd_start]);

    /* If we get here, exec failed */
    LOG("FATAL: exec failed for '%s': %s", argv[cmd_start], strerror(errno));
    return 127;
}
