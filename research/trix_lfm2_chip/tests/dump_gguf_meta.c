/*
 * dump_gguf_meta — Dump all GGUF KV metadata and tensor shapes
 * Usage: ./dump_gguf_meta <model.gguf>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747

/* GGUF value types */
enum { T_U8=0, T_I8=1, T_U16=2, T_I16=3, T_U32=4, T_I32=5,
       T_F32=6, T_BOOL=7, T_STR=8, T_ARR=9, T_U64=10, T_I64=11, T_F64=12 };

static const char *type_names[] = {
    "F32","F16","Q4_0","Q4_1","Q4_2_REMOVED","Q4_3_REMOVED",
    "Q5_0","Q5_1","Q8_0","Q8_1","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","Q8_K"
};

typedef struct { const uint8_t *d; size_t sz; size_t p; } cur_t;

static uint32_t r32(cur_t *c) { uint32_t v; memcpy(&v, c->d+c->p, 4); c->p+=4; return v; }
static uint64_t r64(cur_t *c) { uint64_t v; memcpy(&v, c->d+c->p, 8); c->p+=8; return v; }
static float rf32(cur_t *c) { float v; memcpy(&v, c->d+c->p, 4); c->p+=4; return v; }

typedef struct { const char *s; uint64_t len; } gstr_t;
static gstr_t rstr(cur_t *c) { gstr_t s; s.len = r64(c); s.s = (const char*)(c->d+c->p); c->p += s.len; return s; }

static size_t tsize(int t) {
    switch(t) { case T_U8: case T_I8: case T_BOOL: return 1;
        case T_U16: case T_I16: return 2;
        case T_U32: case T_I32: case T_F32: return 4;
        case T_U64: case T_I64: case T_F64: return 8;
        default: return 0; }
}

static void print_value(cur_t *c, int type) {
    switch (type) {
        case T_U8: printf("%u", c->d[c->p]); c->p++; break;
        case T_I8: printf("%d", (int8_t)c->d[c->p]); c->p++; break;
        case T_U16: { uint16_t v; memcpy(&v, c->d+c->p, 2); printf("%u", v); c->p+=2; } break;
        case T_I16: { int16_t v; memcpy(&v, c->d+c->p, 2); printf("%d", v); c->p+=2; } break;
        case T_U32: printf("%u", r32(c)); break;
        case T_I32: { int32_t v; memcpy(&v, c->d+c->p, 4); printf("%d", v); c->p+=4; } break;
        case T_F32: printf("%.6f", rf32(c)); break;
        case T_BOOL: printf("%s", c->d[c->p] ? "true" : "false"); c->p++; break;
        case T_STR: { gstr_t s = rstr(c); printf("\"%.*s\"", (int)s.len, s.s); } break;
        case T_U64: printf("%llu", (unsigned long long)r64(c)); break;
        case T_I64: { int64_t v; memcpy(&v, c->d+c->p, 8); printf("%lld", (long long)v); c->p+=8; } break;
        case T_F64: { double v; memcpy(&v, c->d+c->p, 8); printf("%f", v); c->p+=8; } break;
        case T_ARR: {
            uint32_t atype = r32(c);
            uint64_t alen = r64(c);
            printf("[%llu x type%u] ", (unsigned long long)alen, atype);
            if (alen <= 20 && atype != T_STR && atype != T_ARR) {
                printf("{ ");
                for (uint64_t i = 0; i < alen; i++) { print_value(c, atype); printf(" "); }
                printf("}");
            } else if (atype == T_STR) {
                printf("{ ");
                for (uint64_t i = 0; i < alen && i < 5; i++) { print_value(c, atype); printf(" "); }
                if (alen > 5) {
                    printf("... ");
                    for (uint64_t i = 5; i < alen; i++) { gstr_t s = rstr(c); (void)s; }
                }
                printf("}");
            } else {
                c->p += alen * tsize(atype);
                printf("(skipped)");
            }
        } break;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }

    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    void *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    cur_t c = { .d = data, .sz = st.st_size, .p = 0 };

    uint32_t magic = r32(&c);
    uint32_t version = r32(&c);
    uint64_t n_tensors = r64(&c);
    uint64_t n_kv = r64(&c);

    printf("GGUF v%u: %llu KV pairs, %llu tensors\n\n", version,
           (unsigned long long)n_kv, (unsigned long long)n_tensors);

    printf("=== KV Metadata ===\n");
    for (uint64_t i = 0; i < n_kv; i++) {
        gstr_t key = rstr(&c);
        uint32_t vtype = r32(&c);
        printf("  [%llu] %.*s = ", (unsigned long long)i, (int)key.len, key.s);
        print_value(&c, vtype);
        printf("\n");
    }

    printf("\n=== Tensors ===\n");
    for (uint64_t i = 0; i < n_tensors; i++) {
        gstr_t name = rstr(&c);
        uint32_t ndims = r32(&c);
        uint64_t ne[4] = {1,1,1,1};
        for (uint32_t d = 0; d < ndims; d++) ne[d] = r64(&c);
        uint32_t type = r32(&c);
        uint64_t offset = r64(&c);

        const char *tname = (type < 16) ? type_names[type] : "???";
        printf("  [%3llu] %-50.*s  %s  [", (unsigned long long)i, (int)name.len, name.s, tname);
        for (uint32_t d = 0; d < ndims; d++) {
            printf("%llu", (unsigned long long)ne[d]);
            if (d < ndims-1) printf(", ");
        }
        printf("]  offset=%llu\n", (unsigned long long)offset);
    }

    munmap(data, st.st_size);
    close(fd);
    return 0;
}
