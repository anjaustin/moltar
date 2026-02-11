/*
 * cpu_baseline.c - CPU baseline for comparison with GPU workloads
 * Tests the same operations on CPU to measure GPU speedup
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

static volatile float sink_f;
static volatile uint8_t sink_u8;
static volatile uint32_t sink_u32;

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* SHA-256 on CPU */
#define RR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z) (((x)&(y))^((~(x))&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define S0(x) (RR(x,2)^RR(x,13)^RR(x,22))
#define S1(x) (RR(x,6)^RR(x,11)^RR(x,25))
#define s0(x) (RR(x,7)^RR(x,18)^((x)>>3))
#define s1(x) (RR(x,17)^RR(x,19)^((x)>>10))

static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static void sha256_block(const uint32_t *msg, uint32_t *hash) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++) W[i] = msg[i];
    for (int i = 16; i < 64; i++) W[i] = s1(W[i-2]) + W[i-7] + s0(W[i-15]) + W[i-16];
    uint32_t a=0x6a09e667, b=0xbb67ae85, c=0x3c6ef372, d=0xa54ff53a;
    uint32_t e=0x510e527f, f=0x9b05688c, g=0x1f83d9ab, h=0x5be0cd19;
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + S1(e) + CH(e,f,g) + K[i] + W[i];
        uint32_t T2 = S0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;
    }
    hash[0]=a+0x6a09e667; hash[1]=b+0xbb67ae85;
    hash[2]=c+0x3c6ef372; hash[3]=d+0xa54ff53a;
    hash[4]=e+0x510e527f; hash[5]=f+0x9b05688c;
    hash[6]=g+0x1f83d9ab; hash[7]=h+0x5be0cd19;
}

int main(void) {
    printf("=== CPU Baseline Benchmarks (single thread) ===\n\n");

    /* [1] 3x3 Gaussian Blur (1080p float) */
    {
        int W = 1920, H = 1080;
        float *in = malloc(W*H*sizeof(float));
        float *out = malloc(W*H*sizeof(float));
        float filt[] = {1/16.f,2/16.f,1/16.f,2/16.f,4/16.f,2/16.f,1/16.f,2/16.f,1/16.f};
        for (int i = 0; i < W*H; i++) in[i] = (float)(rand()%256)/255.f;
        
        int runs = 10;
        double t0 = get_time_ms();
        for (int r = 0; r < runs; r++) {
            for (int y = 1; y < H-1; y++)
                for (int x = 1; x < W-1; x++) {
                    float s = 0;
                    for (int dy=-1; dy<=1; dy++)
                        for (int dx=-1; dx<=1; dx++)
                            s += in[(y+dy)*W+(x+dx)] * filt[(dy+1)*3+(dx+1)];
                    out[y*W+x] = s;
                }
            sink_f = out[W*H/2];
        }
        double ms = (get_time_ms() - t0) / runs;
        printf("[1] 3x3 Blur 1080p (CPU 1T): %.1f ms  |  %.0f FPS\n", ms, 1000/ms);
        free(in); free(out);
    }

    /* [3] Sobel (1080p uchar) */
    {
        int W = 1920, H = 1080;
        uint8_t *in = malloc(W*H);
        uint8_t *out = malloc(W*H);
        for (int i = 0; i < W*H; i++) in[i] = rand()%256;
        
        int runs = 10;
        double t0 = get_time_ms();
        for (int r = 0; r < runs; r++) {
            for (int y = 1; y < H-1; y++)
                for (int x = 1; x < W-1; x++) {
                    int gx = -in[(y-1)*W+(x-1)] + in[(y-1)*W+(x+1)]
                           -2*in[y*W+(x-1)]     + 2*in[y*W+(x+1)]
                           - in[(y+1)*W+(x-1)] + in[(y+1)*W+(x+1)];
                    int gy = -in[(y-1)*W+(x-1)] -2*in[(y-1)*W+x] - in[(y-1)*W+(x+1)]
                           + in[(y+1)*W+(x-1)] +2*in[(y+1)*W+x] + in[(y+1)*W+(x+1)];
                    int mag = abs(gx) + abs(gy);
                    out[y*W+x] = mag > 255 ? 255 : mag;
                }
            sink_u8 = out[W*H/2];
        }
        double ms = (get_time_ms() - t0) / runs;
        printf("[3] Sobel 1080p (CPU 1T): %.1f ms  |  %.0f FPS\n", ms, 1000/ms);
        free(in); free(out);
    }

    /* [4] RGB->YUV (4K) */
    {
        int N = 3840 * 2160;
        uint8_t *rgb = malloc(N*4);
        uint8_t *yuv = malloc(N*4);
        for (int i = 0; i < N*4; i++) rgb[i] = rand()%256;
        
        int runs = 5;
        double t0 = get_time_ms();
        for (int r = 0; r < runs; r++) {
            for (int i = 0; i < N; i++) {
                float R = rgb[i*4], G = rgb[i*4+1], B = rgb[i*4+2];
                float Y = 0.299f*R + 0.587f*G + 0.114f*B;
                float U = -0.169f*R - 0.331f*G + 0.500f*B + 128.f;
                float V = 0.500f*R - 0.419f*G - 0.081f*B + 128.f;
                yuv[i*4] = Y > 255 ? 255 : (Y < 0 ? 0 : (uint8_t)Y);
                yuv[i*4+1] = U > 255 ? 255 : (U < 0 ? 0 : (uint8_t)U);
                yuv[i*4+2] = V > 255 ? 255 : (V < 0 ? 0 : (uint8_t)V);
                yuv[i*4+3] = rgb[i*4+3];
            }
            sink_u8 = yuv[N*2];
        }
        double ms = (get_time_ms() - t0) / runs;
        printf("[4] RGB->YUV 4K (CPU 1T): %.1f ms  |  %.0f FPS\n", ms, 1000/ms);
        free(rgb); free(yuv);
    }

    /* [6] SHA-256 batch */
    {
        int N = 256 * 1024;
        uint32_t *msgs = malloc(N * 64);
        uint32_t *hashes = malloc(N * 32);
        for (int i = 0; i < N*16; i++) msgs[i] = rand();
        
        double t0 = get_time_ms();
        for (int i = 0; i < N; i++) {
            sha256_block(msgs + i*16, hashes + i*8);
        }
        double ms = get_time_ms() - t0;
        double mhash = (double)N / (ms/1000) / 1e6;
        printf("[6] SHA-256 %dK msgs (CPU 1T): %.1f ms  |  %.1f MHash/s\n", N/1024, ms, mhash);
        free(msgs); free(hashes);
    }

    /* [9] SGEMM 512x512x512 (FP32 tiled, single thread) */
    {
        int M=512, N=512, KK=512;
        float *A = malloc(M*KK*4), *B = malloc(KK*N*4), *C = calloc(M*N, 4);
        for (int i = 0; i < M*KK; i++) A[i] = 0.5f;
        for (int i = 0; i < KK*N; i++) B[i] = 0.5f;
        
        double t0 = get_time_ms();
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float s = 0;
                for (int k = 0; k < KK; k++) s += A[i*KK+k] * B[k*N+j];
                C[i*N+j] = s;
            }
        sink_f = C[M*N/2];
        double ms = get_time_ms() - t0;
        double gflops = 2.0*M*N*KK / (ms/1000) / 1e9;
        printf("[9] SGEMM 512^3 (CPU 1T): %.1f ms  |  %.2f GFLOPS\n", ms, gflops);
        free(A); free(B); free(C);
    }

    printf("\n=== CPU Baseline Complete ===\n");
    return 0;
}
