/**
 * src/tsmm_utils.c — 工具函数：内存管理、计时、校验、性能统计。
 */

#include "tsmm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Platform detection for timing and aligned allocation */
#include <time.h>             /* clock(), clock_gettime() — 所有平台 */
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <malloc.h>         /* _aligned_malloc, _aligned_free */
#endif

/* ==========================================================================
 *  预定义问题规模  (m, n, k)
 * ========================================================================== */
const tsmm_shape_t tsmm_shapes[TSMM_NUM_SHAPES] = {
    {   4000,  160000,    128,  "S1: m=4000  n=160000  k=128"    },
    {      8,      16,  16000,  "S2: m=8     n=16      k=16000"  },
    {     32,   16000,     16,  "S3: m=32    n=16000   k=16"     },
    {    144,     144,    144,  "S4: m=144   n=144     k=144"    },
    {     16,   12344,     16,  "S5: m=16    n=12344   k=16"     },
    {      4,      64, 606841,  "S6: m=4     n=64      k=606841" },
    {    442,     193,     11,  "S7: m=442   n=193     k=11"     },
    {     40, 1127228,     40,  "S8: m=40    n=1127228 k=40"     },
};

/* ==========================================================================
 *  内存管理
 * ========================================================================== */

#define TSMM_ALIGNMENT 64   /* bytes — AVX-512 对齐要求 */

double* tsmm_alloc_matrix(int rows, int cols)
{
    size_t nbytes;
    /* 检查溢出 */
    if (rows <= 0 || cols <= 0) return NULL;
    if ((size_t)rows > SIZE_MAX / sizeof(double) / (size_t)cols)
        return NULL;   /* 乘法会溢出 */
    nbytes = (size_t)rows * cols * sizeof(double);

#if defined(_WIN32)
    return (double*)_aligned_malloc(nbytes, TSMM_ALIGNMENT);
#else
    /* POSIX: 使用 posix_memalign——没有 aligned_alloc 的 size-alignment 约束 */
    void *ptr = NULL;
    if (posix_memalign(&ptr, TSMM_ALIGNMENT, nbytes) != 0)
        return NULL;
    return (double*)ptr;
#endif
}

void tsmm_free_matrix(double *mat)
{
    if (!mat) return;
#if defined(_WIN32)
    _aligned_free(mat);
#else
    free(mat);
#endif
}

/* ---------- 确定性随机填充 ---------- */

void tsmm_init_random(double *mat, int rows, int cols)
{
    /* 64-bit LCG with fixed seed — same (rows,cols) → same data, everywhere */
    const uint64_t seed = 123456789ULL;
    const uint64_t a    = 6364136223846793005ULL;
    const uint64_t c    = 1442695040888963407ULL;

    size_t total = (size_t)rows * cols;
    uint64_t state = seed;

    for (size_t i = 0; i < total; i++) {
        state = state * a + c;
        /* 取高 52 位作为尾数，生成 [0, 1) 范围内的 double */
        mat[i] = (double)(state >> 12) * 0x1.0p-52;
    }
}

void tsmm_init_zero(double *mat, int rows, int cols)
{
    memset(mat, 0, (size_t)rows * cols * sizeof(double));
}

/* ==========================================================================
 *  正确性校验
 * ========================================================================== */

double tsmm_max_abs_diff(const double *C1, const double *C2,
                         int rows, int cols)
{
    if (!C1 || !C2 || rows <= 0 || cols <= 0)
        return -1.0;

    size_t total = (size_t)rows * cols;
    double max_diff = 0.0;

    for (size_t i = 0; i < total; i++) {
        double diff = fabs(C1[i] - C2[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

int tsmm_verify(const double *C1, const double *C2,
                int rows, int cols, double tol)
{
    if (!C1 || !C2 || rows <= 0 || cols <= 0)
        return 0;

    size_t total = (size_t)rows * cols;

    for (size_t i = 0; i < total; i++) {
        double abs_diff = fabs(C1[i] - C2[i]);
        double rel_base = fabs(C2[i]);
        if (rel_base < 1e-15) rel_base = 1e-15;   /* 避免除零 */
        double rel_err = abs_diff / rel_base;

        if (rel_err >= tol) {
            size_t r = i / (size_t)cols;
            size_t c = i % (size_t)cols;
            fprintf(stderr,
                    "  verify FAIL at (%zu,%zu): "
                    "C1=%.15e  C2=%.15e  diff=%.3e  rel_err=%.3e\n",
                    r, c, C1[i], C2[i], abs_diff, rel_err);
            return 0;
        }
    }
    return 1;
}

/* ==========================================================================
 *  计时
 * ========================================================================== */

double tsmm_wall_time(void)
{
#if defined(_WIN32)
    /* QueryPerformanceCounter — Windows 高精度计时器 */
    static LARGE_INTEGER freq = { 0 };
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
#elif defined(__APPLE__)
    /* macOS: clock_gettime 从 Sierra 起可用 */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    /* Linux / POSIX */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

double tsmm_cpu_time(void)
{
#if defined(_WIN32)
    return (double)clock() / (double)CLOCKS_PER_SEC;
#elif defined(__APPLE__)
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    /* Linux: CLOCK_THREAD_CPUTIME_ID — 本线程 CPU 时间 */
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* ==========================================================================
 *  性能统计
 * ========================================================================== */

tsmm_perf_t tsmm_compute_perf(int m, int n, int k,
                              double wall_time_ms, int num_threads)
{
    tsmm_perf_t p;
    memset(&p, 0, sizeof(p));

    p.m            = m;
    p.n            = n;
    p.k            = k;
    p.num_threads  = num_threads;
    p.wall_time_ms = wall_time_ms;

    if (wall_time_ms <= 0.0) {
        /* 无法计算速率指标 */
        return p;
    }

    double wall_sec = wall_time_ms * 1e-3;

    /* 计算量：每个 C[i][j] 需要 k 次乘加，每次乘加计 2 FLOP */
    p.total_flops  = 2.0 * (double)m * (double)n * (double)k;
    p.gflops       = p.total_flops / wall_sec * 1e-9;

    /* 内存流量（算法理论值，忽略 cache） */
    p.bytes_read   = ((double)k * m + (double)k * n) * sizeof(double);
    p.bytes_written = (double)m * n * sizeof(double);
    double total_bytes = p.bytes_read + p.bytes_written;
    p.mem_bandwidth_gbs = total_bytes / wall_sec * 1e-9;

    /* 算术强度 = FLOP / byte */
    if (total_bytes > 0.0)
        p.flops_per_byte = p.total_flops / total_bytes;
    else
        p.flops_per_byte = 0.0;

    /* TODO: PAPI 计数器 (L1/L2/L3 miss ratio)，目前置 0 */
    p.l1_miss_ratio = 0.0;
    p.l2_miss_ratio = 0.0;
    p.l3_miss_ratio = 0.0;

    return p;
}

/* ==========================================================================
 *  打印
 * ========================================================================== */

void tsmm_print_perf(const tsmm_perf_t *p)
{
    if (!p) return;

    printf("  Problem size: m=%d  n=%d  k=%d  (%s)\n",
           p->m, p->n, p->k,
           (p->num_threads > 1) ? "multi-thread" : "serial");

    if (p->wall_time_ms <= 0.0) {
        printf("  (no timing data)\n");
        return;
    }

    printf("  Wall time:     %10.3f ms\n",  p->wall_time_ms);
    if (p->cpu_time_ms > 0.0)
        printf("  CPU time:      %10.3f ms\n",  p->cpu_time_ms);

    /* Compute */
    printf("  Total FLOPs:   %10.3e\n",   p->total_flops);
    printf("  Throughput:    %10.3f GFLOPS\n",  p->gflops);

    /* Memory */
    printf("  Bytes read:    %10.3e  (%.2f MB)\n",
           p->bytes_read, p->bytes_read * 1e-6);
    printf("  Bytes written: %10.3e  (%.2f MB)\n",
           p->bytes_written, p->bytes_written * 1e-6);
    printf("  Mem bandwidth: %10.3f GB/s\n",    p->mem_bandwidth_gbs);
    printf("  Arith. intens: %10.3f FLOP/byte\n", p->flops_per_byte);

    /* Threads */
    printf("  Threads:       %d\n", p->num_threads);
}
