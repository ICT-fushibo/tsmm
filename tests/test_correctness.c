/**
 * tests/test_correctness.c — TSMM 正确性验证 (serial + OMP 全 kernel)
 *
 * 用法:
 *   ./build/test_correctness                              # 全部 kernel, 1 thread
 *   ./build/test_correctness 3                            # 仅 shape 3
 *   ./build/test_correctness --kernel omp_s4_3d           # 仅 omp_s4_3d
 *   ./build/test_correctness --threads 48                 # 48 线程 (测 race)
 *   ./build/test_correctness --kernel omp_s4_3d --threads 96
 */

#include "tsmm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------- 可调参数 ---------- */
#define TOLERANCE       (1e-10)

#define TEST_Ti  64
#define TEST_Tj  128
#define TEST_Tk  64

#define MAX_ELEMS_SLOW  (100 * 1000 * 1000)

/* ---------- 全局设置 ---------- */
static int g_num_threads = 1;
static const char *g_filter_kernel = NULL;  /* NULL = 全部 */

/* ---------- kernel 注册表 ---------- */

/* Unified kernel call signature:
 * All kernels called as fn(layout,m,n,k,A,B,C,Ti,Tj,Tk,num_threads).
 * Serial / MN kernels use wrappers to adapt. */
typedef void (*kernel_fn_t)(tsmm_layout_t, int, int, int,
    const double *, const double *, double *,
    int, int, int, int);

typedef struct {
    const char *name;
    kernel_fn_t fn;
} kernel_entry_t;

/* --- Wrappers for kernels with different signatures --- */
static void _w_naive(tsmm_layout_t l, int m, int n, int k,
    const double *A, const double *B, double *C,
    int Ti, int Tj, int Tk, int nt)
    { (void)Ti;(void)Tj;(void)Tk;(void)nt; tsmm_naive(l,m,n,k,A,B,C); }

static void _w_tiled(tsmm_layout_t l, int m, int n, int k,
    const double *A, const double *B, double *C,
    int Ti, int Tj, int Tk, int nt)
    { (void)nt; tsmm_tiled(l,m,n,k,A,B,C,Ti,Tj,Tk); }

static void _w_tiled_mn(tsmm_layout_t l, int m, int n, int k,
    const double *A, const double *B, double *C,
    int Ti, int Tj, int Tk, int nt)
    { (void)Tk;(void)nt; tsmm_tiled_mn(l,m,n,k,A,B,C,Ti,Tj); }

static void _w_omp_mn(tsmm_layout_t l, int m, int n, int k,
    const double *A, const double *B, double *C,
    int Ti, int Tj, int Tk, int nt)
    { (void)Tk; tsmm_tiled_omp_mn(l,m,n,k,A,B,C,Ti,Tj,nt); }

static void _w_omp_s2_mn(tsmm_layout_t l, int m, int n, int k,
    const double *A, const double *B, double *C,
    int Ti, int Tj, int Tk, int nt)
    { (void)Tk; tsmm_tiled_omp_s2_mn(l,m,n,k,A,B,C,Ti,Tj,nt); }

static void _w_omp_s3_mn(tsmm_layout_t l, int m, int n, int k,
    const double *A, const double *B, double *C,
    int Ti, int Tj, int Tk, int nt)
    { (void)Tk; tsmm_tiled_omp_s3_mn(l,m,n,k,A,B,C,Ti,Tj,nt); }

static void _w_omp_s4_mn(tsmm_layout_t l, int m, int n, int k,
    const double *A, const double *B, double *C,
    int Ti, int Tj, int Tk, int nt)
    { (void)Tk; tsmm_tiled_omp_s4_mn(l,m,n,k,A,B,C,Ti,Tj,nt); }

static void _w_avx512_s5(tsmm_layout_t l, int m, int n, int k,
    const double *A, const double *B, double *C,
    int Ti, int Tj, int Tk, int nt)
    { (void)nt; tsmm_avx512_s5(l,m,n,k,A,B,C,Ti,Tj,Tk); }

static const kernel_entry_t g_kernels[] = {
    {"naive",          _w_naive          },
    {"tiled_3d",       _w_tiled          },
    {"tiled_mn",       _w_tiled_mn       },
    {"omp_3d",         tsmm_tiled_omp    },
    {"omp_mn",         _w_omp_mn         },
    {"omp_s2_3d",      tsmm_tiled_omp_s2 },
    {"omp_s2_mn",      _w_omp_s2_mn      },
    {"omp_s3_3d",      tsmm_tiled_omp_s3 },
    {"omp_s3_mn",      _w_omp_s3_mn      },
    {"omp_s4_3d",      tsmm_tiled_omp_s4 },
    {"omp_s4_mn",      _w_omp_s4_mn      },
    {"avx512_s5",      _w_avx512_s5      },
    {"avx512_s5_omp",  tsmm_avx512_s5_omp},
};
#define N_KERNELS (sizeof(g_kernels) / sizeof(g_kernels[0]))

/* ========================================================================== */

static int test_kernel(const kernel_entry_t *ke,
    const tsmm_shape_t *shape, tsmm_layout_t layout,
    const double *A, const double *B, const double *C_ref)
{
    int m = shape->m, n = shape->n, k = shape->k;

    double *C_my = tsmm_alloc_matrix(m, n);
    if (!C_my) { printf("  %-20s  SKIP\n", ke->name); return -1; }
    tsmm_init_zero(C_my, m, n);

    ke->fn(layout, m, n, k, A, B, C_my,
           TEST_Ti, TEST_Tj, TEST_Tk, g_num_threads);

    double max_diff = tsmm_max_abs_diff(C_my, C_ref, m, n);
    double tol = TOLERANCE * (double)k;
    if (tol < 1e-15) tol = 1e-15;

    int ok = tsmm_verify(C_my, C_ref, m, n, tol);
    printf("  %-20s  %s  (max_diff=%.3e)\n", ke->name,
           ok ? "PASS" : "FAIL", max_diff);

    tsmm_free_matrix(C_my);
    return ok;
}

/* ========================================================================== */

static int test_one(const tsmm_shape_t *shape, tsmm_layout_t layout)
{
    int m = shape->m, n = shape->n, k = shape->k;

    size_t total = (size_t)m * n;
    if (total > MAX_ELEMS_SLOW) {
        printf("  (C too large for reference — skip)\n");
        return 1;
    }

    double *A = tsmm_alloc_matrix(k, m);
    double *B = tsmm_alloc_matrix(k, n);
    double *C_ref = tsmm_alloc_matrix(m, n);
    if (!A || !B || !C_ref) {
        printf("  SKIP (alloc)\n");
        tsmm_free_matrix(A); tsmm_free_matrix(B);
        tsmm_free_matrix(C_ref); return 1;
    }
    tsmm_init_random(A, k, m);
    tsmm_init_random(B, k, n);
    tsmm_reference(layout, m, n, k, A, B, C_ref);

    printf("  tol = %.2e\n", TOLERANCE * k);

    int all_pass = 1;
    for (size_t ki = 0; ki < N_KERNELS; ki++) {
        if (g_filter_kernel && strcmp(g_kernels[ki].name, g_filter_kernel))
            continue;
        int r = test_kernel(&g_kernels[ki], shape, layout, A, B, C_ref);
        if (r == 0) all_pass = 0;
    }

    tsmm_free_matrix(A); tsmm_free_matrix(B);
    tsmm_free_matrix(C_ref);
    return all_pass;
}

/* ========================================================================== */

int main(int argc, char **argv)
{
    int shape_start = 0, shape_end = TSMM_NUM_SHAPES;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            g_num_threads = atoi(argv[++i]);
            if (g_num_threads < 1) g_num_threads = 1;
        } else if (!strcmp(argv[i], "--kernel") && i + 1 < argc) {
            g_filter_kernel = argv[++i];
        } else {
            int idx = atoi(argv[i]);
            if (idx >= 1 && idx <= TSMM_NUM_SHAPES) {
                shape_start = idx - 1; shape_end = idx;
            } else {
                fprintf(stderr, "Usage: %s [shape] [--kernel X] [--threads N]\n", argv[0]);
                return 1;
            }
        }
    }

#ifdef _OPENMP
    omp_set_num_threads(g_num_threads);
#endif

    tsmm_layout_t layouts[]       = { TSMM_ROW_MAJOR, TSMM_COL_MAJOR };
    const char   *layout_names[]  = { "ROW_MAJOR", "COL_MAJOR" };

    int total = 0, passed = 0;

    printf("=== TSMM Correctness ===\n");
    printf("Kernels: ");
    for (size_t ki = 0; ki < N_KERNELS; ki++)
        if (!g_filter_kernel || !strcmp(g_kernels[ki].name, g_filter_kernel))
            printf("%s ", g_kernels[ki].name);
    printf("\nThreads: %d\n\n", g_num_threads);

    for (int si = shape_start; si < shape_end; si++) {
        for (int li = 0; li < 2; li++) {
            printf("--- %s  [%s] ---\n",
                   tsmm_shapes[si].name, layout_names[li]);
            total++;
            if (test_one(&tsmm_shapes[si], layouts[li]))
                passed++;
            printf("\n");
        }
    }

    printf("=== %d / %d test groups passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}

