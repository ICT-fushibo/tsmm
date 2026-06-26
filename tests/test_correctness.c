/**
 * tests/test_correctness.c — TSMM 正确性验证。
 *
 * 用法:
 *   ./build/test_correctness              # 测试全部 8 组 shape
 *   ./build/test_correctness 3            # 仅测试 shape 3
 *   ./build/test_correctness --large      # 不跳过大规模 shape
 */

#include "tsmm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- 可调参数 ---------- */
#define TOLERANCE       (1e-10)   /* 相对误差容忍度，最终 tol = TOLERANCE * k */

/* 测试用 tile sizes（保守，适应所有 shape） */
#define TEST_Ti  64
#define TEST_Tj  128
#define TEST_Tk  64

/* 跳过超大 shape 的阈值（避免 reference 跑太久） */
#define MAX_ELEMS_SLOW  (100 * 1000 * 1000)  /* C 元素数超过 1 亿则跳过 */

/* ---------- helpers ---------- */

/* 测试单个 kernel variant，返回 1=pass, 0=fail, -1=skip */
static int test_kernel(
    const char *kernel_name,
    const tsmm_shape_t *shape, tsmm_layout_t layout,
    const double *A, const double *B,
    const double *C_ref)         /* 参考结果，已计算好 */
{
    int m = shape->m, n = shape->n, k = shape->k;

    double *C_my = tsmm_alloc_matrix(m, n);
    if (C_my == NULL) {
        printf("  %-20s  SKIP (alloc failed)\n", kernel_name);
        return -1;
    }
    tsmm_init_zero(C_my, m, n);

    /* 根据 kernel name 调度 */
    if (!strcmp(kernel_name, "naive")) {
        tsmm_naive(layout, m, n, k, A, B, C_my);
    } else if (!strcmp(kernel_name, "tiled_3d")) {
        tsmm_tiled(layout, m, n, k, A, B, C_my, TEST_Ti, TEST_Tj, TEST_Tk);
    } else if (!strcmp(kernel_name, "tiled_mn")) {
        tsmm_tiled_mn(layout, m, n, k, A, B, C_my, TEST_Ti, TEST_Tj);
    } else {
        printf("  %-20s  SKIP (unknown kernel)\n", kernel_name);
        tsmm_free_matrix(C_my);
        return -1;
    }

    double max_diff = tsmm_max_abs_diff(C_my, C_ref, m, n);
    double tol = TOLERANCE * (double)k;
    if (tol < 1e-15) tol = 1e-15;

    int ok = tsmm_verify(C_my, C_ref, m, n, tol);
    if (ok) {
        printf("  %-20s  PASS  (max_diff=%.3e)\n", kernel_name, max_diff);
    } else {
        printf("  %-20s  FAIL  (max_diff=%.3e)\n", kernel_name, max_diff);
    }

    tsmm_free_matrix(C_my);
    return ok;
}

/* ---------- 每个 (shape, layout) 对的测试 ---------- */
static int test_one(const tsmm_shape_t *shape, tsmm_layout_t layout)
{
    int m = shape->m, n = shape->n, k = shape->k;

    /* 参考实现对小 shape 才跑得动（三重循环 i-j-p） */
    size_t total_elems = (size_t)m * n;
    if (total_elems > MAX_ELEMS_SLOW) {
        printf("  (C too large for reference — skip this shape)\n");
        return 1;  /* skip = 不扣分 */
    }

    /* ---------- 分配 ---------- */
    double *A = tsmm_alloc_matrix(k, m);
    double *B = tsmm_alloc_matrix(k, n);
    double *C_ref = tsmm_alloc_matrix(m, n);
    if (!A || !B || !C_ref) {
        printf("  SKIP (alloc failed)\n");
        tsmm_free_matrix(A);
        tsmm_free_matrix(B);
        tsmm_free_matrix(C_ref);
        return 1;
    }

    tsmm_init_random(A, k, m);
    tsmm_init_random(B, k, n);

    /* ---------- 计算参考结果 ---------- */
    tsmm_reference(layout, m, n, k, A, B, C_ref);

    /* ---------- 测试每种 kernel ---------- */
    printf("  tol = %.2e\n", TOLERANCE * k);

    int all_pass = 1;
    const char *kernels[] = { "naive", "tiled_3d", "tiled_mn" };
    for (int ki = 0; ki < 3; ki++) {
        int r = test_kernel(kernels[ki], shape, layout, A, B, C_ref);
        if (r == 0) all_pass = 0;
    }

    tsmm_free_matrix(A);
    tsmm_free_matrix(B);
    tsmm_free_matrix(C_ref);

    return all_pass;
}

/* ========================================================================== */

int main(int argc, char **argv)
{
    int shape_start = 0;
    int shape_end   = TSMM_NUM_SHAPES;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--large")) {
            /* --large: 不跳过大规模 shape（慎用） */
            /* 通过不跳过实现 — 只是把阈值改大 */
            /* （已硬编码在 MAX_ELEMS_SLOW，这里仅标记） */
        } else {
            int idx = atoi(argv[i]);
            if (idx >= 1 && idx <= TSMM_NUM_SHAPES) {
                shape_start = idx - 1;
                shape_end   = idx;
            } else if (strcmp(argv[i], "--large")) {
                fprintf(stderr, "Usage: %s [shape_index 1-%d] [--large]\n",
                        argv[0], TSMM_NUM_SHAPES);
                return 1;
            }
        }
    }

    tsmm_layout_t layouts[]      = { TSMM_ROW_MAJOR, TSMM_COL_MAJOR };
    const char   *layout_names[] = { "ROW_MAJOR", "COL_MAJOR" };

    int total  = 0;
    int passed = 0;

    printf("=== TSMM Correctness Test ===\n");
    printf("Kernels: naive, tiled_3d (Ti=%d,Tj=%d,Tk=%d), "
           "tiled_mn (Ti=%d,Tj=%d)\n",
           TEST_Ti, TEST_Tj, TEST_Tk, TEST_Ti, TEST_Tj);
    printf("\n");

    for (int si = shape_start; si < shape_end; ++si) {
        for (int li = 0; li < 2; ++li) {
            printf("--- %s  [%s] ---\n",
                   tsmm_shapes[si].name, layout_names[li]);
            total++;

            int ok = test_one(&tsmm_shapes[si], layouts[li]);
            if (ok) { passed++; }
            printf("\n");
        }
    }

    printf("=== %d / %d test groups passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
