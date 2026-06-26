/**
 * tests/test_correctness.c — TSMM 正确性验证框架。
 *
 * TODO: 填充测试逻辑。
 *
 * 框架已提供：
 *   - 命令行解析（可选指定单个 shape）
 *   - 遍历所有 shape × layout 组合
 *   - 统一的 PASS/FAIL/SKIP 输出格式
 *
 * 你需要做：
 *   1. 在 test_one() 中分配矩阵、初始化、调用你的 kernel 和参考实现、比较结果。
 *   2. 选择合适的 tolerance（建议 1e-10 * k 量级，考虑浮点累加误差）。
 *   3. 对于大规模 shape（S1/S8），参考实现可能太慢——可以考虑跳过或做随机采样校验。
 *
 * 用法：
 *   ./build/test_correctness              # 测试全部 8 组 shape
 *   ./build/test_correctness 3            # 仅测试 shape 3
 */

#include "tsmm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- 可调参数 ---------- */
#define TOLERANCE  1e-10   /* 相对误差容忍度 */

/* ---------- 每个 (shape, layout) 对的测试 ---------- */
static int test_one(const tsmm_shape_t *shape, tsmm_layout_t layout)
{
    int m = shape->m, n = shape->n, k = shape->k;

    /* TODO:
     *   1. 分配 A(k×m), B(k×n), C_my(m×n), C_ref(m×n)
     *      - 分配失败时打印 SKIP 并返回 1（不是失败，是无法测试）
     *   2. 用 tsmm_init_random 初始化 A, B
     *   3. 清零 C_my，调用你的 tsmm_naive(layout, ...)
     *   4. 调用 tsmm_reference(layout, ...) 得到 C_ref
     *      - 如果 shape 太大导致 reference 太慢，可以跳过并返回 1
     *   5. 用 tsmm_max_abs_diff 和 tsmm_verify 比较 C_my 和 C_ref
     *   6. 打印 max absolute difference
     *   7. 返回 1 (pass) 或 0 (fail)
     *   8. 释放所有分配的内存
     */

    (void)m; (void)n; (void)k; (void)layout;
    // alloc A B C
    double*A=tsmm_alloc_matrix(k,m);
    if(A==NULL){printf("shape (m=%d,n=%d,k=%d).Matrix A alloc error.\n",m,n,k);return 1;}
    double*B=tsmm_alloc_matrix(k,n);
    if(B==NULL)
    {
        printf("shape (m=%d,n=%d,k=%d).Matrix B alloc error.\n",m,n,k); 
        tsmm_free_matrix(A);
        return 1;
    }
    double*C_my=tsmm_alloc_matrix(m,n);
    if(C_my==NULL)
    {
        printf("shape (m=%d,n=%d,k=%d).Matrix C_my alloc error.\n",m,n,k);
        tsmm_free_matrix(A);
        tsmm_free_matrix(B);
        return 1;
    }
    double*C_ref=tsmm_alloc_matrix(m,n);
    if(C_ref==NULL)
    {
        printf("shape (m=%d,n=%d,k=%d).Matrix C_ref alloc error.\n",m,n,k);
        tsmm_free_matrix(A);
        tsmm_free_matrix(B);
        tsmm_free_matrix(C_my);
        return 1;
    }

    tsmm_init_random(A,k,m);
    tsmm_init_random(B,k,n);
    tsmm_init_zero(C_my,m,n);
    tsmm_init_zero(C_ref,m,n);

    tsmm_naive(layout,m,n,k,A,B,C_my);
    tsmm_reference(layout,m,n,k,A,B,C_ref);

    double max_diff = tsmm_max_abs_diff(C_my,C_ref,m,n);
    if(tsmm_verify(C_my,C_ref,m,n,1e-10*k))
    {
        //pass
        printf("shape(m=%d,n=%d,k=%d).max_diff=%f.\n ****PASS**** \n \n",m,n,k,max_diff);
        tsmm_free_matrix(A);
        tsmm_free_matrix(B);
        tsmm_free_matrix(C_my);
        tsmm_free_matrix(C_ref);
        return 1;

    }
    else
    {
        //fail
        printf("shape(m=%d,n=%d,k=%d).max_diff=%f.\n ****FAIL**** \n \n",m,n,k,max_diff);
        tsmm_free_matrix(A);
        tsmm_free_matrix(B);
        tsmm_free_matrix(C_my);
        tsmm_free_matrix(C_ref);
        return 0;
    }



    printf("Something wrong in Fuc (test_one) \n");
    return 0; 
}

/* ========================================================================== */

int main(int argc, char **argv)
{
    int shape_start = 0;
    int shape_end   = TSMM_NUM_SHAPES;

    if (argc > 1) {
        int idx = atoi(argv[1]);
        if (idx >= 1 && idx <= TSMM_NUM_SHAPES) {
            shape_start = idx - 1;
            shape_end   = idx;
        } else {
            fprintf(stderr, "Usage: %s [shape_index 1-%d]\n",
                    argv[0], TSMM_NUM_SHAPES);
            return 1;
        }
    }

    tsmm_layout_t layouts[]      = { TSMM_ROW_MAJOR, TSMM_COL_MAJOR };
    const char   *layout_names[] = { "ROW_MAJOR", "COL_MAJOR" };
    int n_layouts = 2;

    int total  = 0;
    int passed = 0;

    printf("=== TSMM Correctness Test ===\n");
    printf("Tolerance: %.2e (relative)\n\n", TOLERANCE);

    for (int si = shape_start; si < shape_end; ++si) {
        for (int li = 0; li < n_layouts; ++li) {
            printf("--- %s  [%s] ---\n",
                   tsmm_shapes[si].name, layout_names[li]);
            total++;

            int ok = test_one(&tsmm_shapes[si], layouts[li]);
            if (ok) {
                printf("  PASS\n");
                passed++;
            } else {
                printf("  FAIL\n");
            }
            printf("\n");
        }
    }

    printf("=== %d / %d tests passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
