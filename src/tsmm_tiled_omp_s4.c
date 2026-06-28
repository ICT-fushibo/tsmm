/**
 * src/tsmm_tiled_omp_s4.c — OpenMP Step 4: k 维度并行 (3D tiling only)
 *
 * 仅 row-major。MN-only fallback 到 Step 3。
 *
 * ===========================================================================
 *  Step 3 → Step 4 变化
 * ===========================================================================
 *
 *  Step 3:
 *    collapse(2) 并行 (ii,jj) — 每个 C tile 由一个线程独享
 *    pk 串行 — 线程内部逐轮累加 private buffer
 *    写回 C 一次
 *
 *  Step 4:
 *    所有线程合作处理同一个 C tile 的不同 pk 范围。
 *    每个线程有自己的 local buffer，完成后规约到 shared buffer，再写回 C。
 *
 *    收益：
 *    - k 方向并行：k 很大时 (S6: 606841) pk loop 有 ~2372 次迭代
 *    - collapse(2) 并行度不够时 (S2/S6: 1 tile)，k 并行是唯一出路
 *
 *    代价：
 *    - 每个 tile 需要规约 (omp critical) + barrier
 *    - 每个 tile 需要 shared buffer 分配/释放
 *
 * ===========================================================================
 *  线程模型
 * ===========================================================================
 *
 *    只 fork 一次 (#pragma omp parallel 在最外层)
 *
 *    每个 (ii,jj) tile 内：
 *      1. #pragma omp single → 分配/清零 shared buffer
 *      2. 每个线程清零自己的 local buffer（memset）
 *      3. #pragma omp for schedule(static) → 分 pk 迭代给所有线程
 *         各线程把 micro-kernel 结果累加到自己的 local buffer
 *      4. #pragma omp critical → local → shared 规约
 *      5. #pragma omp barrier    → 等所有线程完成规约
 *      6. #pragma omp single     → shared → C 写回 + free(shared)
 *      7. #pragma omp barrier    → 等写回完成，进入下一个 tile
 *
 * ===========================================================================
 *  调用约定
 * ===========================================================================
 *
 *  C 在调用前必须已初始化为零。
 *  local/shared buffer 由线程自己管理。
 *  num_threads=0 → 使用 OMP_NUM_THREADS。
 */

#include "tsmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* ==========================================================================
 *  Step 4: 3D tiling + collapse(2) tile 分配 + pk 并行 + 规约
 *
 *  结构模板：
 *
 *  #ifdef _OPENMP
 *  #pragma omp parallel num_threads(nt)
 *  {
 *      int tid = omp_get_thread_num();
 *      double *local  = calloc(Ti * Tj, sizeof(double));  // 线程私有
 *      double *shared = NULL;                              // team 共享, 每个 tile 重新分配
 *      if (!local) { fprintf(stderr, "..."); exit(1); }
 *
 *      // 手动 tile 遍历（每个 tile 所有线程合作处理）
 *      int nt_i = (m + Ti - 1) / Ti;
 *      int nt_j = (n + Tj - 1) / Tj;
 *      int total_tiles = nt_i * nt_j;
 *
 *      for (int tile = 0; tile < total_tiles; tile++) {
 *          int ii = (tile / nt_j) * Ti;
 *          int jj = (tile % nt_j) * Tj;
 *          int i_end = MIN(ii + Ti, m);
 *          int j_end = MIN(jj + Tj, n);
 *
 *          // ① 分配 shared buffer（一个线程做）
 *          #pragma omp single
 *          {
 *              shared = calloc(Ti * Tj, sizeof(double));
 *              // shared 在 omp single 内分配，所有线程可见（shared 是 shared 变量）
 *          }
 *          // implicit barrier after single
 *
 *          // ② 清零 local
 *          memset(local, 0, (size_t)Ti * Tj * sizeof(double));
 *
 *          // ③ 所有线程并行 pk
 *          #pragma omp for schedule(static) nowait
 *          for (int pk = 0; pk < k; pk += Tk) {
 *              int k_end = MIN(pk + Tk, k);
 *              for (int p = pk; p < k_end; p++) {
 *                  for (int i = ii; i < i_end; i++) {
 *                      double a_ip = A[p * m + i];
 *                      for (int j = jj; j < j_end; j++) {
 *                          local[(i-ii)*Tj + (j-jj)] += a_ip * B[p*n+j];
 *                      }
 *                  }
 *              }
 *          }
 *          // nowait: 每个线程完成自己的 pk 后可以继续
 *          // 但规约前必须等所有线程完成 → 需要显式 barrier
 *          #pragma omp barrier
 *
 *          // ④ 规约: local → shared
 *          #pragma omp critical
 *          {
 *              int n_elems = (i_end - ii) * Tj;
 *              // 只累加这个 tile 实际使用的元素
 *              for (int i = ii; i < i_end; i++)
 *                  for (int j = jj; j < j_end; j++)
 *                      shared[(i-ii)*Tj + (j-jj)] += local[(i-ii)*Tj + (j-jj)];
 *          }
 *          #pragma omp barrier
 *
 *          // ⑤ 写回 shared → C（一个线程做）
 *          #pragma omp single
 *          {
 *              for (int i = ii; i < i_end; i++)
 *                  for (int j = jj; j < j_end; j++)
 *                      C[i*n+j] += shared[(i-ii)*Tj + (j-jj)];
 *              free(shared);
 *              shared = NULL;
 *          }
 *          // implicit barrier
 *      }
 *
 *      free(local);
 *  }
 *  #else
 *  // 串行 fallback: tsmm_tiled_rowmajor
 *  #endif
 *
 *  TODO: 按照上述模板实现。
 *        注意 shared 变量需要在 tile 循环头部声明，
 *        omp single 内分配的值对所有线程可见（shared 是 implicitly shared）。
 *
 *  WARNING: 如果 total_tiles 很大（如 S1: ~39000 tiles），sequential tile loop
 *           会有大量 single/critical/barrier 开销。
 *           后续可优化：当 total_tiles 很大时 fallback 到 Step 3 的 collapse(2) 策略。
 * ========================================================================== */
void tsmm_tiled_omp_s4_rowmajor(int m, int n, int k,
                                 const double *A, const double *B, double *C,
                                 int Ti, int Tj, int Tk, int num_threads)
{

#ifdef _OPENMP
    int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#else
    int nt = 1;
    (void)num_threads;
#endif

    /* Performance guard: S4 per-tile overhead too high for many tiles.
     * Fall back to S3 when tiles > 1000 (e.g. S1: 38750, S8: 4404). */
    int nt_i = (m + Ti - 1) / Ti;
    int nt_j = (n + Tj - 1) / Tj;
    int total_tiles = nt_i * nt_j;

#ifdef _OPENMP
    int pk_iters = (k + Tk - 1) / Tk;  /* number of pk loop iterations */

    /* S4 only beneficial when: tiles are few AND k has enough iterations to
     * distribute across threads. Otherwise the per-tile overhead (critical,
     * barrier) and idle threads dominate. */
    if (total_tiles > 1000 || pk_iters < nt / 4) {
        tsmm_tiled_omp_s3_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk, num_threads);
        return;
    }

    /* shared: allocated ONCE, reused per tile via memset (saves malloc/free per tile) */
    double *shared = malloc((size_t)Ti * Tj * sizeof(double));
    if (!shared) { fprintf(stderr, "s4: shared alloc failed\n"); exit(1); }

    #pragma omp parallel num_threads(nt)
    {
        double *local = calloc(Ti * Tj, sizeof(double));  /* thread-private */
        if (!local) { fprintf(stderr, "s4: local alloc failed\n"); exit(1); }

        for (int tile = 0; tile < total_tiles; tile++) {
            int ii = (tile / nt_j) * Ti;
            int jj = (tile % nt_j) * Tj;
            int i_end = MIN(ii + Ti, m);
            int j_end = MIN(jj + Tj, n);

            /* ① zero shared (one thread; implicit barrier after single) */
            #pragma omp single
            memset(shared, 0, (size_t)Ti * Tj * sizeof(double));

            /* ② zero local */
            memset(local, 0, (size_t)Ti * Tj * sizeof(double));

            /* ③ parallel pk */
            #pragma omp for schedule(static)
            for (int pk = 0; pk < k; pk += Tk) {
                int k_end = MIN(pk + Tk, k);
                for (int p = pk; p < k_end; p++) {
                    for (int i = ii; i < i_end; i++) {
                        double a_ip = A[p * m + i];
                        for (int j = jj; j < j_end; j++) {
                            local[(i-ii)*Tj + (j-jj)] += a_ip * B[p*n+j];
                        }
                    }
                }
            }
            /* implicit barrier after omp for */

            /* ④ reduction: local → shared (serialized, correctness over speed) */
            #pragma omp critical
            {
                for (int i = ii; i < i_end; i++)
                    for (int j = jj; j < j_end; j++)
                        shared[(i-ii)*Tj + (j-jj)] += local[(i-ii)*Tj + (j-jj)];
            }
            #pragma omp barrier

            /* ⑤ writeback shared → C (one thread; implicit barrier after single) */
            #pragma omp single
            {
                for (int i = ii; i < i_end; i++)
                    for (int j = jj; j < j_end; j++)
                        C[i*n+j] += shared[(i-ii)*Tj + (j-jj)];
            }
        }
        free(local);
    }
    free(shared);
#else
    /* 串行 fallback */
    for (int ii = 0; ii < m; ii += Ti) {
        for (int jj = 0; jj < n; jj += Tj) {
            int i_end = MIN(ii + Ti, m);
            int j_end = MIN(jj + Tj, n);
            for (int pk = 0; pk < k; pk += Tk) {
                int k_end = MIN(pk + Tk, k);
                for (int p = pk; p < k_end; p++) {
                    for (int i = ii; i < i_end; i++) {
                        double a_ip = A[p * m + i];
                        for (int j = jj; j < j_end; j++) {
                            C[i * n + j] += a_ip * B[p * n + j];
                        }
                    }
                }
            }
        }
    }
#endif

}

/* ==========================================================================
 *  通用分发（col-major / MN → 串行 fallback）
 * ========================================================================== */

void tsmm_tiled_omp_s4(tsmm_layout_t layout, int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_tiled_omp_s4_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk, num_threads);
    else
        tsmm_tiled_colmajor(m, n, k, A, B, C, Ti, Tj, Tk);
}

/* ==========================================================================
 *  S4 MN-only: fallback 到 S3
 * ========================================================================== */

void tsmm_tiled_omp_s4_mn_rowmajor(int m, int n, int k,
                                    const double *A, const double *B, double *C,
                                    int Ti, int Tj, int num_threads)
{
    tsmm_tiled_omp_s3_mn_rowmajor(m, n, k, A, B, C, Ti, Tj, num_threads);
}

void tsmm_tiled_omp_s4_mn(tsmm_layout_t layout, int m, int n, int k,
                           const double *A, const double *B, double *C,
                           int Ti, int Tj, int num_threads)
{
    tsmm_tiled_omp_s3_mn(layout, m, n, k, A, B, C, Ti, Tj, num_threads);
}
