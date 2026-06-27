/**
 * src/tsmm_tiled_omp_s3.c — OpenMP Step 3: 线程私有累加区
 *
 * 仅 row-major。
 *
 * ===========================================================================
 *  Step 2 → Step 3 变化
 * ===========================================================================
 *
 *  Step 2 问题：
 *    C[i*n+j] += ...  直接写全局 C。
 *    - 相邻 tile 边界可能落在同一 cache line → false sharing
 *    - C 分配在 NUMA node 0，远端线程写 C 走 UPI
 *    - 同一个 C tile 的 += 写回 k/Tk 次
 *
 *  Step 3 做法：
 *    每个线程分配一个 Ti×Tj 的私有 buffer (≈128 KB, fits L2)。
 *    在 local buffer 里完成全部 k 维累加，最后一次性写回 C。
 *
 *    收益：
 *    - local buffer 在线程所在 core 的 L1/L2 里，零 false sharing
 *    - 写回 C 只有 1 次（不是 k/Tk 次）
 *    - 大幅减少 NUMA 远端访存
 *
 *    代价：
 *    - 每个 tile 多一次 malloc/free 或复用 thread-local buffer
 *    - local buffer 需要 memset 清零（等同 C 的清零工作，不额外增加总量）
 *
 * ===========================================================================
 *  两种 buffer 管理方案
 * ===========================================================================
 *
 *  方案 A: 每个 tile malloc + free（简单，但 malloc 开销大）
 *
 *    double *buf = calloc(Ti * Tj, sizeof(double));
 *    ... micro-kernel 累加到 buf ...
 *    // 写回 buf → C
 *    free(buf);
 *
 *  方案 B: thread-local buffer 复用（推荐）
 *
 *    #pragma omp parallel
 *    {
 *        double *buf = calloc(Ti * Tj, sizeof(double));  // 每线程分配一次
 *
 *        #pragma omp for collapse(2)
 *        for (ii) for (jj) {
 *            memset(buf, 0, Ti * Tj * sizeof(double));   // 每 tile 清零
 *            ... micro-kernel 累加到 buf ...
 *            // 写回 buf → C
 *        }
 *
 *        free(buf);
 *    }
 *
 *  方案 B 只有一次分配，memset 开销可忽略（远小于 k 次 C += 的 cache miss）。
 *
 * ===========================================================================
 *  局部索引映射
 * ===========================================================================
 *
 *    全局 C[i][j]  →  C[i * n + j]
 *    局部 buf[i_loc][j_loc]  →  buf[i_loc * Tj + j_loc]
 *    其中 i_loc = i - ii,  j_loc = j - jj
 *
 *    微核内循环不变（仍从 ii到i_end, jj到j_end），只是 += 的目标变了。
 *
 * ===========================================================================
 *  调用约定
 * ===========================================================================
 *
 *  C 在调用前必须已初始化为零。
 *  内部 local buffer 由线程自己管理。
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
 *  Step 3: 3D tiling + collapse(2) + private buffer
 * ========================================================================== */
void tsmm_tiled_omp_s3_rowmajor(int m, int n, int k,
                                 const double *A, const double *B, double *C,
                                 int Ti, int Tj, int Tk, int num_threads)
{
#ifdef _OPENMP
    int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#else
    int nt = 1;
    (void)num_threads;
#endif

        #ifdef _OPENMP
            #pragma omp parallel num_threads(nt)
            {
                /* 每个线程分配一次 buf，跨所有 pk 轮复用 */
                double *buf = (double*)calloc((size_t)Ti * Tj, sizeof(double));
                if (!buf) {
                    fprintf(stderr, "omp_s3_3d: buf alloc failed\n");
                    exit(1);
                }

                /* 分发 tile：每个线程拿到一批 (ii,jj) 对 */
                #pragma omp for collapse(2) schedule(static)
                for (int ii = 0; ii < m; ii += Ti) {
                    for (int jj = 0; jj < n; jj += Tj) {
                        int i_end = MIN(ii + Ti, m);
                        int j_end = MIN(jj + Tj, n);

                        /* 清一次 buf，然后所有 pk 轮累加到同一个 buf 里 */
                        memset(buf, 0, (size_t)Ti * Tj * sizeof(double));

                        for (int pk = 0; pk < k; pk += Tk) {
                            int k_end = MIN(pk + Tk, k);

                            for (int p = pk; p < k_end; p++) {
                                for (int i = ii; i < i_end; i++) {
                                    double a_ip = A[p * m + i];
                                    for (int j = jj; j < j_end; j++) {
                                        buf[(i - ii) * Tj + (j - jj)] += a_ip * B[p * n + j];
                                    }
                                }
                            }
                        }

                        /* 所有 k 完成，写回 C 一次 */
                        for (int i = ii; i < i_end; i++) {
                            for (int j = jj; j < j_end; j++) {
                                C[i * n + j] += buf[(i - ii) * Tj + (j - jj)];
                            }
                        }
                    }
                }

                free(buf);
            }

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
 *  Step 3: 2D tiling (MN-only) + collapse(2) + private buffer
 * ========================================================================== */
void tsmm_tiled_omp_s3_mn_rowmajor(int m, int n, int k,
                                    const double *A, const double *B, double *C,
                                    int Ti, int Tj, int num_threads)
{
#ifdef _OPENMP
    int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#else
    int nt = 1;
    (void)num_threads;
#endif
    #ifdef _OPENMP
        #pragma omp parallel num_threads(nt)
        {
            double *buf = (double*)calloc((size_t)Ti * Tj, sizeof(double));
            if (!buf) {
                fprintf(stderr, "omp_s3_mn: buf alloc failed\n");
                exit(1);
            }
            #pragma omp for collapse(2) schedule(static)
            
            for (int ii = 0; ii < m; ii += Ti) {
                for (int jj = 0; jj < n; jj += Tj) {
                    int i_end = MIN(ii + Ti, m);
                    int j_end = MIN(jj + Tj, n);
                    memset(buf, 0, (size_t)Ti * Tj * sizeof(double));
                    /* ---- micro-kernel: p → i → j ---- */
                    for (int p = 0; p < k; p++) {
                        for (int i = ii; i < i_end; i++) {
                            double a_ip = A[p * m + i];       /* hoist A */
                            for (int j = jj; j < j_end; j++) {
                                buf[(i-ii)*Tj + (j-jj)] += a_ip * B[p*n+j];
                            }
                        }
                    }
                    /* ---- end micro-kernel ---- */
                    /* write back once after all k */
                    for (int i = ii; i < i_end; i++) {
                        for (int j = jj; j < j_end; j++) {
                            C[i * n + j] += buf[(i - ii) * Tj + (j - jj)];
                        }
                    }
                }
                
            }
            free(buf);
        
        }
    #else
        for (int ii = 0; ii < m; ii += Ti) {
            for (int jj = 0; jj < n; jj += Tj) {
                int i_end = MIN(ii + Ti, m);
                int j_end = MIN(jj + Tj, n);
                /* ---- micro-kernel: p → i → j ---- */
                for (int p = 0; p < k; p++) {
                    for (int i = ii; i < i_end; i++) {
                        double a_ip = A[p * m + i];       /* hoist A */
                        for (int j = jj; j < j_end; j++) {
                            C[i * n + j] += a_ip * B[p * n + j];
                        }
                    }
                }
                /* ---- end micro-kernel ---- */
            }
        }
    #endif
    
}

/* ==========================================================================
 *  分发（col-major → 串行 fallback）
 * ========================================================================== */
void tsmm_tiled_omp_s3(tsmm_layout_t layout, int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_tiled_omp_s3_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk, num_threads);
    else
        tsmm_tiled_colmajor(m, n, k, A, B, C, Ti, Tj, Tk);
}

void tsmm_tiled_omp_s3_mn(tsmm_layout_t layout, int m, int n, int k,
                           const double *A, const double *B, double *C,
                           int Ti, int Tj, int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_tiled_omp_s3_mn_rowmajor(m, n, k, A, B, C, Ti, Tj, num_threads);
    else
        tsmm_tiled_mn_colmajor(m, n, k, A, B, C, Ti, Tj);
}
