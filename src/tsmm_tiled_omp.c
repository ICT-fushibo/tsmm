/**
 * src/tsmm_tiled_omp.c — Multi-threaded TSMM (OpenMP on tiled loops).
 *
 * 仅 row-major。列主序 OMP 不做。
 *
 * ===========================================================================
 *  优化路线图 (row-major only)
 * ===========================================================================
 *
 *  Step 1 (当前): Ti×n strip 并行
 *    并行 ii loop，每个线程处理一条 Ti×n 的 C。
 *    线程内部：jj 和 pk 串行，micro-kernel 和串行版一致。
 *    优点：实现最简单，无数据竞争。
 *    缺点：strip 太大 (Ti×n) 可能超出 L2；m/Ti 小时并行度不足。
 *
 *  Step 2 (后续): Ti×Tj tile + collapse(2)
 *    并行 ii+jj (collapse)，每个线程处理 Ti×Tj 的 C tile (fit L2)。
 *    并行度 = (m/Ti)×(n/Tj)，远高于 Step 1。
 *
 *  Step 3 (后续): 线程私有累加区
 *    每线程 malloc 一个 Ti×Tj 的 local buffer。
 *    micro-kernel 累加到 local buffer，完成后写回 C。
 *    减少 false sharing 和 NUMA remote write。
 *
 *  Step 4 (后续): Ti×Tj×Tk 三维并行
 *    collapse(3) 并行 ii+jj+pk，或 pk loop 做 reduction。
 *    适用于 k 极大的 shape (S6: k=606841)。
 *
 * ===========================================================================
 *  调用约定
 * ===========================================================================
 *
 *  C 在调用前必须已初始化为零。
 *  num_threads=0 → 使用 OMP_NUM_THREADS 环境变量。
 *  无 _OPENMP 时 fallback 到串行 tsmm_tiled_rowmajor。
 *
 * ===========================================================================
 *  NUMA 注意事项
 * ===========================================================================
 *
 *  C 单线程 memset → 所有页在 node 0。
 *  多线程跨 NUMA 写 C 时通过 numactl --interleave 缓解。
 *  脚本层 run_benchmark_omp.sh 已处理 NUMA 绑定。
 * ===========================================================================
 */

#include "tsmm.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* ==========================================================================
 *  Step 1: Row-major 3D tiling, Ti×n strip 并行
 *
 *  并行 ii loop。每个线程处理一条 Ti×n 的 strip。
 *
 *  Tile loop 顺序：ii (并行) → pk → jj → micro-kernel
 *
 *    #pragma omp parallel for
 *    for (ii = 0; ii < m; ii += Ti)          // ← 并行
 *      for (pk = 0; pk < k; pk += Tk)         // 串行
 *        for (jj = 0; jj < n; jj += Tj)       // 串行
 *          // micro-kernel: p → i → j
 *
 *  ii 提到最外层，pk 和 jj 在内部串行。
 *  这样线程 n 只负责 ii∈[n*Ti, (n+1)*Ti)，写 C 区间不重叠 → 零数据竞争。
 *
 *  注意：
 *    - m/Ti 小时 (S2: m=8→1 tile) 只有 1 个线程工作
 *    - Step 2 的 collapse(2) 会解决这个问题
 * ========================================================================== */
void tsmm_tiled_omp_rowmajor(int m, int n, int k,
                              const double *A, const double *B, double *C,
                              int Ti, int Tj, int Tk, int num_threads)
{
#ifdef _OPENMP
    int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#else
    int nt = 1;
    (void)num_threads;
#endif

    /* ---- Step 1: 并行 ii，每个线程一个 Ti×n strip ---- */
#ifdef _OPENMP
    #pragma omp parallel for num_threads(nt) schedule(static)
#endif
    for (int ii = 0; ii < m; ii += Ti) {
        int i_end = MIN(ii + Ti, m);

        /* 线程内部：串行 pk 和 jj，和串行 tiled 完全一致 */
        for (int pk = 0; pk < k; pk += Tk) {
            int k_end = MIN(pk + Tk, k);
            for (int jj = 0; jj < n; jj += Tj) {
                int j_end = MIN(jj + Tj, n);

                /* ---- micro-kernel: p → i → j ---- */
                for (int p = pk; p < k_end; p++) {
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
    }
}

/* ==========================================================================
 *  Step 1: Row-major 2D tiling (MN-only), Ti×n strip 并行
 *
 *  同上，但 k 不分块（无 pk loop），p 跑满 0..k-1。
 *
 *  Tile loop: ii (并行) → jj (串行) → p → i → j
 * ========================================================================== */
void tsmm_tiled_omp_mn_rowmajor(int m, int n, int k,
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
    #pragma omp parallel for num_threads(nt) schedule(static)
#endif
    for (int ii = 0; ii < m; ii += Ti) {
        int i_end = MIN(ii + Ti, m);

        for (int jj = 0; jj < n; jj += Tj) {
            int j_end = MIN(jj + Tj, n);

            /* k 全量遍历，无 pk loop */
            for (int p = 0; p < k; p++) {
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

/* ==========================================================================
 *  Column-major — 不做 OMP 优化，fallback 到串行 tiled。
 * ========================================================================== */
void tsmm_tiled_omp_colmajor(int m, int n, int k,
                              const double *A, const double *B, double *C,
                              int Ti, int Tj, int Tk, int num_threads)
{
    (void)num_threads;
    tsmm_tiled_colmajor(m, n, k, A, B, C, Ti, Tj, Tk);
}

void tsmm_tiled_omp_mn_colmajor(int m, int n, int k,
                                 const double *A, const double *B, double *C,
                                 int Ti, int Tj, int num_threads)
{
    (void)num_threads;
    tsmm_tiled_mn_colmajor(m, n, k, A, B, C, Ti, Tj);
}

/* ==========================================================================
 *  通用分发
 * ========================================================================== */

void tsmm_tiled_omp(tsmm_layout_t layout, int m, int n, int k,
                    const double *A, const double *B, double *C,
                    int Ti, int Tj, int Tk, int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_tiled_omp_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk, num_threads);
    else
        tsmm_tiled_omp_colmajor(m, n, k, A, B, C, Ti, Tj, Tk, num_threads);
}

void tsmm_tiled_omp_mn(tsmm_layout_t layout, int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_tiled_omp_mn_rowmajor(m, n, k, A, B, C, Ti, Tj, num_threads);
    else
        tsmm_tiled_omp_mn_colmajor(m, n, k, A, B, C, Ti, Tj, num_threads);
}
