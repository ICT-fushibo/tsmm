/**
 * src/tsmm_tiled_omp_s2.c — OpenMP Step 2: Ti×Tj tile + collapse(2)
 *
 * 仅 row-major。
 *
 * Step 1 → Step 2 唯一变化：pragma 加 collapse(2)。
 * 从 Ti×n strip 变为 Ti×Tj tile，并行度从 m/Ti 提升为 (m/Ti)×(n/Tj)。
 */

#include "tsmm.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* ==========================================================================
 *  3D tiling + collapse(2)
 * ========================================================================== */
void tsmm_tiled_omp_s2_rowmajor(int m, int n, int k,
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

    for(int pk =0;pk<k;pk+=Tk){
        int k_end = MIN(pk + Tk, k);
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(nt) schedule(static) collapse(2)
        #endif
        for (int ii = 0; ii < m; ii += Ti) {
            for (int jj = 0; jj < n; jj += Tj) {
                int i_end = MIN(ii + Ti, m);
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
 *  2D tiling (MN-only) + collapse(2)
 * ========================================================================== */
void tsmm_tiled_omp_s2_mn_rowmajor(int m, int n, int k,
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
    #pragma omp parallel for num_threads(nt) schedule(static) collapse(2)
#endif
    for (int ii = 0; ii < m; ii += Ti) {
        

        for (int jj = 0; jj < n; jj += Tj) {
            int i_end = MIN(ii + Ti, m);
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
 *  分发（col-major → 串行 fallback）
 * ========================================================================== */
void tsmm_tiled_omp_s2(tsmm_layout_t layout, int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_tiled_omp_s2_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk, num_threads);
    else
        tsmm_tiled_colmajor(m, n, k, A, B, C, Ti, Tj, Tk);
}

void tsmm_tiled_omp_s2_mn(tsmm_layout_t layout, int m, int n, int k,
                           const double *A, const double *B, double *C,
                           int Ti, int Tj, int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_tiled_omp_s2_mn_rowmajor(m, n, k, A, B, C, Ti, Tj, num_threads);
    else
        tsmm_tiled_mn_colmajor(m, n, k, A, B, C, Ti, Tj);
}
