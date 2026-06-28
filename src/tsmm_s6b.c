/**
 * src/tsmm_s6b.c — Stage 6b: S2 (8×16×16000) 全寄存器驻留
 *
 * C = 8×16 = 128 doubles = 1 KB → 可全部放进 16 个 ZMM 寄存器。
 * 不需要外层 cache tiling (Ti,Tj,Tk)；不需要 private buffer。
 * k=16000 → k 方向并行 (omp for on p), 线程私有 C 寄存器累加,
 * 最后 reduction 到共享 C。
 *
 * 仅 row-major (S2 太小, col-major 无关紧要)。
 */

#include "tsmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __AVX512F__
#include <immintrin.h>
#endif

void tsmm_s6b_rowmajor(int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int num_threads)
{
    (void)m; (void)n;  /* hardcoded 8×16 */
    int nt = 1;
#ifdef _OPENMP
    nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#endif

#ifdef __AVX512F__
    /* shared C buffer for reduction (allocated once) */
    double *shared = (double*)calloc(8 * 16, sizeof(double));
    if (!shared) { fprintf(stderr, "s6b: shared alloc failed\n"); exit(1); }

    #pragma omp parallel num_threads(nt)
    {
        /* ---- per-thread private C in 16 ZMM registers ---- */
        __m512d c00 = _mm512_setzero_pd();
        __m512d c01 = _mm512_setzero_pd();
        __m512d c10 = _mm512_setzero_pd();
        __m512d c11 = _mm512_setzero_pd();
        __m512d c20 = _mm512_setzero_pd();
        __m512d c21 = _mm512_setzero_pd();
        __m512d c30 = _mm512_setzero_pd();
        __m512d c31 = _mm512_setzero_pd();
        __m512d c40 = _mm512_setzero_pd();
        __m512d c41 = _mm512_setzero_pd();
        __m512d c50 = _mm512_setzero_pd();
        __m512d c51 = _mm512_setzero_pd();
        __m512d c60 = _mm512_setzero_pd();
        __m512d c61 = _mm512_setzero_pd();
        __m512d c70 = _mm512_setzero_pd();
        __m512d c71 = _mm512_setzero_pd();

        /* ---- k-parallel: each thread processes a chunk of k ---- */
        #pragma omp for schedule(static)
        for (int p = 0; p < k; p++) {
            __m512d b0 = _mm512_loadu_pd(&B[p * n + 0]);
            __m512d b1 = _mm512_loadu_pd(&B[p * n + 8]);

            __m512d a0 = _mm512_set1_pd(A[p * m + 0]);
            c00 = _mm512_fmadd_pd(a0, b0, c00);
            c01 = _mm512_fmadd_pd(a0, b1, c01);

            __m512d a1 = _mm512_set1_pd(A[p * m + 1]);
            c10 = _mm512_fmadd_pd(a1, b0, c10);
            c11 = _mm512_fmadd_pd(a1, b1, c11);

            __m512d a2 = _mm512_set1_pd(A[p * m + 2]);
            c20 = _mm512_fmadd_pd(a2, b0, c20);
            c21 = _mm512_fmadd_pd(a2, b1, c21);

            __m512d a3 = _mm512_set1_pd(A[p * m + 3]);
            c30 = _mm512_fmadd_pd(a3, b0, c30);
            c31 = _mm512_fmadd_pd(a3, b1, c31);

            __m512d a4 = _mm512_set1_pd(A[p * m + 4]);
            c40 = _mm512_fmadd_pd(a4, b0, c40);
            c41 = _mm512_fmadd_pd(a4, b1, c41);

            __m512d a5 = _mm512_set1_pd(A[p * m + 5]);
            c50 = _mm512_fmadd_pd(a5, b0, c50);
            c51 = _mm512_fmadd_pd(a5, b1, c51);

            __m512d a6 = _mm512_set1_pd(A[p * m + 6]);
            c60 = _mm512_fmadd_pd(a6, b0, c60);
            c61 = _mm512_fmadd_pd(a6, b1, c61);

            __m512d a7 = _mm512_set1_pd(A[p * m + 7]);
            c70 = _mm512_fmadd_pd(a7, b0, c70);
            c71 = _mm512_fmadd_pd(a7, b1, c71);
        }
        /* implicit barrier after omp for */

        /* ---- reduction: per-thread C registers → shared ---- */
        #pragma omp critical
        {
            /* row 0 */
            _mm512_storeu_pd(&shared[0],  _mm512_add_pd(_mm512_loadu_pd(&shared[0]),  c00));
            _mm512_storeu_pd(&shared[8],  _mm512_add_pd(_mm512_loadu_pd(&shared[8]),  c01));
            /* row 1 */
            _mm512_storeu_pd(&shared[16], _mm512_add_pd(_mm512_loadu_pd(&shared[16]), c10));
            _mm512_storeu_pd(&shared[24], _mm512_add_pd(_mm512_loadu_pd(&shared[24]), c11));
            /* row 2 */
            _mm512_storeu_pd(&shared[32], _mm512_add_pd(_mm512_loadu_pd(&shared[32]), c20));
            _mm512_storeu_pd(&shared[40], _mm512_add_pd(_mm512_loadu_pd(&shared[40]), c21));
            /* row 3 */
            _mm512_storeu_pd(&shared[48], _mm512_add_pd(_mm512_loadu_pd(&shared[48]), c30));
            _mm512_storeu_pd(&shared[56], _mm512_add_pd(_mm512_loadu_pd(&shared[56]), c31));
            /* row 4 */
            _mm512_storeu_pd(&shared[64], _mm512_add_pd(_mm512_loadu_pd(&shared[64]), c40));
            _mm512_storeu_pd(&shared[72], _mm512_add_pd(_mm512_loadu_pd(&shared[72]), c41));
            /* row 5 */
            _mm512_storeu_pd(&shared[80], _mm512_add_pd(_mm512_loadu_pd(&shared[80]), c50));
            _mm512_storeu_pd(&shared[88], _mm512_add_pd(_mm512_loadu_pd(&shared[88]), c51));
            /* row 6 */
            _mm512_storeu_pd(&shared[96], _mm512_add_pd(_mm512_loadu_pd(&shared[96]), c60));
            _mm512_storeu_pd(&shared[104],_mm512_add_pd(_mm512_loadu_pd(&shared[104]),c61));
            /* row 7 */
            _mm512_storeu_pd(&shared[112],_mm512_add_pd(_mm512_loadu_pd(&shared[112]),c70));
            _mm512_storeu_pd(&shared[120],_mm512_add_pd(_mm512_loadu_pd(&shared[120]),c71));
        }
    }
    /* implicit barrier after parallel region — all reductions done */

    /* ---- writeback: shared → C (serial, outside parallel) ---- */
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 16; j++)
            C[i * n + j] += shared[i * 16 + j];

    free(shared);
#else
    /* No AVX-512: scalar fallback (p-i-j, full k, full C in L1) */
    (void)num_threads;
    for (int p = 0; p < k; p++) {
        for (int i = 0; i < m; i++) {
            double a_ip = A[p * m + i];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a_ip * B[p * n + j];
        }
    }
#endif
}

/* ==========================================================================
 *  通用分发
 * ========================================================================== */
void tsmm_s6b(tsmm_layout_t layout, int m, int n, int k,
              const double *A, const double *B, double *C,
              int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_s6b_rowmajor(m, n, k, A, B, C, num_threads);
    else
        tsmm_tiled_colmajor(m, n, k, A, B, C, 8, 16, 256);
}
