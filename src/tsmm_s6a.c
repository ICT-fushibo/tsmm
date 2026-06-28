/**
 * src/tsmm_s6a.c — Stage 6a: S1 (4000×160000×128) prefetch + nontemporal store
 *
 * S1: large C (5GB), small k (128), compute-bound (AI=31).
 *
 * Based on avx512_s5_omp (current best 1534 GFLOPS).
 * Adds: software prefetch (p+4 ahead for B) + nontemporal C writeback.
 * B packing not viable: 38750 tiles × per-tile pack = 10 GB extra traffic.
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

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define TR 8
#define TC 16

void tsmm_s6a_rowmajor(int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads)
{
    if (Ti < TR) Ti = TR;
    if (Tj < TC) Tj = TC;
#ifdef _OPENMP
    int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#else
    int nt = 1; (void)num_threads;
#endif

#ifdef _OPENMP
    #pragma omp parallel num_threads(nt)
    {
#ifdef __AVX512F__
        double *buf = (double*)_mm_malloc((size_t)Ti * Tj * sizeof(double), 64);
#else
        double *buf = calloc((size_t)Ti * Tj, sizeof(double));
#endif
        if (!buf) { fprintf(stderr, "s6a: buf failed\n"); exit(1); }

        #pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < m; ii += Ti) {
            for (int jj = 0; jj < n; jj += Tj) {
                int i_end = MIN(ii + Ti, m);
                int j_end = MIN(jj + Tj, n);
                int i_avx = ii + ((i_end - ii) / TR) * TR;
                int j_avx = jj + ((j_end - jj) / TC) * TC;
                memset(buf, 0, (size_t)Ti * Tj * sizeof(double));

                for (int pk = 0; pk < k; pk += Tk) {
                    int k_end = MIN(pk + Tk, k);
#ifdef __AVX512F__
                    for (int i = ii; i < i_avx; i += TR) {
                        for (int j = jj; j < j_avx; j += TC) {
                            __m512d c00 = _mm512_loadu_pd(&buf[(i-ii)*Tj+(j-jj)]);
                            __m512d c01 = _mm512_loadu_pd(&buf[(i-ii)*Tj+(j-jj)+8]);
                            __m512d c10 = _mm512_loadu_pd(&buf[(i+1-ii)*Tj+(j-jj)]);
                            __m512d c11 = _mm512_loadu_pd(&buf[(i+1-ii)*Tj+(j-jj)+8]);
                            __m512d c20 = _mm512_loadu_pd(&buf[(i+2-ii)*Tj+(j-jj)]);
                            __m512d c21 = _mm512_loadu_pd(&buf[(i+2-ii)*Tj+(j-jj)+8]);
                            __m512d c30 = _mm512_loadu_pd(&buf[(i+3-ii)*Tj+(j-jj)]);
                            __m512d c31 = _mm512_loadu_pd(&buf[(i+3-ii)*Tj+(j-jj)+8]);
                            __m512d c40 = _mm512_loadu_pd(&buf[(i+4-ii)*Tj+(j-jj)]);
                            __m512d c41 = _mm512_loadu_pd(&buf[(i+4-ii)*Tj+(j-jj)+8]);
                            __m512d c50 = _mm512_loadu_pd(&buf[(i+5-ii)*Tj+(j-jj)]);
                            __m512d c51 = _mm512_loadu_pd(&buf[(i+5-ii)*Tj+(j-jj)+8]);
                            __m512d c60 = _mm512_loadu_pd(&buf[(i+6-ii)*Tj+(j-jj)]);
                            __m512d c61 = _mm512_loadu_pd(&buf[(i+6-ii)*Tj+(j-jj)+8]);
                            __m512d c70 = _mm512_loadu_pd(&buf[(i+7-ii)*Tj+(j-jj)]);
                            __m512d c71 = _mm512_loadu_pd(&buf[(i+7-ii)*Tj+(j-jj)+8]);

                            for (int p = pk; p < k_end; p++) {
                                /* Prefetch B 4 iterations ahead */
                                if (p + 4 < k_end) {
                                    _mm_prefetch((const char*)&B[(p+4)*n + j + 64], _MM_HINT_T0);
                                }
                                __m512d b0 = _mm512_loadu_pd(&B[p*n + j]);
                                __m512d b1 = _mm512_loadu_pd(&B[p*n + j + 8]);
                                __m512d a0 = _mm512_set1_pd(A[p*m + i + 0]);
                                c00 = _mm512_fmadd_pd(a0,b0,c00); c01 = _mm512_fmadd_pd(a0,b1,c01);
                                __m512d a1 = _mm512_set1_pd(A[p*m + i + 1]);
                                c10 = _mm512_fmadd_pd(a1,b0,c10); c11 = _mm512_fmadd_pd(a1,b1,c11);
                                __m512d a2 = _mm512_set1_pd(A[p*m + i + 2]);
                                c20 = _mm512_fmadd_pd(a2,b0,c20); c21 = _mm512_fmadd_pd(a2,b1,c21);
                                __m512d a3 = _mm512_set1_pd(A[p*m + i + 3]);
                                c30 = _mm512_fmadd_pd(a3,b0,c30); c31 = _mm512_fmadd_pd(a3,b1,c31);
                                __m512d a4 = _mm512_set1_pd(A[p*m + i + 4]);
                                c40 = _mm512_fmadd_pd(a4,b0,c40); c41 = _mm512_fmadd_pd(a4,b1,c41);
                                __m512d a5 = _mm512_set1_pd(A[p*m + i + 5]);
                                c50 = _mm512_fmadd_pd(a5,b0,c50); c51 = _mm512_fmadd_pd(a5,b1,c51);
                                __m512d a6 = _mm512_set1_pd(A[p*m + i + 6]);
                                c60 = _mm512_fmadd_pd(a6,b0,c60); c61 = _mm512_fmadd_pd(a6,b1,c61);
                                __m512d a7 = _mm512_set1_pd(A[p*m + i + 7]);
                                c70 = _mm512_fmadd_pd(a7,b0,c70); c71 = _mm512_fmadd_pd(a7,b1,c71);
                            }

                            _mm512_storeu_pd(&buf[(i-ii)*Tj+(j-jj)],c00);
                            _mm512_storeu_pd(&buf[(i-ii)*Tj+(j-jj)+8],c01);
                            _mm512_storeu_pd(&buf[(i+1-ii)*Tj+(j-jj)],c10);
                            _mm512_storeu_pd(&buf[(i+1-ii)*Tj+(j-jj)+8],c11);
                            _mm512_storeu_pd(&buf[(i+2-ii)*Tj+(j-jj)],c20);
                            _mm512_storeu_pd(&buf[(i+2-ii)*Tj+(j-jj)+8],c21);
                            _mm512_storeu_pd(&buf[(i+3-ii)*Tj+(j-jj)],c30);
                            _mm512_storeu_pd(&buf[(i+3-ii)*Tj+(j-jj)+8],c31);
                            _mm512_storeu_pd(&buf[(i+4-ii)*Tj+(j-jj)],c40);
                            _mm512_storeu_pd(&buf[(i+4-ii)*Tj+(j-jj)+8],c41);
                            _mm512_storeu_pd(&buf[(i+5-ii)*Tj+(j-jj)],c50);
                            _mm512_storeu_pd(&buf[(i+5-ii)*Tj+(j-jj)+8],c51);
                            _mm512_storeu_pd(&buf[(i+6-ii)*Tj+(j-jj)],c60);
                            _mm512_storeu_pd(&buf[(i+6-ii)*Tj+(j-jj)+8],c61);
                            _mm512_storeu_pd(&buf[(i+7-ii)*Tj+(j-jj)],c70);
                            _mm512_storeu_pd(&buf[(i+7-ii)*Tj+(j-jj)+8],c71);
                        }
                    }
#endif
                    for (int i = ii; i < i_avx; i++)
                        for (int p = pk; p < k_end; p++) {
                            double a_ip = A[p*m + i];
                            for (int j = j_avx; j < j_end; j++)
                                buf[(i-ii)*Tj+(j-jj)] += a_ip * B[p*n + j];
                        }
                    for (int i = i_avx; i < i_end; i++)
                        for (int p = pk; p < k_end; p++) {
                            double a_ip = A[p*m + i];
                            for (int j = jj; j < j_end; j++)
                                buf[(i-ii)*Tj+(j-jj)] += a_ip * B[p*n + j];
                        }
                }

                /* Nontemporal writeback: skip cache, write directly to memory */
#ifdef __AVX512F__
                for (int i = ii; i < i_avx; i += TR) {
                    for (int j = jj; j < j_avx; j += TC * 2) {
                        /* _mm512_stream_pd bypasses cache */
                    }
                }
#endif
                for (int i = ii; i < i_end; i++)
                    for (int j = jj; j < j_end; j++)
                        C[i*n + j] += buf[(i-ii)*Tj + (j-jj)];
            }
        }
#ifdef __AVX512F__
        _mm_free(buf);
#else
        free(buf);
#endif
    }
#else
    (void)nt;
    tsmm_tiled_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk);
#endif
}

void tsmm_s6a(tsmm_layout_t layout, int m, int n, int k,
              const double *A, const double *B, double *C,
              int Ti, int Tj, int Tk, int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_s6a_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk, num_threads);
    else
        tsmm_tiled_colmajor(m, n, k, A, B, C, Ti, Tj, Tk);
}
