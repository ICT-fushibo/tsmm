/**
 * src/tsmm_s6f.c — Stage 6f: S6 (4×64×606841) k-parallel + B packing
 *
 * S6: C=2KB, k=606841 massive, AI=0.94 bandwidth-bound.
 *
 * Strategy: S4 (k-parallel) + AVX-512 micro-kernel + B packing.
 * Pack B[pk:k_end][jj:jj+Tj] into contiguous buffer → eliminates
 * stride-n TLB misses (n=64 for S6, but k is huge → each B row load
 * jumps 64 doubles = 512 bytes → good stride, so TLB issue is minor.
 * Main win: AVX-512 continuous load from packed B enables 8× FMA).
 *
 * The key insight for S6: k is so large (606841) that the entire
 * computation is dominated by loading A and B repeatedly.
 * With k-parallel (S4), we distribute k across threads.
 * With B packing, we eliminate gather-loads for B.
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

void tsmm_s6f_rowmajor(int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads)
{
#ifdef _OPENMP
    int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();
#else
    int nt = 1; (void)num_threads;
#endif
    int nt_i = (m + Ti - 1) / Ti;
    int nt_j = (n + Tj - 1) / Tj;
    int total_tiles = nt_i * nt_j;
    int pk_iters = (k + Tk - 1) / Tk;

#ifdef _OPENMP
    if (total_tiles > 1000 || pk_iters < nt / 4) {
        tsmm_tiled_omp_s3_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk, num_threads);
        return;
    }

    /* shared for reduction */
    double *shared = malloc((size_t)Ti * Tj * sizeof(double));
    if (!shared) { fprintf(stderr, "s6f: shared failed\n"); exit(1); }

    #pragma omp parallel num_threads(nt)
    {
#ifdef __AVX512F__
        double *buf   = (double*)_mm_malloc((size_t)Ti * Tj * sizeof(double), 64);
        double *packB = (double*)_mm_malloc((size_t)Tk * Tj * sizeof(double), 64);
#else
        double *buf   = calloc((size_t)Ti * Tj, sizeof(double));
        double *packB = malloc((size_t)Tk * Tj * sizeof(double));
#endif
        if (!buf || !packB) { fprintf(stderr, "s6f: alloc failed\n"); exit(1); }

        for (int tile = 0; tile < total_tiles; tile++) {
            int ii = (tile / nt_j) * Ti;
            int jj = (tile % nt_j) * Tj;
            int i_end = MIN(ii + Ti, m);
            int j_end = MIN(jj + Tj, n);
            int i_avx = ii + ((i_end - ii) / TR) * TR;
            int j_avx = jj + ((j_end - jj) / TC) * TC;

            #pragma omp single
            memset(shared, 0, (size_t)Ti * Tj * sizeof(double));
            memset(buf, 0, (size_t)Ti * Tj * sizeof(double));

            #pragma omp for schedule(static)
            for (int pk = 0; pk < k; pk += Tk) {
                int k_end = MIN(pk + Tk, k);

                /* --- Pack B[pk:k_end][jj:j_end] → packB --- */
                for (int p = pk; p < k_end; p++)
                    for (int j = jj; j < j_end; j++)
                        packB[(p - pk) * Tj + (j - jj)] = B[p * n + j];

#ifdef __AVX512F__
                for (int i = ii; i < i_avx; i += TR) {
                    for (int j = jj; j < j_avx; j += TC) {
                        __m512d c00 = _mm512_loadu_pd(&buf[(i-ii)*Tj + (j-jj)]);
                        __m512d c01 = _mm512_loadu_pd(&buf[(i-ii)*Tj + (j-jj) + 8]);
                        __m512d c10 = _mm512_loadu_pd(&buf[(i+1-ii)*Tj + (j-jj)]);
                        __m512d c11 = _mm512_loadu_pd(&buf[(i+1-ii)*Tj + (j-jj) + 8]);
                        __m512d c20 = _mm512_loadu_pd(&buf[(i+2-ii)*Tj + (j-jj)]);
                        __m512d c21 = _mm512_loadu_pd(&buf[(i+2-ii)*Tj + (j-jj) + 8]);
                        __m512d c30 = _mm512_loadu_pd(&buf[(i+3-ii)*Tj + (j-jj)]);
                        __m512d c31 = _mm512_loadu_pd(&buf[(i+3-ii)*Tj + (j-jj) + 8]);
                        __m512d c40 = _mm512_loadu_pd(&buf[(i+4-ii)*Tj + (j-jj)]);
                        __m512d c41 = _mm512_loadu_pd(&buf[(i+4-ii)*Tj + (j-jj) + 8]);
                        __m512d c50 = _mm512_loadu_pd(&buf[(i+5-ii)*Tj + (j-jj)]);
                        __m512d c51 = _mm512_loadu_pd(&buf[(i+5-ii)*Tj + (j-jj) + 8]);
                        __m512d c60 = _mm512_loadu_pd(&buf[(i+6-ii)*Tj + (j-jj)]);
                        __m512d c61 = _mm512_loadu_pd(&buf[(i+6-ii)*Tj + (j-jj) + 8]);
                        __m512d c70 = _mm512_loadu_pd(&buf[(i+7-ii)*Tj + (j-jj)]);
                        __m512d c71 = _mm512_loadu_pd(&buf[(i+7-ii)*Tj + (j-jj) + 8]);

                        for (int p = pk; p < k_end; p++) {
                            __m512d b0 = _mm512_loadu_pd(&packB[(p-pk)*Tj + (j-jj)]);
                            __m512d b1 = _mm512_loadu_pd(&packB[(p-pk)*Tj + (j-jj) + 8]);
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
                /* Scalar fallbacks */
                for (int i = ii; i < i_avx; i++)
                    for (int p = pk; p < k_end; p++) {
                        double a_ip = A[p*m + i];
                        for (int j = j_avx; j < j_end; j++)
                            buf[(i-ii)*Tj+(j-jj)] += a_ip * packB[(p-pk)*Tj+(j-jj)];
                    }
                for (int i = i_avx; i < i_end; i++)
                    for (int p = pk; p < k_end; p++) {
                        double a_ip = A[p*m + i];
                        for (int j = jj; j < j_end; j++)
                            buf[(i-ii)*Tj+(j-jj)] += a_ip * packB[(p-pk)*Tj+(j-jj)];
                    }
            }

            #pragma omp critical
            for (int i = ii; i < i_end; i++)
                for (int j = jj; j < j_end; j++)
                    shared[(i-ii)*Tj+(j-jj)] += buf[(i-ii)*Tj+(j-jj)];
            #pragma omp barrier

            #pragma omp single
            {
                for (int i = ii; i < i_end; i++)
                    for (int j = jj; j < j_end; j++)
                        C[i*n+j] += shared[(i-ii)*Tj+(j-jj)];
            }
        }
#ifdef __AVX512F__
        _mm_free(buf); _mm_free(packB);
#else
        free(buf); free(packB);
#endif
    }
    free(shared);
#else
    (void)nt; (void)total_tiles; (void)pk_iters;
    tsmm_tiled_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk);
#endif
}

void tsmm_s6f(tsmm_layout_t layout, int m, int n, int k,
              const double *A, const double *B, double *C,
              int Ti, int Tj, int Tk, int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_s6f_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk, num_threads);
    else
        tsmm_tiled_colmajor(m, n, k, A, B, C, Ti, Tj, Tk);
}
