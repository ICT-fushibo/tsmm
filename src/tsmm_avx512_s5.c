/**
 * src/tsmm_avx512_s5.c — Stage 5: AVX-512 微内核 (8×16 register tile)
 *
 * 仅 row-major。编译需要 -mavx512f -mavx512dq（-march=native on Cascade Lake+）。
 *
 * 两层 tiling:
 *   外层: cache-level tile (Ti, Tj, Tk) — 适配 L2/L3
 *   内层: register-level tile (TR=8, TC=16) — 适配 32 ZMM 寄存器
 *
 * 两种模式:
 *   tsmm_avx512_s5          — 串行
 *   tsmm_avx512_s5_omp      — 多线程 (collapse(2) + private buffer, 同 S3)
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
#define TR  8
#define TC  16

/* ==========================================================================
 *  micro_kernel_8x16 — AVX-512 寄存器微内核
 *
 *  处理 C[ii:ii+8][jj:jj+16] += A^T[ii:ii+8][pk:k_end] × B[pk:k_end][jj:jj+16]
 *
 *  寄存器布局 (16 个 ZMM 存 C):
 *    c[ 0.. 1] = C[ii+0][jj..jj+15]
 *    c[ 2.. 3] = C[ii+1][jj..jj+15]
 *    c[14..15] = C[ii+7][jj..jj+15]
 *
 *  每轮 p:
 *    b0 = B[p][jj..jj+7], b1 = B[p][jj+8..jj+15]
 *    for r = 0..7:
 *      a_bcst = A[p][ii+r]
 *      c[2r]   = FMA(a_bcst, b0, c[2r])
 *      c[2r+1] = FMA(a_bcst, b1, c[2r+1])
 * ========================================================================== */
static inline void micro_kernel_8x16(
    int m, int n,
    const double *A, const double *B, double *C,
    int ii, int jj, int pk, int k_end)
{
#ifdef __AVX512F__
    /* Step 1: load C tile into 16 ZMM registers */
    __m512d c00 = _mm512_load_pd(&C[(ii + 0) * n + jj]);
    __m512d c01 = _mm512_load_pd(&C[(ii + 0) * n + jj + 8]);
    __m512d c10 = _mm512_load_pd(&C[(ii + 1) * n + jj]);
    __m512d c11 = _mm512_load_pd(&C[(ii + 1) * n + jj + 8]);
    __m512d c20 = _mm512_load_pd(&C[(ii + 2) * n + jj]);
    __m512d c21 = _mm512_load_pd(&C[(ii + 2) * n + jj + 8]);
    __m512d c30 = _mm512_load_pd(&C[(ii + 3) * n + jj]);
    __m512d c31 = _mm512_load_pd(&C[(ii + 3) * n + jj + 8]);
    __m512d c40 = _mm512_load_pd(&C[(ii + 4) * n + jj]);
    __m512d c41 = _mm512_load_pd(&C[(ii + 4) * n + jj + 8]);
    __m512d c50 = _mm512_load_pd(&C[(ii + 5) * n + jj]);
    __m512d c51 = _mm512_load_pd(&C[(ii + 5) * n + jj + 8]);
    __m512d c60 = _mm512_load_pd(&C[(ii + 6) * n + jj]);
    __m512d c61 = _mm512_load_pd(&C[(ii + 6) * n + jj + 8]);
    __m512d c70 = _mm512_load_pd(&C[(ii + 7) * n + jj]);
    __m512d c71 = _mm512_load_pd(&C[(ii + 7) * n + jj + 8]);

    /* Step 2: loop over p, accumulate into registers */
    for (int p = pk; p < k_end; p++) {
        /* load B row p: two 512-bit vectors */
        __m512d b0 = _mm512_loadu_pd(&B[p * n + jj]);
        __m512d b1 = _mm512_loadu_pd(&B[p * n + jj + 8]);

        /* row 0 */
        __m512d a0 = _mm512_set1_pd(A[p * m + ii + 0]);
        c00 = _mm512_fmadd_pd(a0, b0, c00);
        c01 = _mm512_fmadd_pd(a0, b1, c01);

        /* row 1 */
        __m512d a1 = _mm512_set1_pd(A[p * m + ii + 1]);
        c10 = _mm512_fmadd_pd(a1, b0, c10);
        c11 = _mm512_fmadd_pd(a1, b1, c11);

        /* row 2 */
        __m512d a2 = _mm512_set1_pd(A[p * m + ii + 2]);
        c20 = _mm512_fmadd_pd(a2, b0, c20);
        c21 = _mm512_fmadd_pd(a2, b1, c21);

        /* row 3 */
        __m512d a3 = _mm512_set1_pd(A[p * m + ii + 3]);
        c30 = _mm512_fmadd_pd(a3, b0, c30);
        c31 = _mm512_fmadd_pd(a3, b1, c31);

        /* row 4 */
        __m512d a4 = _mm512_set1_pd(A[p * m + ii + 4]);
        c40 = _mm512_fmadd_pd(a4, b0, c40);
        c41 = _mm512_fmadd_pd(a4, b1, c41);

        /* row 5 */
        __m512d a5 = _mm512_set1_pd(A[p * m + ii + 5]);
        c50 = _mm512_fmadd_pd(a5, b0, c50);
        c51 = _mm512_fmadd_pd(a5, b1, c51);

        /* row 6 */
        __m512d a6 = _mm512_set1_pd(A[p * m + ii + 6]);
        c60 = _mm512_fmadd_pd(a6, b0, c60);
        c61 = _mm512_fmadd_pd(a6, b1, c61);

        /* row 7 */
        __m512d a7 = _mm512_set1_pd(A[p * m + ii + 7]);
        c70 = _mm512_fmadd_pd(a7, b0, c70);
        c71 = _mm512_fmadd_pd(a7, b1, c71);
    }

    /* Step 3: write C tile back */
    _mm512_store_pd(&C[(ii + 0) * n + jj],      c00);
    _mm512_store_pd(&C[(ii + 0) * n + jj + 8],  c01);
    _mm512_store_pd(&C[(ii + 1) * n + jj],      c10);
    _mm512_store_pd(&C[(ii + 1) * n + jj + 8],  c11);
    _mm512_store_pd(&C[(ii + 2) * n + jj],      c20);
    _mm512_store_pd(&C[(ii + 2) * n + jj + 8],  c21);
    _mm512_store_pd(&C[(ii + 3) * n + jj],      c30);
    _mm512_store_pd(&C[(ii + 3) * n + jj + 8],  c31);
    _mm512_store_pd(&C[(ii + 4) * n + jj],      c40);
    _mm512_store_pd(&C[(ii + 4) * n + jj + 8],  c41);
    _mm512_store_pd(&C[(ii + 5) * n + jj],      c50);
    _mm512_store_pd(&C[(ii + 5) * n + jj + 8],  c51);
    _mm512_store_pd(&C[(ii + 6) * n + jj],      c60);
    _mm512_store_pd(&C[(ii + 6) * n + jj + 8],  c61);
    _mm512_store_pd(&C[(ii + 7) * n + jj],      c70);
    _mm512_store_pd(&C[(ii + 7) * n + jj + 8],  c71);

#else
    (void)m; (void)n; (void)A; (void)B; (void)C;
    (void)ii; (void)jj; (void)pk; (void)k_end;
#endif
}

/* ==========================================================================
 *  S5 row-major: cache tiling + AVX-512 微内核调度
 *
 *  外层 (pk, ii, jj) — 标准 cache tiling，和 S3 相同。
 *  内层 (i, j)   — 8×16 register tiling，调用 micro_kernel_8x16。
 *  尾部           — 标量标量回退。
 * ========================================================================== */
void tsmm_avx512_s5_rowmajor(int m, int n, int k,
                              const double *A, const double *B, double *C,
                              int Ti, int Tj, int Tk)
{
    /* Ensure Ti, Tj are at least micro-tile sizes */
    if (Ti < TR) Ti = TR;
    if (Tj < TC) Tj = TC;

    for (int pk = 0; pk < k; pk += Tk) {
        int k_end = MIN(pk + Tk, k);

        for (int ii = 0; ii < m; ii += Ti) {
            int i_end = MIN(ii + Ti, m);

            for (int jj = 0; jj < n; jj += Tj) {
                int j_end = MIN(jj + Tj, n);

                /* Rounded-down boundaries: only full 8×16 blocks use AVX-512 */
                int i_avx = ii + ((i_end - ii) / TR) * TR;
                int j_avx = jj + ((j_end - jj) / TC) * TC;

                /* AVX-512: full 8×16 micro-tiles */
#ifdef __AVX512F__
                for (int i = ii; i < i_avx; i += TR) {
                    for (int j = jj; j < j_avx; j += TC) {
                        micro_kernel_8x16(m, n, A, B, C, i, j, pk, k_end);
                    }
                }
#endif

                /* Scalar fallback: remainder columns in AVX rows */
                for (int i = ii; i < i_avx; i++) {
                    for (int p = pk; p < k_end; p++) {
                        double a_ip = A[p * m + i];
                        for (int j = j_avx; j < j_end; j++) {
                            C[i * n + j] += a_ip * B[p * n + j];
                        }
                    }
                }

                /* Scalar fallback: remainder rows (all columns) */
                for (int i = i_avx; i < i_end; i++) {
                    for (int p = pk; p < k_end; p++) {
                        double a_ip = A[p * m + i];
                        for (int j = jj; j < j_end; j++) {
                            C[i * n + j] += a_ip * B[p * n + j];
                        }
                    }
                }
            }
        }
    }
}

/* ==========================================================================
 *  S5 OMP variant: AVX-512 微内核 + collapse(2) + private buffer
 *
 *  并行策略和 S3 完全相同：
 *    - collapse(2) 分 (ii,jj) tile
 *    - 每线程 private buffer (Ti×Tj)
 *    - 微内核写到 private buffer → 所有 k 完成 → 写回 C
 *
 *  和 S3 的唯一区别：内层用 AVX-512 micro_kernel_8x16 替代标量 p-i-j。
 * ========================================================================== */
void tsmm_avx512_s5_omp_rowmajor(int m, int n, int k,
                                  const double *A, const double *B, double *C,
                                  int Ti, int Tj, int Tk, int num_threads)
{
    if (Ti < TR) Ti = TR;
    if (Tj < TC) Tj = TC;

#ifdef _OPENMP
    int nt = (num_threads > 0) ? num_threads : omp_get_max_threads();

    #pragma omp parallel num_threads(nt)
    {
        double *buf = (double*)calloc((size_t)Ti * Tj, sizeof(double));
        if (!buf) { fprintf(stderr, "s5_omp: buf alloc failed\n"); exit(1); }

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
                    /* AVX-512: full 8×16 blocks */
                    for (int i = ii; i < i_avx; i += TR) {
                        for (int j = jj; j < j_avx; j += TC) {
                            /* inline micro-kernel writing to buf */
                            __m512d c00 = _mm512_load_pd(&buf[(i - ii) * Tj + (j - jj)]);
                            __m512d c01 = _mm512_load_pd(&buf[(i - ii) * Tj + (j - jj) + 8]);
                            __m512d c10 = _mm512_load_pd(&buf[(i + 1 - ii) * Tj + (j - jj)]);
                            __m512d c11 = _mm512_load_pd(&buf[(i + 1 - ii) * Tj + (j - jj) + 8]);
                            __m512d c20 = _mm512_load_pd(&buf[(i + 2 - ii) * Tj + (j - jj)]);
                            __m512d c21 = _mm512_load_pd(&buf[(i + 2 - ii) * Tj + (j - jj) + 8]);
                            __m512d c30 = _mm512_load_pd(&buf[(i + 3 - ii) * Tj + (j - jj)]);
                            __m512d c31 = _mm512_load_pd(&buf[(i + 3 - ii) * Tj + (j - jj) + 8]);
                            __m512d c40 = _mm512_load_pd(&buf[(i + 4 - ii) * Tj + (j - jj)]);
                            __m512d c41 = _mm512_load_pd(&buf[(i + 4 - ii) * Tj + (j - jj) + 8]);
                            __m512d c50 = _mm512_load_pd(&buf[(i + 5 - ii) * Tj + (j - jj)]);
                            __m512d c51 = _mm512_load_pd(&buf[(i + 5 - ii) * Tj + (j - jj) + 8]);
                            __m512d c60 = _mm512_load_pd(&buf[(i + 6 - ii) * Tj + (j - jj)]);
                            __m512d c61 = _mm512_load_pd(&buf[(i + 6 - ii) * Tj + (j - jj) + 8]);
                            __m512d c70 = _mm512_load_pd(&buf[(i + 7 - ii) * Tj + (j - jj)]);
                            __m512d c71 = _mm512_load_pd(&buf[(i + 7 - ii) * Tj + (j - jj) + 8]);

                            for (int p = pk; p < k_end; p++) {
                                __m512d b0 = _mm512_loadu_pd(&B[p * n + j]);
                                __m512d b1 = _mm512_loadu_pd(&B[p * n + j + 8]);
                                __m512d a0 = _mm512_set1_pd(A[p * m + i + 0]);
                                c00 = _mm512_fmadd_pd(a0, b0, c00); c01 = _mm512_fmadd_pd(a0, b1, c01);
                                __m512d a1 = _mm512_set1_pd(A[p * m + i + 1]);
                                c10 = _mm512_fmadd_pd(a1, b0, c10); c11 = _mm512_fmadd_pd(a1, b1, c11);
                                __m512d a2 = _mm512_set1_pd(A[p * m + i + 2]);
                                c20 = _mm512_fmadd_pd(a2, b0, c20); c21 = _mm512_fmadd_pd(a2, b1, c21);
                                __m512d a3 = _mm512_set1_pd(A[p * m + i + 3]);
                                c30 = _mm512_fmadd_pd(a3, b0, c30); c31 = _mm512_fmadd_pd(a3, b1, c31);
                                __m512d a4 = _mm512_set1_pd(A[p * m + i + 4]);
                                c40 = _mm512_fmadd_pd(a4, b0, c40); c41 = _mm512_fmadd_pd(a4, b1, c41);
                                __m512d a5 = _mm512_set1_pd(A[p * m + i + 5]);
                                c50 = _mm512_fmadd_pd(a5, b0, c50); c51 = _mm512_fmadd_pd(a5, b1, c51);
                                __m512d a6 = _mm512_set1_pd(A[p * m + i + 6]);
                                c60 = _mm512_fmadd_pd(a6, b0, c60); c61 = _mm512_fmadd_pd(a6, b1, c61);
                                __m512d a7 = _mm512_set1_pd(A[p * m + i + 7]);
                                c70 = _mm512_fmadd_pd(a7, b0, c70); c71 = _mm512_fmadd_pd(a7, b1, c71);
                            }

                            _mm512_store_pd(&buf[(i - ii) * Tj + (j - jj)],      c00);
                            _mm512_store_pd(&buf[(i - ii) * Tj + (j - jj) + 8],  c01);
                            _mm512_store_pd(&buf[(i + 1 - ii) * Tj + (j - jj)],      c10);
                            _mm512_store_pd(&buf[(i + 1 - ii) * Tj + (j - jj) + 8],  c11);
                            _mm512_store_pd(&buf[(i + 2 - ii) * Tj + (j - jj)],      c20);
                            _mm512_store_pd(&buf[(i + 2 - ii) * Tj + (j - jj) + 8],  c21);
                            _mm512_store_pd(&buf[(i + 3 - ii) * Tj + (j - jj)],      c30);
                            _mm512_store_pd(&buf[(i + 3 - ii) * Tj + (j - jj) + 8],  c31);
                            _mm512_store_pd(&buf[(i + 4 - ii) * Tj + (j - jj)],      c40);
                            _mm512_store_pd(&buf[(i + 4 - ii) * Tj + (j - jj) + 8],  c41);
                            _mm512_store_pd(&buf[(i + 5 - ii) * Tj + (j - jj)],      c50);
                            _mm512_store_pd(&buf[(i + 5 - ii) * Tj + (j - jj) + 8],  c51);
                            _mm512_store_pd(&buf[(i + 6 - ii) * Tj + (j - jj)],      c60);
                            _mm512_store_pd(&buf[(i + 6 - ii) * Tj + (j - jj) + 8],  c61);
                            _mm512_store_pd(&buf[(i + 7 - ii) * Tj + (j - jj)],      c70);
                            _mm512_store_pd(&buf[(i + 7 - ii) * Tj + (j - jj) + 8],  c71);
                        }
                    }
#endif

                    /* Scalar fallback: remainder columns in AVX rows */
                    for (int i = ii; i < i_avx; i++) {
                        for (int p = pk; p < k_end; p++) {
                            double a_ip = A[p * m + i];
                            for (int j = j_avx; j < j_end; j++)
                                buf[(i - ii) * Tj + (j - jj)] += a_ip * B[p * n + j];
                        }
                    }

                    /* Scalar fallback: remainder rows */
                    for (int i = i_avx; i < i_end; i++) {
                        for (int p = pk; p < k_end; p++) {
                            double a_ip = A[p * m + i];
                            for (int j = jj; j < j_end; j++)
                                buf[(i - ii) * Tj + (j - jj)] += a_ip * B[p * n + j];
                        }
                    }
                }

                /* Write back buf → C */
                for (int i = ii; i < i_end; i++)
                    for (int j = jj; j < j_end; j++)
                        C[i * n + j] += buf[(i - ii) * Tj + (j - jj)];
            }
        }
        free(buf);
    }
#else
    /* No OpenMP: fallback to serial S5 */
    (void)num_threads;
    tsmm_avx512_s5_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk);
#endif
}

/* ==========================================================================
 *  S5 分发 (col-major → 串行 fallback)
 * ========================================================================== */
void tsmm_avx512_s5(tsmm_layout_t layout, int m, int n, int k,
                     const double *A, const double *B, double *C,
                     int Ti, int Tj, int Tk)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_avx512_s5_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk);
    else
        tsmm_tiled_colmajor(m, n, k, A, B, C, Ti, Tj, Tk);
}

void tsmm_avx512_s5_omp(tsmm_layout_t layout, int m, int n, int k,
                         const double *A, const double *B, double *C,
                         int Ti, int Tj, int Tk, int num_threads)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_avx512_s5_omp_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk, num_threads);
    else
        tsmm_tiled_colmajor(m, n, k, A, B, C, Ti, Tj, Tk);
}
