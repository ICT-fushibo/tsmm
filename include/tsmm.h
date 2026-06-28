#ifndef TSMM_H
#define TSMM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * TSMM: C = A^T * B
 *
 * A is k x m, B is k x n, C is m x n.  All matrices are double-precision
 * (IEEE 754 binary64).  Dense, no special structure assumed.
 * --------------------------------------------------------------------------- */

/* ---------- Matrix storage layout ---------- */
typedef enum {
    TSMM_ROW_MAJOR,   /* C-style: A[i][j] = A[i*ld + j] */
    TSMM_COL_MAJOR    /* Fortran-style: A[i][j] = A[i + j*ld] */
} tsmm_layout_t;

/* ---------- Performance / profiling structure ----------
 * Collected for every run so higher-level tooling (scripts, notebooks) can
 * build roofline plots, trace optimisation progress, and compare across
 * tiling / threading strategies. */
typedef struct {
    /* Primary timing */
    double wall_time_ms;        /* wall-clock elapsed (ms)         */
    double cpu_time_ms;         /* CPU time (ms)                   */

    /* Compute */
    double total_flops;         /* 2 * m * n * k (fma counts 2)   */
    double gflops;              /* total_flops / wall_time / 1e9   */

    /* Memory-traffic estimates (useful for roofline) */
    double bytes_read;          /* (k*m + k*n) * sizeof(double)   */
    double bytes_written;       /* (m*n) * sizeof(double)         */
    double mem_bandwidth_gbs;   /* (bytes_read+bytes_written)
                                   / wall_time / 1e9              */
    double flops_per_byte;      /* arithmetic intensity           */

    /* Execution context */
    int    num_threads;         /* threads used (1 for serial)     */
    int    m, n, k;             /* problem dimensions (cached)     */

    /* Reserved for future use (PAPI counters, cache misses, ...) */
    double l1_miss_ratio;
    double l2_miss_ratio;
    double l3_miss_ratio;
} tsmm_perf_t;

/* ---------- Pre-defined problem shapes ---------- */
typedef struct {
    int m, n, k;
    const char *name;
} tsmm_shape_t;

#define TSMM_NUM_SHAPES 8
extern const tsmm_shape_t tsmm_shapes[TSMM_NUM_SHAPES];

/* ---------- TSMM kernels (naive / serial baseline) ---------- */

void tsmm_naive_rowmajor(int m, int n, int k,
                         const double *A, const double *B, double *C);
void tsmm_naive_colmajor(int m, int n, int k,
                         const double *A, const double *B, double *C);
void tsmm_naive(tsmm_layout_t layout, int m, int n, int k,
                const double *A, const double *B, double *C);

void tsmm_reference(tsmm_layout_t layout, int m, int n, int k,
                    const double *A, const double *B, double *C);

/* ---------- Tiled kernels (serial, cache-blocked) ---------- */

/* 3D tiling (m, n, k).  Ti,Tj,Tk -- tile sizes, runtime-tunable. */
void tsmm_tiled_rowmajor(int m, int n, int k,
                         const double *A, const double *B, double *C,
                         int Ti, int Tj, int Tk);
void tsmm_tiled_colmajor(int m, int n, int k,
                         const double *A, const double *B, double *C,
                         int Ti, int Tj, int Tk);
void tsmm_tiled(tsmm_layout_t layout, int m, int n, int k,
                const double *A, const double *B, double *C,
                int Ti, int Tj, int Tk);

/* 2D tiling (m, n only; k traversed in full per tile). */
void tsmm_tiled_mn_rowmajor(int m, int n, int k,
                            const double *A, const double *B, double *C,
                            int Ti, int Tj);
void tsmm_tiled_mn_colmajor(int m, int n, int k,
                            const double *A, const double *B, double *C,
                            int Ti, int Tj);
void tsmm_tiled_mn(tsmm_layout_t layout, int m, int n, int k,
                   const double *A, const double *B, double *C,
                   int Ti, int Tj);

/* ---------- OpenMP tiled kernels (multi-threaded, row-major only) ---------- */

/* 3D tiling + OpenMP.  num_threads=0 → use OMP_NUM_THREADS env var.
 * col-major variants dispatch to serial tiled (no OMP optimisation).   */
void tsmm_tiled_omp_rowmajor(int m, int n, int k,
                              const double *A, const double *B, double *C,
                              int Ti, int Tj, int Tk, int num_threads);
void tsmm_tiled_omp_colmajor(int m, int n, int k,
                              const double *A, const double *B, double *C,
                              int Ti, int Tj, int Tk, int num_threads);
void tsmm_tiled_omp(tsmm_layout_t layout, int m, int n, int k,
                    const double *A, const double *B, double *C,
                    int Ti, int Tj, int Tk, int num_threads);

/* 2D tiling + OpenMP. */
void tsmm_tiled_omp_mn_rowmajor(int m, int n, int k,
                                 const double *A, const double *B, double *C,
                                 int Ti, int Tj, int num_threads);
void tsmm_tiled_omp_mn_colmajor(int m, int n, int k,
                                 const double *A, const double *B, double *C,
                                 int Ti, int Tj, int num_threads);
void tsmm_tiled_omp_mn(tsmm_layout_t layout, int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int num_threads);

/* ---------- OpenMP Step 2: Ti×Tj + collapse(2) ---------- */

void tsmm_tiled_omp_s2_rowmajor(int m, int n, int k,
                                 const double *A, const double *B, double *C,
                                 int Ti, int Tj, int Tk, int num_threads);
void tsmm_tiled_omp_s2(tsmm_layout_t layout, int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads);

void tsmm_tiled_omp_s2_mn_rowmajor(int m, int n, int k,
                                    const double *A, const double *B, double *C,
                                    int Ti, int Tj, int num_threads);
void tsmm_tiled_omp_s2_mn(tsmm_layout_t layout, int m, int n, int k,
                           const double *A, const double *B, double *C,
                           int Ti, int Tj, int num_threads);

/* ---------- OpenMP Step 3: Ti×Tj + collapse(2) + private buffer ---------- */

void tsmm_tiled_omp_s3_rowmajor(int m, int n, int k,
                                 const double *A, const double *B, double *C,
                                 int Ti, int Tj, int Tk, int num_threads);
void tsmm_tiled_omp_s3(tsmm_layout_t layout, int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads);

void tsmm_tiled_omp_s3_mn_rowmajor(int m, int n, int k,
                                    const double *A, const double *B, double *C,
                                    int Ti, int Tj, int num_threads);
void tsmm_tiled_omp_s3_mn(tsmm_layout_t layout, int m, int n, int k,
                           const double *A, const double *B, double *C,
                           int Ti, int Tj, int num_threads);

/* ---------- OpenMP Step 4: 3D k-parallel + reduction (MN falls back to S3) ---------- */

void tsmm_tiled_omp_s4_rowmajor(int m, int n, int k,
                                 const double *A, const double *B, double *C,
                                 int Ti, int Tj, int Tk, int num_threads);
void tsmm_tiled_omp_s4(tsmm_layout_t layout, int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads);

void tsmm_tiled_omp_s4_mn_rowmajor(int m, int n, int k,
                                    const double *A, const double *B, double *C,
                                    int Ti, int Tj, int num_threads);
void tsmm_tiled_omp_s4_mn(tsmm_layout_t layout, int m, int n, int k,
                           const double *A, const double *B, double *C,
                           int Ti, int Tj, int num_threads);

/* ---------- Stage 6b: S2 (8×16) full register-resident + k-parallel ---------- */

void tsmm_s6b_rowmajor(int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int num_threads);
void tsmm_s6b(tsmm_layout_t layout, int m, int n, int k,
              const double *A, const double *B, double *C,
              int num_threads);

/* ---------- Stage 5: AVX-512 微内核 (8×16 register tile) ---------- */

void tsmm_avx512_s5_rowmajor(int m, int n, int k,
                              const double *A, const double *B, double *C,
                              int Ti, int Tj, int Tk);
void tsmm_avx512_s5(tsmm_layout_t layout, int m, int n, int k,
                     const double *A, const double *B, double *C,
                     int Ti, int Tj, int Tk);

void tsmm_avx512_s5_omp_rowmajor(int m, int n, int k,
                                  const double *A, const double *B, double *C,
                                  int Ti, int Tj, int Tk, int num_threads);
void tsmm_avx512_s5_omp(tsmm_layout_t layout, int m, int n, int k,
                         const double *A, const double *B, double *C,
                         int Ti, int Tj, int Tk, int num_threads);

/* ---------- BLAS wrapper (requires -DTSMM_USE_BLAS) ---------- */
#ifdef TSMM_USE_BLAS
void tsmm_blas(tsmm_layout_t layout, int m, int n, int k,
               const double *A, const double *B, double *C);
#endif

/* ---------- Memory management ---------- */
double* tsmm_alloc_matrix(int rows, int cols);
void    tsmm_free_matrix(double *mat);
void    tsmm_init_random(double *mat, int rows, int cols);
void    tsmm_init_zero(double *mat, int rows, int cols);

/* ---------- Correctness ---------- */
double  tsmm_max_abs_diff(const double *C1, const double *C2,
                          int rows, int cols);
int     tsmm_verify(const double *C1, const double *C2,
                    int rows, int cols, double tol);

/* ---------- Timing ---------- */
double  tsmm_wall_time(void);
double  tsmm_cpu_time(void);

/* ---------- Performance helpers ---------- */
tsmm_perf_t tsmm_compute_perf(int m, int n, int k,
                              double wall_time_ms, int num_threads);
void        tsmm_print_perf(const tsmm_perf_t *p);

#ifdef __cplusplus
}
#endif

#endif /* TSMM_H */
