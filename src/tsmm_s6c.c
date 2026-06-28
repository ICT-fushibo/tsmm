/**
 * src/tsmm_s6c.c — Stage 6c: S3 (32×16000×16) AVX-512 + optimal tiles
 *
 * Ti=m=32 (eliminate i-tiling), Tj=256, Tk=k=16 (eliminate pk loop).
 * Wraps avx512_s5_omp with per-shape optimal tiles.
 */
#include "tsmm.h"
void tsmm_s6c_rowmajor(int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads)
{ (void)Ti;(void)Tj;(void)Tk; tsmm_avx512_s5_omp_rowmajor(m,n,k,A,B,C,32,256,16,num_threads); }
void tsmm_s6c(tsmm_layout_t l, int m, int n, int k,
              const double *A, const double *B, double *C,
              int Ti, int Tj, int Tk, int nt)
{ (void)Ti;(void)Tj;(void)Tk; tsmm_avx512_s5_omp(l,m,n,k,A,B,C,32,256,16,nt); }
