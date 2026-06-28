/** src/tsmm_s6d.c — S4 (144³) AVX-512 + optimal tiles. Ti=72, Tj=144, Tk=144. */
#include "tsmm.h"
void tsmm_s6d_rowmajor(int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads)
{ (void)Ti;(void)Tj;(void)Tk; tsmm_avx512_s5_omp_rowmajor(m,n,k,A,B,C,72,144,144,num_threads); }
void tsmm_s6d(tsmm_layout_t l, int m, int n, int k,
              const double *A, const double *B, double *C,
              int Ti, int Tj, int Tk, int nt)
{ (void)Ti;(void)Tj;(void)Tk; tsmm_avx512_s5_omp(l,m,n,k,A,B,C,72,144,144,nt); }
