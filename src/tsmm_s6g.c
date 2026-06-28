/** src/tsmm_s6g.c — S7 (442×193×11) AVX-512 + optimal tiles. Ti=64, Tj=193, Tk=11. */
#include "tsmm.h"
void tsmm_s6g_rowmajor(int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads)
{ (void)Ti;(void)Tj;(void)Tk; tsmm_avx512_s5_omp_rowmajor(m,n,k,A,B,C,64,193,11,num_threads); }
void tsmm_s6g(tsmm_layout_t l, int m, int n, int k,
              const double *A, const double *B, double *C,
              int Ti, int Tj, int Tk, int nt)
{ (void)Ti;(void)Tj;(void)Tk; tsmm_avx512_s5_omp(l,m,n,k,A,B,C,64,193,11,nt); }
