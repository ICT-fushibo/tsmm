/** src/tsmm_s6e.c — S5 (16×12344×16) AVX-512 + optimal tiles. Ti=16, Tj=256, Tk=16. */
#include "tsmm.h"
void tsmm_s6e_rowmajor(int m, int n, int k,
                        const double *A, const double *B, double *C,
                        int Ti, int Tj, int Tk, int num_threads)
{ (void)Ti;(void)Tj;(void)Tk; tsmm_avx512_s5_omp_rowmajor(m,n,k,A,B,C,16,256,16,num_threads); }
void tsmm_s6e(tsmm_layout_t l, int m, int n, int k,
              const double *A, const double *B, double *C,
              int Ti, int Tj, int Tk, int nt)
{ (void)Ti;(void)Tj;(void)Tk; tsmm_avx512_s5_omp(l,m,n,k,A,B,C,16,256,16,nt); }
