/**
 * src/tsmm_naive.c — Serial TSMM kernels (baseline, no tiling / no SIMD).
 *
 * TODO: 实现以下函数。
 *
 * C = A^T * B  其中 A: k×m,  B: k×n,  C: m×n  (全部 double)
 *
 * 调用约定：
 *   C 在调用前必须已初始化为零（memset 或等价方式）。
 *   本 kernel 在 C 上累加，以便后续 tiling / 多线程 时复用同一模式。
 *
 * 提示：
 *   - 选择合适的 loop order 让最内层循环为 stride-1 访问。
 *   - 行主序和列主序的最优 loop order 不同。
 *   - tsmm_reference 是故意写慢的，仅用于小规模正确性校验。
 */

#include "tsmm.h"
#include <string.h>

/* --------------------------------------------------------------------------
 *  Row-major  (C layout)
 *
 *  A[k][m] row-major  →  A[p][i] = A[p*m + i]
 *  B[k][n] row-major  →  B[p][j] = B[p*n + j]
 *  C[m][n] row-major  →  C[i][j] = C[i*n + j]
 *
 *  C[i][j] = Σ_{p=0}^{k-1} A[p][i] * B[p][j]
 *
 * -------------------------------------------------------------------------- */
void tsmm_naive_rowmajor(int m, int n, int k,
                         const double *A, const double *B, double *C)
{
    (void)m; (void)n; (void)k; (void)A; (void)B; (void)C;
    
    // // naive i-j-p loop
    // // A B cache miss

    // for (int i = 0; i < m; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         for (int p = 0; p < k; p++)
    //         {
    //             // cal c[i][j]
    //             // c[i][j]+=a^t[i][p]*b[p][j]=a[p][i]*b[p][j]
    //             // row major a:m*n a[i][j]=a[i*n+j]
    //             C[i*n+j]+=A[p*m+i]*B[p*n+j];
    //         }
    //     }
        
    // }

    // opt p-i-j loop
    for (int p =0; p<k;p++)
    {
        for ( int i = 0; i<m;i++)
        {
            double a_pi=A[p*m+i];
            for (int j=0;j<n;j++)
            {
                C[i*n+j]+=a_pi*B[p*n+j];
            }
        }
    }
    
}

/* --------------------------------------------------------------------------
 *  Column-major  (Fortran layout)
 *
 *  A[k][m] col-major  →  A[p][i] = A[p + i*k]
 *  B[k][n] col-major  →  B[p][j] = B[p + j*k]
 *  C[m][n] col-major  →  C[i][j] = C[i + j*m]
 *
 *  C[i][j] = Σ_{p=0}^{k-1} A[p][i] * B[p][j]
 *
 * -------------------------------------------------------------------------- */
void tsmm_naive_colmajor(int m, int n, int k,
                         const double *A, const double *B, double *C)
{
    (void)m; (void)n; (void)k; (void)A; (void)B; (void)C;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int p = 0; p < k; p++)
            {
                // cal c[i][j]
                // c[i][j]+=a^t[i][p]*b[p][j]=a[p][i]*b[p][j]
                // col major a:m*n a[i][j]=a[i+j*m]
                C[i+j*m]+=A[p+i*k]*B[p+j*k];
            }
        }
        
    }
    
}

/* ---------- 通用分发（按 layout 枚举选择上面两个实现） ---------- */

void tsmm_naive(tsmm_layout_t layout, int m, int n, int k,
                const double *A, const double *B, double *C)
{
    if (layout == TSMM_ROW_MAJOR) {
        tsmm_naive_rowmajor(m, n, k, A, B, C);
    } else {
        tsmm_naive_colmajor(m, n, k, A, B, C);
    }
}

/* --------------------------------------------------------------------------
 *  参考实现（仅用于正确性校验，不追求性能）
 *
 *  最直白的三重循环 i-j-p（内积），每步都直接计算 C[i][j] 的完整点积。
 *  实现简单、不易出错，但性能很差——仅在对小矩阵做 gold-reference 时使用。
 *
 *  注意：此函数自己负责清零 C（调用者不需要预先清零）。
 * -------------------------------------------------------------------------- */
void tsmm_reference(tsmm_layout_t layout, int m, int n, int k,
                    const double *A, const double *B, double *C)
{

    (void)layout; (void)m; (void)n; (void)k; (void)A; (void)B; (void)C;
    tsmm_init_zero(C,m,n);
    if (layout == TSMM_ROW_MAJOR) {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int p = 0; p < k; p++)
                {
                    // cal c[i][j]
                    // c[i][j]+=a^t[i][p]*b[p][j]=a[p][i]*b[p][j]
                    // row major a:m*n a[i][j]=a[i*n+j]
                    C[i*n+j]+=A[p*m+i]*B[p*n+j];
                }
            }
        
        }
    } else {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int p = 0; p < k; p++)
                {
                    // cal c[i][j]
                    // c[i][j]+=a^t[i][p]*b[p][j]=a[p][i]*b[p][j]
                    // col major a:m*n a[i][j]=a[i+j*m]
                    C[i+j*m]+=A[p+i*k]*B[p+j*k];
                }
            }
            
        }
    }
}
