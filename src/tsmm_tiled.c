/**
 * src/tsmm_tiled.c — Serial TSMM with cache tiling (blocking).
 *
 * C = A^T * B  其中 A: kxm,  B: kxn,  C: mxn  (全部 double)
 *
 * 调用约定：C 在调用前必须已初始化为零。
 *
 * Tile sizes (Ti, Tj, Tk) 作为运行时参数传入，方便后续对不同 shape 调参对比。
 *
 * ===========================================================================
 *  Tiling 原理
 * ===========================================================================
 *
 *  Naive p-i-j 的问题：
 *    每轮 p 迭代都要读+写整个 C(mxn)。共 k 轮 -> C 的 DRAM 流量 = 2*m*n*k*8 B。
 *    以 S1(4000,160000,128) 为例: C=5.12GB, 128轮 -> C 总流量 >1TB。
 *    L3 只有 36MB, C 完全装不下, 性能被内存带宽锁死。
 *
 *  Tiling 思想：
 *    将 C 切成小块(TixTj), 每块完整遍历 k 后再移动到下一块。
 *    每个 C 元素只 DRAM load 1 次 + store 1 次（不是 naive 的 k 次）。
 *
 *  3D tiling (m,n,k):
 *    - C 的 TixTj 块留在 L2/L1
 *    - B 的 TkxTj 条带也留在 L2
 *    - 适合 k 很大的 shape (S6: k=606841)
 *
 *  2D tiling (m,n only):
 *    - 只切 i,j 方向, k 全量遍历
 *    - 少一层 tile loop, 循环开销更低
 *    - 适合 k 很小的 shape (S1: k=128)
 *
 *  边界处理：
 *    每维最后一个 tile 天然更小, 直接用 MIN(ii+Ti, m) 得到实际边界。
 *    不需要 pad, 不需要填零——micro-kernel 用小一点的尺寸直接算。
 *
 *  Tile 大小建议 (BSCC-T6: L1d=32KB, L2=1024KB, cacheline=64B):
 *    Ti = 64 ~ 128   (cache line 对齐: 8 的倍数)
 *    Tj = 256 ~ 512
 *    Tk = 256 ~ 512
 *    约束: Ti*Tj*8 + Tk*Tj*8 <= L2 ~ 1MB
 * ===========================================================================
 */

#include "tsmm.h"
#include <string.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* ==========================================================================
 *  Row-major  3D tiling (m, n, k)
 *
 *  六重循环:
 *    for (pk = 0; pk < k; pk += Tk)
 *      for (ii = 0; ii < m; ii += Ti)
 *        for (jj = 0; jj < n; jj += Tj)
 *          // micro-kernel (p,i,j)
 *          for (p = pk; p < k_end; p++)
 *            for (i = ii; i < i_end; i++) {
 *              a = A[p*m + i];         // hoist
 *              for (j = jj; j < j_end; j++)
 *                C[i*n + j] += a * B[p*n + j];
 *            }
 *
 *  TODO: 填充 micro-kernel 代码。
 *        框架 (tile loops, 边界计算) 已写好。
 * ========================================================================== */
void tsmm_tiled_rowmajor(int m, int n, int k,
                         const double *A, const double *B, double *C,
                         int Ti, int Tj, int Tk)
{
    for (int pk = 0; pk < k; pk += Tk) {
        int k_end = MIN(pk + Tk, k);
        for (int ii = 0; ii < m; ii += Ti) {
            int i_end = MIN(ii + Ti, m);
            for (int jj = 0; jj < n; jj += Tj) {
                int j_end = MIN(jj + Tj, n);

                /* ---- micro-kernel ---- */
                /* TODO: 实现 p-i-j inner loops */
                
                for(int p =pk;p<k_end;p++){
                    for(int i=ii;i<i_end;i++){
                        double a_pi=A[p*m+i];
                        for (int j=jj;j<j_end;j++){
                            //c[i,j]+=a^t[i,p]*b[p,j]=a[p,j]*b[p,j]
                            C[i*n+j]+=a_pi*B[p*n+j];
                        }
                    }
                }
                /* ---- end micro-kernel ---- */
            }
        }
    }
}

/* ==========================================================================
 *  Row-major  2D tiling (m, n only)
 *
 *  五重循环 (比 3D 少一层 pk):
 *    for (ii = 0; ii < m; ii += Ti)
 *      for (jj = 0; jj < n; jj += Tj)
 *        // micro-kernel (p,i,j) with k 全量遍历
 *        for (p = 0; p < k; p++)
 *          for (i = ii; i < i_end; i++) {
 *            a = A[p*m + i];
 *            for (j = jj; j < j_end; j++)
 *              C[i*n + j] += a * B[p*n + j];
 *          }
 *
 *  TODO: 填充 micro-kernel 代码。
 * ========================================================================== */
void tsmm_tiled_mn_rowmajor(int m, int n, int k,
                            const double *A, const double *B, double *C,
                            int Ti, int Tj)
{
    // for(int p=0;p<k;p++){ //change to p-i-j loop
    //     for (int ii = 0; ii < m; ii += Ti) {
    //         int i_end = MIN(ii + Ti, m);
    //         for (int jj = 0; jj < n; jj += Tj) {
    //             int j_end = MIN(jj + Tj, n);

    //             /* ---- micro-kernel: k 全量遍历 ---- */
    //             for (int i=ii;i<i_end;i++){
    //                 double a_pi=A[p*m+i];
    //                 for(int j=jj;j<j_end;j++){
    //                     C[i*m+j]+=a_pi*B[p*n+j];
                    
    //                 }
    //             }
    //             /* ---- end micro-kernel ---- */
    //         }
    //     }
    // }
    for (int ii = 0; ii < m; ii += Ti) {
        int i_end = MIN(ii + Ti, m);
        for (int jj = 0; jj < n; jj += Tj) {
            int j_end = MIN(jj + Tj, n);

            /* ---- micro-kernel: k 全量遍历 ---- */
            for(int p=0;p<k;p++){
                for (int i=ii;i<i_end;i++){
                    for(int j=jj;j<j_end;j++){
                        C[i*n+j]+=A[p*m+i]*B[p*n+j];
                    }
                }
            }
            /* ---- end micro-kernel ---- */
        }
    }
}

/* ==========================================================================
 *  Column-major  3D tiling (m, n, k)
 *
 *  列主序索引:
 *    A[p][i] = A[p + i*k]     (p 连续 = stride-1)
 *    B[p][j] = B[p + j*k]     (p 连续 = stride-1)
 *    C[i][j] = C[i + j*m]     (i 连续 = stride-1)
 *
 *  参考 micro-kernel (naive i-j-p 风格, p 在最内层为 stride-1):
 *    for (i = ii; i < i_end; i++)
 *      for (j = jj; j < j_end; j++)
 *        for (p = pk; p < k_end; p++)
 *          C[i + j*m] += A[p + i*k] * B[p + j*k];
 *
 *  TODO: 实现完整的 tile loop + micro-kernel。
 *        注意选择外层 tile loop 顺序以减少 cache miss。
 * ========================================================================== */
void tsmm_tiled_colmajor(int m, int n, int k,
                         const double *A, const double *B, double *C,
                         int Ti, int Tj, int Tk)
{
    /* TODO: 实现列主序 3D tiling */
    (void)m; (void)n; (void)k; (void)A; (void)B; (void)C;
    (void)Ti; (void)Tj; (void)Tk;
    for(int ii=0;ii<m;ii+=Ti){
        int i_end=MIN(ii+Ti,m);
        for(int jj=0;jj<n;jj+=Tj){
            int j_end=MIN(jj+Tj,n);
            for(int pk=0;pk<k;pk+=Tk){
                int p_end=MIN(pk+Tk,k);
                /*micro-kernel*/
                for(int i=ii;i<i_end;i++){
                    for(int j=jj;j<j_end;j++){
                        for(int p=pk;p<p_end;p++){
                            C[i+j*m]+=A[p+i*k]*B[p+j*k];
                        }
                    }
                }

            }
        }
    }
}

/* ==========================================================================
 *  Column-major  2D tiling (m, n only)
 *
 *  TODO: 实现列主序 m,n-only tiling。
 *        k 不分块, p 跑满 0..k-1。
 * ========================================================================== */
void tsmm_tiled_mn_colmajor(int m, int n, int k,
                            const double *A, const double *B, double *C,
                            int Ti, int Tj)
{
    /* TODO: 实现列主序 2D tiling */
    for(int ii=0;ii<m;ii+=Ti){
        int i_end=MIN(ii+Ti,m);
        for(int jj=0;jj<n;jj+=Tj){
            int j_end=MIN(jj+Tj,n);
            /*micro-kernel*/
            for(int i=ii;i<i_end;i++){
                for(int j=jj;j<j_end;j++){
                    for(int p=0;p<k;p++){
                        C[i+j*m]+=A[p+i*k]*B[p+j*k];
                    }
                }
            }
        }
    }
}

/* ==========================================================================
 *  通用分发
 * ========================================================================== */
void tsmm_tiled(tsmm_layout_t layout, int m, int n, int k,
                const double *A, const double *B, double *C,
                int Ti, int Tj, int Tk)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_tiled_rowmajor(m, n, k, A, B, C, Ti, Tj, Tk);
    else
        tsmm_tiled_colmajor(m, n, k, A, B, C, Ti, Tj, Tk);
}

void tsmm_tiled_mn(tsmm_layout_t layout, int m, int n, int k,
                   const double *A, const double *B, double *C,
                   int Ti, int Tj)
{
    if (layout == TSMM_ROW_MAJOR)
        tsmm_tiled_mn_rowmajor(m, n, k, A, B, C, Ti, Tj);
    else
        tsmm_tiled_mn_colmajor(m, n, k, A, B, C, Ti, Tj);
}
