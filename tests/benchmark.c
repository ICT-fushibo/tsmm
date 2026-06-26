/**
 * tests/benchmark.c — TSMM 性能对比框架（你的实现 vs OpenBLAS vs MKL）。
 *
 * TODO: 填充 benchmark 逻辑。
 *
 * 框架已提供：
 *   - 命令行参数解析（--csv, --runs, --shape, --layout, --threads, --no-blas）
 *   - 线程数控制（OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS）
 *   - 遍历 shape × layout 组合
 *   - CSV / 文本双输出模式
 *
 * 你需要做：
 *   1. bench_one() 中：分配矩阵 → warmup → 多次计时 → 取最优 → 计算 perf → 输出。
 *   2. 对自己的 kernel 和 cblas_dgemm 分别计时（BLAS 部分用 #ifdef TSMM_USE_BLAS 包裹）。
 *   3. 公平对比：相同数据、相同线程数、warmup 后取 min time。
 *
 * 性能指标不只 GFLOPS：
 *   通过 tsmm_compute_perf() 同时得到 wall_time、GFLOPS、内存带宽、算术强度（AI），
 *   便于后续画 Roofline 图、追踪优化效果。
 *
 * 用法：
 *   ./build/benchmark                           # 全部 shape，文本输出
 *   ./build/benchmark --csv                     # CSV 输出（便于 Python 分析）
 *   ./build/benchmark --shape 4 --layout 0      # 只测 shape 4, row-major
 *   ./build/benchmark --runs 10 --threads 1     # 每例跑 10 次，单线程
 *   ./build/benchmark --no-blas                 # 不测 BLAS（无外部依赖时）
 */

#include "tsmm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TSMM_USE_BLAS
#include <cblas.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------- 默认参数 ---------- */
#define DEFAULT_NRUNS    5
#define DEFAULT_NWARMUP  2

/* ---------- 命令行选项 ---------- */
typedef struct {
    int csv_output;
    int nruns;
    int shape_index;    /* 0=all, 1-8=specific */
    int layout_opt;     /* 0=row, 1=col, 2=both */
    int no_blas;
    int num_threads;
} bench_opts_t;

/* ---------- 前向声明 ---------- */
static void parse_args(int argc, char **argv, bench_opts_t *opts);
static void bench_one(const tsmm_shape_t *shape, tsmm_layout_t layout,
                      const bench_opts_t *opts);

/* ========================================================================== */

int main(int argc, char **argv)
{
    bench_opts_t opts;
    parse_args(argc, argv, &opts);

    /* --- 线程控制 --- */
#ifdef _OPENMP
    omp_set_num_threads(opts.num_threads);
#endif
    /* MKL / OpenBLAS 线程数通过环境变量控制更好（benchmark 脚本设置），
       此处仅打印提示。 */
    printf("# TSMM Benchmark\n");
    printf("# Threads: OMP_NUM_THREADS=%s  requested=%d\n",
           getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "(unset)",
           opts.num_threads);
    printf("# Runs: %d (warmup: %d)\n", opts.nruns, DEFAULT_NWARMUP);
    printf("#\n");

    /* --- 迭代 --- */
    tsmm_layout_t layouts[2] = { TSMM_ROW_MAJOR, TSMM_COL_MAJOR };
    int n_layouts = (opts.layout_opt == 2) ? 2 : 1;
    int loffset   = (opts.layout_opt == 1) ? 1 : 0;

    int istart = (opts.shape_index == 0) ? 0 : (opts.shape_index - 1);
    int iend   = (opts.shape_index == 0) ? TSMM_NUM_SHAPES : opts.shape_index;

    if (opts.csv_output) {
        printf("kernel,shape,m,n,k,layout,wall_ms,gflops,"
               "mem_bw_gbs,ai,total_flops,bytes_read,bytes_written,threads\n");
    }

    for (int si = istart; si < iend; ++si) {
        for (int li = 0; li < n_layouts; ++li) {
            bench_one(&tsmm_shapes[si], layouts[loffset + li], &opts);
        }
    }

    return 0;
}

/* ==========================================================================
 *  bench_one — 对单个 (shape, layout) 执行 benchmark
 *
 *  TODO: 实现以下流程
 *    1. 分配 A(k×m), B(k×n), C(m×n)。失败则 SKIP。
 *    2. tsmm_init_random 填充 A, B（固定种子，跨实现可复现）。
 *    3. 对自己的 tsmm_naive：
 *       a. warmup: 清零 C → 调用 tsmm_naive，重复 DEFAULT_NWARMUP 次。
 *       b. timed:  清零 C → 计时 → 调用 tsmm_naive → 停止计时，重复 nruns 次。
 *       c. 取 min wall time（代表最佳 cache 状态下的性能）。
 *       d. 用 tsmm_compute_perf 计算指标。
 *       e. 调用 tsmm_print_perf 输出（文本模式）或按 CSV 格式打印。
 *    4. 对 BLAS（#ifdef TSMM_USE_BLAS）：
 *       a. 同样 warmup + timed 流程。
 *       b. 注意 cblas_dgemm 的参数：
 *          row-major: cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
 *                                 m, n, k, 1.0, A, m, B, n, 0.0, C, n)
 *          col-major: cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
 *                                 m, n, k, 1.0, A, k, B, k, 0.0, C, m)
 *       c. 同样计算并输出 perf 指标。
 *    5. 释放所有矩阵。
 * ========================================================================== */
static void bench_one(const tsmm_shape_t *shape, tsmm_layout_t layout,
                      const bench_opts_t *opts)
{
    int m = shape->m, n = shape->n, k = shape->k;
    const char *lname = (layout == TSMM_ROW_MAJOR) ? "ROW_MAJOR" : "COL_MAJOR";

    /* ---------- 1. 分配矩阵 ---------- */
    double *A = tsmm_alloc_matrix(k, m);
    if (A == NULL) {
        printf("SKIP %s [%s] — A alloc failed\n", shape->name, lname);
        return;
    }
    double *B = tsmm_alloc_matrix(k, n);
    if (B == NULL) {
        printf("SKIP %s [%s] — B alloc failed\n", shape->name, lname);
        tsmm_free_matrix(A);
        return;
    }
    double *C = tsmm_alloc_matrix(m, n);
    if (C == NULL) {
        printf("SKIP %s [%s] — C alloc failed\n", shape->name, lname);
        tsmm_free_matrix(A);
        tsmm_free_matrix(B);
        return;
    }

    /* ---------- 2. 初始化数据 ---------- */
    tsmm_init_random(A, k, m);
    tsmm_init_random(B, k, n);

    /* ---------- 3. Warmup（稳定 cache / CPU 频率） ---------- */
    tsmm_init_zero(C, m, n);
    for (int i = 0; i < DEFAULT_NWARMUP; i++) {
        tsmm_naive(layout, m, n, k, A, B, C);
    }

    /* ---------- 4. Timed runs — 取 min wall time ---------- */
    double min_time_s = 1e100;
    for (int i = 0; i < opts->nruns; i++) {
        tsmm_init_zero(C, m, n);          /* 在计时之外清零 */
        double t1 = tsmm_wall_time();
        tsmm_naive(layout, m, n, k, A, B, C);
        double t2 = tsmm_wall_time();
        double elapsed = t2 - t1;
        if (elapsed < min_time_s) min_time_s = elapsed;
    }

    /* ---------- 5. 计算 & 输出性能 ---------- */
    tsmm_perf_t perf = tsmm_compute_perf(m, n, k, min_time_s * 1000.0,
                                         opts->num_threads);

    if (opts->csv_output) {
        printf("tsmm_naive,%s,%d,%d,%d,%s,%.6f,%.3f,%.3f,%.3f,%.3e,%.3e,%.3e,%d\n",
               shape->name, m, n, k, lname,
               perf.wall_time_ms, perf.gflops,
               perf.mem_bandwidth_gbs, perf.flops_per_byte,
               perf.total_flops, perf.bytes_read, perf.bytes_written,
               perf.num_threads);
    } else {
        printf("\n--- %s [%s] ---\n", shape->name, lname);
        tsmm_print_perf(&perf);
    }

    /* ---------- 6. 清理 ---------- */
    tsmm_free_matrix(A);
    tsmm_free_matrix(B);
    tsmm_free_matrix(C);
}

/* ==========================================================================
 *  命令行解析（简单手工解析，不依赖 getopt）
 * ========================================================================== */
static void parse_args(int argc, char **argv, bench_opts_t *opts)
{
    opts->csv_output   = 0;
    opts->nruns        = DEFAULT_NRUNS;
    opts->shape_index  = 0;
    opts->layout_opt   = 2;
    opts->no_blas      = 0;
    opts->num_threads  = 1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--csv")) {
            opts->csv_output = 1;
        } else if (!strcmp(argv[i], "--runs") && i + 1 < argc) {
            opts->nruns = atoi(argv[++i]);
            if (opts->nruns < 1) opts->nruns = 1;
        } else if (!strcmp(argv[i], "--shape") && i + 1 < argc) {
            opts->shape_index = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--layout") && i + 1 < argc) {
            opts->layout_opt = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--no-blas")) {
            opts->no_blas = 1;
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            opts->num_threads = atoi(argv[++i]);
            if (opts->num_threads < 1) opts->num_threads = 1;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("Usage: ./benchmark [options]\n");
            printf("  --csv          CSV output\n");
            printf("  --runs N       timed runs (default %d)\n", DEFAULT_NRUNS);
            printf("  --shape I      shape 1-%d, 0=all\n", TSMM_NUM_SHAPES);
            printf("  --layout L     0=row, 1=col, 2=both (default)\n");
            printf("  --no-blas      skip BLAS comparison\n");
            printf("  --threads N    thread count (default 1)\n");
            exit(0);
        }
    }
}
