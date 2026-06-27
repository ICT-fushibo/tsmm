/**
 * tests/benchmark.c — TSMM 性能对比。
 *
 * 每个 kernel 独立计时，通过 --kernel 选择运行哪个。
 * 输出 CSV 或可读文本，含 GFLOPS、带宽、算术强度。
 *
 * 用法：
 *   ./build/benchmark --kernel all                      # 全部 kernel
 *   ./build/benchmark --kernel naive                    # 只测 naive
 *   ./build/benchmark --kernel tiled_3d --tile-m 128    # 只测 tiled_3d, 调参
 *   ./build/benchmark --kernel tiled_mn                 # 只测 tiled_mn
 *   ./build/benchmark --csv --shape 4 --layout 0        # CSV, 单 shape
 */

#include "tsmm.h"
#include "tsmm_tune.h"

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
#define DEFAULT_Ti  64
#define DEFAULT_Tj  256
#define DEFAULT_Tk  256

/* Kernel 选择位掩码 */
#define KERNEL_NAIVE       (1<<0)
#define KERNEL_TILED_3D    (1<<1)
#define KERNEL_TILED_MN    (1<<2)
#define KERNEL_TILED_OMP_3D (1<<3)
#define KERNEL_TILED_OMP_MN (1<<4)
#define KERNEL_OMP_S2_3D   (1<<5)
#define KERNEL_OMP_S2_MN   (1<<6)
#define KERNEL_OMP_S3_3D   (1<<7)
#define KERNEL_OMP_S3_MN   (1<<8)
#define KERNEL_OMP_S4_3D   (1<<9)
#define KERNEL_OMP_S4_MN   (1<<10)
#define KERNEL_AVX512_S5   (1<<11)
#define KERNEL_OPTIMAL     (1<<12)
#define KERNEL_ALL         (KERNEL_NAIVE | KERNEL_TILED_3D | KERNEL_TILED_MN | \
                            KERNEL_TILED_OMP_3D | KERNEL_TILED_OMP_MN | \
                            KERNEL_OMP_S2_3D | KERNEL_OMP_S2_MN | \
                            KERNEL_OMP_S3_3D | KERNEL_OMP_S3_MN | \
                            KERNEL_OMP_S4_3D | KERNEL_OMP_S4_MN | \
                            KERNEL_AVX512_S5 | KERNEL_OPTIMAL)

/* ---------- 命令行选项 ---------- */
typedef struct {
    int csv_output;
    int nruns;
    int shape_index;
    int layout_opt;
    int no_blas;
    int kernel_mask;
    int num_threads;
    int Ti, Tj, Tk;
} bench_opts_t;

/* ---------- 前向声明 ---------- */
static void parse_args(int argc, char **argv, bench_opts_t *opts);
static void bench_one(const tsmm_shape_t *shape, tsmm_layout_t layout,
                      const bench_opts_t *opts);

/* ---------- 辅助宏 ---------- */
#define TIME_KERNEL(init_zero, call_expr)                       \
    do {                                                        \
        double _min_t = 1e100;                                  \
        for (int _i = 0; _i < opts->nruns; _i++) {              \
            if (init_zero) tsmm_init_zero(C, m, n);             \
            double _t1 = tsmm_wall_time();                      \
            call_expr;                                          \
            double _t2 = tsmm_wall_time();                      \
            double _elapsed = _t2 - _t1;                        \
            if (_elapsed < _min_t) _min_t = _elapsed;           \
        }                                                       \
        min_time_s = _min_t;                                    \
    } while (0)

static void emit_csv(const char *kernel, const tsmm_shape_t *shape,
                     const char *lname, const tsmm_perf_t *p)
{
    printf("%s,%s,%d,%d,%d,%s,%.6f,%.3f,%.3f,%.3f,%.3e,%.3e,%.3e,%d\n",
           kernel, shape->name, p->m, p->n, p->k, lname,
           p->wall_time_ms, p->gflops,
           p->mem_bandwidth_gbs, p->flops_per_byte,
           p->total_flops, p->bytes_read, p->bytes_written,
           p->num_threads);
}

static void bench_and_report(const char *kernel_name,
                             const tsmm_shape_t *shape,
                             const tsmm_layout_t layout,
                             double min_time_s,
                             const bench_opts_t *opts,
                             const tsmm_perf_t *p_baseline)
{
    int m = shape->m, n = shape->n, k = shape->k;
    const char *lname = (layout == TSMM_ROW_MAJOR) ? "ROW_MAJOR" : "COL_MAJOR";

    tsmm_perf_t p = tsmm_compute_perf(m, n, k, min_time_s * 1000.0,
                                      opts->num_threads);

    if (opts->csv_output) {
        emit_csv(kernel_name, shape, lname, &p);
    } else {
        printf("  --- %s ---\n", kernel_name);
        tsmm_print_perf(&p);
        if (p_baseline && p_baseline->wall_time_ms > 0.0) {
            printf("  Speedup vs naive: %.2fx\n",
                   p_baseline->wall_time_ms / p.wall_time_ms);
        }
    }
}

/* --- OMP kernel timing helpers --- */
static double time_tiled_omp_3d(tsmm_layout_t layout,
                                 int m, int n, int k,
                                 const double *A, const double *B,
                                 double *C, int nruns,
                                 int Ti, int Tj, int Tk, int nthreads)
{
    double min_time = 1e100;
    for (int i = 0; i < nruns; i++) {
        tsmm_init_zero(C, m, n);
        double t1 = tsmm_wall_time();
        tsmm_tiled_omp(layout, m, n, k, A, B, C, Ti, Tj, Tk, nthreads);
        double t2 = tsmm_wall_time();
        double elapsed = t2 - t1;
        if (elapsed < min_time) min_time = elapsed;
    }
    return min_time;
}

static double time_tiled_omp_mn(tsmm_layout_t layout,
                                 int m, int n, int k,
                                 const double *A, const double *B,
                                 double *C, int nruns,
                                 int Ti, int Tj, int nthreads)
{
    double min_time = 1e100;
    for (int i = 0; i < nruns; i++) {
        tsmm_init_zero(C, m, n);
        double t1 = tsmm_wall_time();
        tsmm_tiled_omp_mn(layout, m, n, k, A, B, C, Ti, Tj, nthreads);
        double t2 = tsmm_wall_time();
        double elapsed = t2 - t1;
        if (elapsed < min_time) min_time = elapsed;
    }
    return min_time;
}

/* ========================================================================== */

int main(int argc, char **argv)
{
    bench_opts_t opts;
    parse_args(argc, argv, &opts);

#ifdef _OPENMP
    omp_set_num_threads(opts.num_threads);
#endif

    printf("# TSMM Benchmark\n");
    printf("# Kernel mask: %d (naive=%d tiled_3d=%d tiled_mn=%d)\n",
           opts.kernel_mask,
           !!(opts.kernel_mask & KERNEL_NAIVE),
           !!(opts.kernel_mask & KERNEL_TILED_3D),
           !!(opts.kernel_mask & KERNEL_TILED_MN));
    printf("# Threads: %d   Runs: %d   Warmup: %d\n",
           opts.num_threads, opts.nruns, DEFAULT_NWARMUP);
    printf("# Tile: Ti=%d Tj=%d Tk=%d\n", opts.Ti, opts.Tj, opts.Tk);
    printf("#\n");

    tsmm_layout_t layouts[2] = { TSMM_ROW_MAJOR, TSMM_COL_MAJOR };
    int n_layouts = (opts.layout_opt == 2) ? 2 : 1;
    int loffset   = (opts.layout_opt == 1) ? 1 : 0;
    int istart = (opts.shape_index == 0) ? 0 : (opts.shape_index - 1);
    int iend   = (opts.shape_index == 0) ? TSMM_NUM_SHAPES : opts.shape_index;

    if (opts.csv_output) {
        printf("kernel,shape,m,n,k,layout,wall_ms,gflops,"
               "mem_bw_gbs,ai,total_flops,bytes_read,bytes_written,threads\n");
    }

    for (int si = istart; si < iend; ++si)
        for (int li = 0; li < n_layouts; ++li)
            bench_one(&tsmm_shapes[si], layouts[loffset + li], &opts);

    return 0;
}

/* ========================================================================== */
static void bench_one(const tsmm_shape_t *shape, tsmm_layout_t layout,
                      const bench_opts_t *opts)
{
    int m = shape->m, n = shape->n, k = shape->k;
    const char *lname = (layout == TSMM_ROW_MAJOR) ? "ROW_MAJOR" : "COL_MAJOR";

    double *A = tsmm_alloc_matrix(k, m);
    double *B = tsmm_alloc_matrix(k, n);
    double *C = tsmm_alloc_matrix(m, n);
    if (!A || !B || !C) {
        printf("SKIP %s [%s] — alloc failed\n", shape->name, lname);
        tsmm_free_matrix(A); tsmm_free_matrix(B); tsmm_free_matrix(C);
        return;
    }
    tsmm_init_random(A, k, m);
    tsmm_init_random(B, k, n);

    if (!opts->csv_output)
        printf("\n--- %s [%s] ---\n", shape->name, lname);

    /* --- warmup with naive --- */
    tsmm_init_zero(C, m, n);
    for (int i = 0; i < DEFAULT_NWARMUP; i++)
        tsmm_naive(layout, m, n, k, A, B, C);

    /* --- timing & reporting --- */
    double min_time_s;
    tsmm_perf_t p_naive_baseline;
    int have_baseline = 0;

    /* 1. Naive */
    if (opts->kernel_mask & KERNEL_NAIVE) {
        TIME_KERNEL(1, tsmm_naive(layout, m, n, k, A, B, C));
        bench_and_report("naive", shape, layout, min_time_s, opts, NULL);
        p_naive_baseline = tsmm_compute_perf(m, n, k, min_time_s * 1000.0,
                                              opts->num_threads);
        have_baseline = 1;
    }

    /* 2. Tiled 3D */
    if (opts->kernel_mask & KERNEL_TILED_3D) {
        TIME_KERNEL(1, tsmm_tiled(layout, m, n, k, A, B, C,
                                  opts->Ti, opts->Tj, opts->Tk));
        bench_and_report("tiled_3d", shape, layout, min_time_s, opts,
                         have_baseline ? &p_naive_baseline : NULL);
    }

    /* 3. Tiled MN-only */
    if (opts->kernel_mask & KERNEL_TILED_MN) {
        TIME_KERNEL(1, tsmm_tiled_mn(layout, m, n, k, A, B, C,
                                     opts->Ti, opts->Tj));
        bench_and_report("tiled_mn", shape, layout, min_time_s, opts,
                         have_baseline ? &p_naive_baseline : NULL);
    }

    /* 4. Tiled OMP 3D */
    if (opts->kernel_mask & KERNEL_TILED_OMP_3D) {
        double t = time_tiled_omp_3d(layout, m, n, k, A, B, C,
                                     opts->nruns,
                                     opts->Ti, opts->Tj, opts->Tk,
                                     opts->num_threads);
        tsmm_perf_t p = tsmm_compute_perf(m, n, k, t * 1000.0,
                                           opts->num_threads);
        p.num_threads = opts->num_threads;
        if (opts->csv_output)
            emit_csv("tiled_omp_3d", shape, lname, &p);
        else {
            printf("  --- tiled_omp_3d (nt=%d Ti=%d Tj=%d Tk=%d) ---\n",
                   opts->num_threads, opts->Ti, opts->Tj, opts->Tk);
            tsmm_print_perf(&p);
            if (have_baseline)
                printf("  Speedup vs naive: %.2fx\n",
                       p_naive_baseline.wall_time_ms / p.wall_time_ms);
        }
    }

    /* 5. Tiled OMP MN-only (Step 1) */
    if (opts->kernel_mask & KERNEL_TILED_OMP_MN) {
        double t = time_tiled_omp_mn(layout, m, n, k, A, B, C,
                                     opts->nruns,
                                     opts->Ti, opts->Tj,
                                     opts->num_threads);
        tsmm_perf_t p = tsmm_compute_perf(m, n, k, t * 1000.0,
                                           opts->num_threads);
        p.num_threads = opts->num_threads;
        if (opts->csv_output)
            emit_csv("tiled_omp_mn", shape, lname, &p);
        else {
            printf("  --- tiled_omp_mn (Step1, nt=%d Ti=%d Tj=%d) ---\n",
                   opts->num_threads, opts->Ti, opts->Tj);
            tsmm_print_perf(&p);
            if (have_baseline)
                printf("  Speedup vs naive: %.2fx\n",
                       p_naive_baseline.wall_time_ms / p.wall_time_ms);
        }
    }

    /* 6. Tiled OMP Step2 3D (collapse) */
    if (opts->kernel_mask & KERNEL_OMP_S2_3D) {
        TIME_KERNEL(1, tsmm_tiled_omp_s2(layout, m, n, k, A, B, C,
                                         opts->Ti, opts->Tj, opts->Tk,
                                         opts->num_threads));
        bench_and_report("omp_s2_3d", shape, layout, min_time_s, opts,
                         have_baseline ? &p_naive_baseline : NULL);
    }

    /* 7. Tiled OMP Step2 MN-only (collapse) */
    if (opts->kernel_mask & KERNEL_OMP_S2_MN) {
        TIME_KERNEL(1, tsmm_tiled_omp_s2_mn(layout, m, n, k, A, B, C,
                                            opts->Ti, opts->Tj,
                                            opts->num_threads));
        bench_and_report("omp_s2_mn", shape, layout, min_time_s, opts,
                         have_baseline ? &p_naive_baseline : NULL);
    }

    /* 8. Tiled OMP Step3 3D (private buffer) */
    if (opts->kernel_mask & KERNEL_OMP_S3_3D) {
        TIME_KERNEL(1, tsmm_tiled_omp_s3(layout, m, n, k, A, B, C,
                                         opts->Ti, opts->Tj, opts->Tk,
                                         opts->num_threads));
        bench_and_report("omp_s3_3d", shape, layout, min_time_s, opts,
                         have_baseline ? &p_naive_baseline : NULL);
    }

    /* 9. Tiled OMP Step3 MN-only (private buffer) */
    if (opts->kernel_mask & KERNEL_OMP_S3_MN) {
        TIME_KERNEL(1, tsmm_tiled_omp_s3_mn(layout, m, n, k, A, B, C,
                                            opts->Ti, opts->Tj,
                                            opts->num_threads));
        bench_and_report("omp_s3_mn", shape, layout, min_time_s, opts,
                         have_baseline ? &p_naive_baseline : NULL);
    }

    /* 10. Tiled OMP Step4 3D (k-parallel + reduction) */
    if (opts->kernel_mask & KERNEL_OMP_S4_3D) {
        TIME_KERNEL(1, tsmm_tiled_omp_s4(layout, m, n, k, A, B, C,
                                         opts->Ti, opts->Tj, opts->Tk,
                                         opts->num_threads));
        bench_and_report("omp_s4_3d", shape, layout, min_time_s, opts,
                         have_baseline ? &p_naive_baseline : NULL);
    }

    /* 11. Tiled OMP Step4 MN (fallback to S3) */
    if (opts->kernel_mask & KERNEL_OMP_S4_MN) {
        TIME_KERNEL(1, tsmm_tiled_omp_s4_mn(layout, m, n, k, A, B, C,
                                            opts->Ti, opts->Tj,
                                            opts->num_threads));
        bench_and_report("omp_s4_mn", shape, layout, min_time_s, opts,
                         have_baseline ? &p_naive_baseline : NULL);
    }

    /* 12. AVX-512 Stage 5 (serial) */
    if (opts->kernel_mask & KERNEL_AVX512_S5) {
        TIME_KERNEL(1, tsmm_avx512_s5(layout, m, n, k, A, B, C,
                                       opts->Ti, opts->Tj, opts->Tk));
        bench_and_report("avx512_s5", shape, layout, min_time_s, opts,
                         have_baseline ? &p_naive_baseline : NULL);
    }

    /* 13. Optimal (auto-select best kernel + tiles from tsmm_tune.h) */
    if (opts->kernel_mask & KERNEL_OPTIMAL) {
        int si = (int)(shape - tsmm_shapes);  /* shape index 0..7 */
        TIME_KERNEL(1, tsmm_optimal(si, layout, m, n, k, A, B, C,
                                     opts->num_threads));
        bench_and_report("optimal", shape, layout, min_time_s, opts,
                         have_baseline ? &p_naive_baseline : NULL);
    }

    tsmm_free_matrix(A);
    tsmm_free_matrix(B);
    tsmm_free_matrix(C);
}

/* ========================================================================== */
static void parse_args(int argc, char **argv, bench_opts_t *opts)
{
    opts->csv_output   = 0;
    opts->nruns        = DEFAULT_NRUNS;
    opts->shape_index  = 0;
    opts->layout_opt   = 2;
    opts->no_blas      = 0;
    opts->kernel_mask  = KERNEL_ALL;
    opts->num_threads  = 1;
    opts->Ti = DEFAULT_Ti;
    opts->Tj = DEFAULT_Tj;
    opts->Tk = DEFAULT_Tk;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--csv"))
            opts->csv_output = 1;
        else if (!strcmp(argv[i], "--runs") && i + 1 < argc)
            { opts->nruns = atoi(argv[++i]); if (opts->nruns < 1) opts->nruns = 1; }
        else if (!strcmp(argv[i], "--shape") && i + 1 < argc)
            opts->shape_index = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layout") && i + 1 < argc)
            opts->layout_opt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-blas"))
            opts->no_blas = 1;
        else if (!strcmp(argv[i], "--kernel") && i + 1 < argc) {
            char *k = argv[++i];
            if      (!strcmp(k, "naive"))         opts->kernel_mask = KERNEL_NAIVE;
            else if (!strcmp(k, "tiled_3d"))      opts->kernel_mask = KERNEL_TILED_3D;
            else if (!strcmp(k, "tiled_mn"))      opts->kernel_mask = KERNEL_TILED_MN;
            else if (!strcmp(k, "tiled_omp_3d"))  opts->kernel_mask = KERNEL_TILED_OMP_3D;
            else if (!strcmp(k, "tiled_omp_mn"))    opts->kernel_mask = KERNEL_TILED_OMP_MN;
            else if (!strcmp(k, "omp_s2_3d"))      opts->kernel_mask = KERNEL_OMP_S2_3D;
            else if (!strcmp(k, "omp_s2_mn"))      opts->kernel_mask = KERNEL_OMP_S2_MN;
            else if (!strcmp(k, "omp_s3_3d"))      opts->kernel_mask = KERNEL_OMP_S3_3D;
            else if (!strcmp(k, "omp_s3_mn"))      opts->kernel_mask = KERNEL_OMP_S3_MN;
            else if (!strcmp(k, "omp_s4_3d"))      opts->kernel_mask = KERNEL_OMP_S4_3D;
            else if (!strcmp(k, "omp_s4_mn"))      opts->kernel_mask = KERNEL_OMP_S4_MN;
            else if (!strcmp(k, "avx512_s5"))      opts->kernel_mask = KERNEL_AVX512_S5;
            else if (!strcmp(k, "optimal"))        opts->kernel_mask = KERNEL_OPTIMAL;
            else if (!strcmp(k, "all"))            opts->kernel_mask = KERNEL_ALL;
            else if (!strcmp(k, "all_omp"))        opts->kernel_mask = KERNEL_TILED_OMP_3D | KERNEL_TILED_OMP_MN;
            else if (!strcmp(k, "all_omp_s2"))     opts->kernel_mask = KERNEL_OMP_S2_3D | KERNEL_OMP_S2_MN;
            else if (!strcmp(k, "all_omp_s3"))     opts->kernel_mask = KERNEL_OMP_S3_3D | KERNEL_OMP_S3_MN;
            else if (!strcmp(k, "all_omp_s4"))     opts->kernel_mask = KERNEL_OMP_S4_3D | KERNEL_OMP_S4_MN;
            else { fprintf(stderr, "Unknown kernel: %s\n", k); exit(1); }
        }
        else if (!strcmp(argv[i], "--no-tiled"))
            opts->kernel_mask = KERNEL_NAIVE;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            { opts->num_threads = atoi(argv[++i]); if (opts->num_threads < 1) opts->num_threads = 1; }
        else if (!strcmp(argv[i], "--tile-m") && i + 1 < argc)
            { opts->Ti = atoi(argv[++i]); if (opts->Ti < 1) opts->Ti = 1; }
        else if (!strcmp(argv[i], "--tile-n") && i + 1 < argc)
            { opts->Tj = atoi(argv[++i]); if (opts->Tj < 1) opts->Tj = 1; }
        else if (!strcmp(argv[i], "--tile-k") && i + 1 < argc)
            { opts->Tk = atoi(argv[++i]); if (opts->Tk < 1) opts->Tk = 1; }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("Usage: ./benchmark [options]\n");
            printf("  --kernel naive|tiled_3d|tiled_mn|all   (default: all)\n");
            printf("  --csv\n");
            printf("  --runs N           timed runs (default %d)\n", DEFAULT_NRUNS);
            printf("  --shape I          shape 1-%d, 0=all\n", TSMM_NUM_SHAPES);
            printf("  --layout L         0=row, 1=col, 2=both\n");
            printf("  --threads N        (default 1)\n");
            printf("  --tile-m N --tile-n N --tile-k N\n");
            printf("  --no-tiled         alias for --kernel naive\n");
            exit(0);
        }
    }
}
