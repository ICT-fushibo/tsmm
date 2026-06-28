/**
 * tsmm_tune.h — Per-shape optimal kernel & tile size selection
 *
 * Theory-guided estimates based on S2/S3/S4 benchmark data.
 * Will be refined after grid-search tuning completes.
 *
 * Selection logic:
 *   S1 (large C, small k): S3_3d for ≥48t (private buf reduces false sharing)
 *   S2 (tiny C, long k):   S4_3d for ≤16t (k-parallel), S3_3d fallback
 *   S3 (medium, tiny k):   S2_3d (collapse(2), lowest overhead for many tiles)
 *   S4 (cube, fit L3):     S4_3d at 1t, S2_3d otherwise
 *   S5 (medium n, tiny k): S2_3d (many tiles, low overhead)
 *   S6 (tiny C, massive k):S4_3d (k-parallel essential: 2372 pk iters)
 *   S7 (small, tiny k):    S2_3d
 *   S8 (massive C, small k):S2_mn for ≥48t (no k block, lowest overhead)
 */

#ifndef TSMM_TUNE_H
#define TSMM_TUNE_H

#include "tsmm.h"
#include <string.h>

typedef struct {
    const char *kernel;
    int Ti, Tj, Tk;
} tsmm_tune_entry_t;

/* --- Best per (shape, thread) — theory-guided prior --- */

/* Low threads (≤16): minimize overhead, maximize tile count */
static const tsmm_tune_entry_t tune_lo[TSMM_NUM_SHAPES] = {
    /* S1 */ {"omp_s2_3d",  64, 256, 128},  // k=128→full k, 38750 tiles
    /* S2 */ {"omp_s4_3d",   8,  16, 256},  // 1 tile, k-parallel essential
    /* S3 */ {"omp_s2_3d",  32, 256,  16},  // full m as Ti, max parallelism
    /* S4 */ {"omp_s4_3d",  72, 144, 144},  // half in i, full in j,k
    /* S5 */ {"omp_s2_3d",  16, 256,  16},  // full m as Ti
    /* S6 */ {"omp_s4_3d",   4,  64, 512},  // 1 tile, k-parallel, larger Tk
    /* S7 */ {"omp_s2_3d",  64, 193,  11},  // full n as Tj
    /* S8 */ {"omp_s2_mn",  40, 256,  40},  // full m as Ti, MN-only
};

/* Mid threads (24-48): balance overhead vs parallelism */
static const tsmm_tune_entry_t tune_mid[TSMM_NUM_SHAPES] = {
    /* S1 */ {"omp_s3_3d",  32, 256, 128},  // smaller Ti → tile fits L1
    /* S2 */ {"omp_s4_3d",   8,  16, 256},
    /* S3 */ {"omp_s2_3d",  32, 256,  16},
    /* S4 */ {"omp_s2_3d",  72, 144, 144},
    /* S5 */ {"omp_s2_3d",  16, 256,  16},
    /* S6 */ {"omp_s4_3d",   4,  64, 256},
    /* S7 */ {"omp_s2_3d",  64, 193,  11},
    /* S8 */ {"omp_s2_mn",  40, 256,  40},
};

/* High threads (64-96): private buffer wins for large shapes */
static const tsmm_tune_entry_t tune_hi[TSMM_NUM_SHAPES] = {
    /* S1 */ {"omp_s3_3d",  32, 256, 128},  // 746 GFLOPS at 96t
    /* S2 */ {"omp_s3_3d",   8,  16, 256},  // S4 overhead > benefit at 96t
    /* S3 */ {"omp_s2_3d",  32, 256,  16},  // S2 > S3 for many tiles
    /* S4 */ {"omp_s2_3d",  72, 144, 144},
    /* S5 */ {"omp_s2_3d",  16, 256,  16},
    /* S6 */ {"omp_s4_3d",   4,  64, 256},
    /* S7 */ {"omp_s2_3d",  64, 193,  11},
    /* S8 */ {"omp_s2_mn",  40, 256,  40},  // 549 GFLOPS at 96t, best
};

static inline tsmm_tune_entry_t tsmm_tune_lookup(int shape_idx,
                                                  int num_threads)
{
    const tsmm_tune_entry_t *table;
    if      (num_threads <= 16) table = tune_lo;
    else if (num_threads <= 48) table = tune_mid;
    else                         table = tune_hi;
    return table[shape_idx];
}

static inline int tsmm_optimal(int shape_idx,
                                tsmm_layout_t layout,
                                int m, int n, int k,
                                const double *A, const double *B, double *C,
                                int num_threads)
{
    tsmm_tune_entry_t e = tsmm_tune_lookup(shape_idx, num_threads);

    if      (!strcmp(e.kernel, "omp_s2_3d"))
        tsmm_tiled_omp_s2(layout, m, n, k, A, B, C, e.Ti, e.Tj, e.Tk, num_threads);
    else if (!strcmp(e.kernel, "omp_s2_mn"))
        tsmm_tiled_omp_s2_mn(layout, m, n, k, A, B, C, e.Ti, e.Tj, num_threads);
    else if (!strcmp(e.kernel, "omp_s3_3d"))
        tsmm_tiled_omp_s3(layout, m, n, k, A, B, C, e.Ti, e.Tj, e.Tk, num_threads);
    else if (!strcmp(e.kernel, "omp_s3_mn"))
        tsmm_tiled_omp_s3_mn(layout, m, n, k, A, B, C, e.Ti, e.Tj, num_threads);
    else if (!strcmp(e.kernel, "omp_s4_3d"))
        tsmm_tiled_omp_s4(layout, m, n, k, A, B, C, e.Ti, e.Tj, e.Tk, num_threads);
    else if (!strcmp(e.kernel, "avx512_s5"))
        { (void)num_threads; tsmm_avx512_s5(layout, m, n, k, A, B, C, e.Ti, e.Tj, e.Tk); }
    else if (!strcmp(e.kernel, "avx512_s5_omp"))
        tsmm_avx512_s5_omp(layout, m, n, k, A, B, C, e.Ti, e.Tj, e.Tk, num_threads);
    else  /* fallback */
        tsmm_tiled_omp_s3(layout, m, n, k, A, B, C, e.Ti, e.Tj, e.Tk, num_threads);
    return 0;
}

#endif /* TSMM_TUNE_H */
