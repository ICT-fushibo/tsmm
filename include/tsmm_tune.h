/**
 * tsmm_tune.h — Per-shape optimal kernel & tile size selection
 *
 * After running tile grid-search (scripts/run_tune_all.sh),
 * fill in the best (kernel, Ti, Tj, Tk) for each (shape, threads) combo.
 *
 * Currently populated with safe defaults (S3 3D at 48t, Ti=64 Tj=256 Tk=256).
 * Replace with actual tuning results.
 *
 * kernel names: omp_s3_3d | omp_s3_mn | omp_s4_3d | avx512_s5
 */

#ifndef TSMM_TUNE_H
#define TSMM_TUNE_H

#include "tsmm.h"

/* one entry per (shape, thread count) */
typedef struct {
    const char *kernel;   /* kernel name string */
    int Ti, Tj, Tk;       /* optimal tile sizes (Tk unused for MN kernels) */
} tsmm_tune_entry_t;

/* --------------------------------------------------------------------------
 * Tuning table — populated with DEFAULT values.
 * TODO: after run_tune_all.sh completes, replace with actual best combinations.
 * -------------------------------------------------------------------------- */

/* --- 24 threads --- */
static const tsmm_tune_entry_t tune_24t[TSMM_NUM_SHAPES] = {
    /* S1 */ {"omp_s3_3d",  64, 256, 256},
    /* S2 */ {"omp_s3_3d",  64, 256, 256},
    /* S3 */ {"omp_s3_3d",  64, 256, 256},
    /* S4 */ {"omp_s3_3d",  64, 256, 256},
    /* S5 */ {"omp_s3_3d",  64, 256, 256},
    /* S6 */ {"omp_s4_3d",  64, 256, 256},  // S6 needs k-parallel
    /* S7 */ {"omp_s3_3d",  64, 256, 256},
    /* S8 */ {"omp_s3_3d",  64, 256, 256},
};

/* --- 48 threads --- */
static const tsmm_tune_entry_t tune_48t[TSMM_NUM_SHAPES] = {
    /* S1 */ {"omp_s3_3d",  64, 256, 256},
    /* S2 */ {"omp_s3_3d",  64, 256, 256},
    /* S3 */ {"omp_s3_3d",  64, 256, 256},
    /* S4 */ {"omp_s3_3d",  64, 256, 256},
    /* S5 */ {"omp_s3_3d",  64, 256, 256},
    /* S6 */ {"omp_s4_3d",  64, 256, 256},
    /* S7 */ {"omp_s3_3d",  64, 256, 256},
    /* S8 */ {"omp_s3_3d",  64, 256, 256},
};

/* --- 96 threads --- */
static const tsmm_tune_entry_t tune_96t[TSMM_NUM_SHAPES] = {
    /* S1 */ {"omp_s3_3d",  64, 256, 256},
    /* S2 */ {"omp_s3_3d",  64, 256, 256},
    /* S3 */ {"omp_s3_3d",  64, 256, 256},
    /* S4 */ {"omp_s3_3d",  64, 256, 256},
    /* S5 */ {"omp_s3_3d",  64, 256, 256},
    /* S6 */ {"omp_s4_3d",  64, 256, 256},
    /* S7 */ {"omp_s3_3d",  64, 256, 256},
    /* S8 */ {"omp_s3_3d",  64, 256, 256},
};

/* --------------------------------------------------------------------------
 * tsmm_tune_lookup — return the best (kernel, Ti, Tj, Tk) for a shape + threads
 *
 * Falls back to the 48t table if thread count doesn't match exactly.
 * -------------------------------------------------------------------------- */
static inline tsmm_tune_entry_t tsmm_tune_lookup(int shape_idx,
                                                  int num_threads)
{
    const tsmm_tune_entry_t *table;
    switch (num_threads) {
        case 24: table = tune_24t; break;
        case 48: table = tune_48t; break;
        case 96: table = tune_96t; break;
        default: table = tune_48t; break;  /* fallback */
    }
    return table[shape_idx];
}

/* --------------------------------------------------------------------------
 * tsmm_optimal — dispatch to the best kernel with optimal tiles
 *
 * Single entry point that auto-selects based on (shape, threads).
 * Returns 0 on success.
 * -------------------------------------------------------------------------- */
static inline int tsmm_optimal(int shape_idx,
                                tsmm_layout_t layout,
                                int m, int n, int k,
                                const double *A, const double *B, double *C,
                                int num_threads)
{
    tsmm_tune_entry_t e = tsmm_tune_lookup(shape_idx, num_threads);

    if (!strcmp(e.kernel, "omp_s3_3d")) {
        tsmm_tiled_omp_s3(layout, m, n, k, A, B, C,
                          e.Ti, e.Tj, e.Tk, num_threads);
    } else if (!strcmp(e.kernel, "omp_s3_mn")) {
        tsmm_tiled_omp_s3_mn(layout, m, n, k, A, B, C,
                             e.Ti, e.Tj, num_threads);
    } else if (!strcmp(e.kernel, "omp_s4_3d")) {
        tsmm_tiled_omp_s4(layout, m, n, k, A, B, C,
                          e.Ti, e.Tj, e.Tk, num_threads);
    } else if (!strcmp(e.kernel, "avx512_s5")) {
        (void)num_threads;
        tsmm_avx512_s5(layout, m, n, k, A, B, C, e.Ti, e.Tj, e.Tk);
    } else {
        /* fallback: S3 3D with defaults */
        tsmm_tiled_omp_s3(layout, m, n, k, A, B, C,
                          e.Ti, e.Tj, e.Tk, num_threads);
    }
    return 0;  /* ignore unused shape_idx warning */
}

#endif /* TSMM_TUNE_H */
