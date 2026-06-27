# TSMM Optimization Project

## Project Description

Optimizing **TSMM (Transposed Matrix-Matrix Multiplication)**: `C = A^T * B`

- A: k×m, B: k×n, C: m×n — all dense, double precision (IEEE 754 binary64)
- 8 problem shapes (m, n, k):
  - S1: (4000, 160000, 128)
  - S2: (8, 16, 16000)
  - S3: (32, 16000, 16)
  - S4: (144, 144, 144)
  - S5: (16, 12344, 16)
  - S6: (4, 64, 606841)
  - S7: (442, 193, 11)
  - S8: (40, 1127228, 40)
- Comparison targets: OpenBLAS and Intel MKL (fair: same threads, same NUMA binding)
- Code written locally (Windows), compiled & run on HPC (BSCC-T6) via SLURM

## Target Machine (BSCC-T6)

- CPU: 2× Intel Xeon Platinum 9242 @ 2.30 GHz (max 3.80 GHz)
- Cores: 96 physical (48/socket), HT off, 4 NUMA nodes (24 cores each)
- Cache: L1d 32 KB, L2 1024 KB, L3 36608 KB (36 MB) per socket, cache line 64 B
- ISA: AVX-512F/CD/DQ/BW/VL/VNNI, 2× 512-bit FMA → 32 DP FLOPs/cycle/core
- Peak: 74.56 GFLOPS/core, 7157.76 GFLOPS/node
- Module system: `source /public5/soft/modules/module.sh`
- Compiler: `gcc/10.2.0`
- OpenBLAS: `openblas/0.3.17-ips18` → `/public5/soft/openblas/0.3.17/`
- MKL: `intel/2022.1` → `/public5/soft/oneAPI/2022.1/mkl/latest/`

## File Structure

```
tsmm/
├── CLAUDE.md                    ← this file
├── Makefile                     ← build system (gcc, -O3, -std=c11)
├── include/
│   └── tsmm.h                   ← public API: kernels, perf struct, shapes, utils
├── src/
│   ├── tsmm_naive.c             ← serial baseline
│   ├── tsmm_tiled.c             ← cache-tiled (3D and MN-only, 2 layouts)
│   ├── tsmm_tiled_omp.c         ← Stage 2 Step 1: OMP Ti×n strip
│   ├── tsmm_tiled_omp_s2.c      ← Stage 2 Step 2: OMP Ti×Tj + collapse(2)
│   ├── tsmm_tiled_omp_s3.c      ← Stage 2 Step 3: OMP + private buffer
│   ├── tsmm_tiled_omp_s4.c      ← Stage 2 Step 4: OMP pk-parallel + reduction
│   ├── tsmm_avx512_s5.c         ← Stage 5: AVX-512 8×16 micro-kernel
│   └── tsmm_utils.c             ← memory, timing, verification, perf computation
├── tests/
│   ├── test_correctness.c       ← correctness: 11 kernels × 7 shapes × 2 layouts
│   └── benchmark.c              ← performance: --kernel selects variant
├── scripts/
│   ├── check_env.sh             ← HPC environment probe
│   ├── run_correctness.sh       ← single-job correctness
│   ├── run_correctness_all.sh   ← batch submit correctness
│   ├── run_correctness_omp.sh   ← OMP multi-thread correctness
│   ├── run_correctness_omp_all.sh ← OMP all-kernel all-thread correctness
│   ├── run_benchmark.sh         ← single-job benchmark
│   ├── run_benchmark_all.sh     ← batch submit (serial kernels)
│   ├── run_benchmark_omp.sh     ← OMP benchmark with NUMA-aware thread sweep
│   ├── run_benchmark_omp_all.sh ← batch submit OMP benchmarks
│   ├── run_benchmark_omp_s2_all.sh
│   ├── run_benchmark_omp_s3_all.sh
│   ├── run_benchmark_omp_s4_all.sh
│   ├── run_profile.sh           ← single-job perf stat
│   ├── run_profile_all.sh       ← batch submit profiling
│   ├── run_profile_omp.sh       ← OMP profiling
│   ├── run_profile_omp_all.sh   ← batch submit OMP profiling
│   ├── run_profile_omp_s2_all.sh
│   ├── run_profile_omp_s3_all.sh
│   ├── run_profile_omp_s4_all.sh
│   ├── run_tune_tiles.sh        ← single (shape,kernel) tile grid-search
│   ├── run_tune_by_kernel.sh    ← full shape×thread tuning per kernel
│   ├── run_tune_all.sh          ← submit 3 kernel tuning jobs
│   └── run_slurm.sh             ← legacy combined script
└── logs/                        ← SLURM output (gitignored)
```

## Optimization Roadmap

| Stage | File | Strategy | Status |
|-------|------|----------|--------|
| 1 | `tsmm_tiled.c` | Serial cache tiling (3D + MN-only) | ✅ |
| 2-S1 | `tsmm_tiled_omp.c` | OMP Ti×n strip | ✅ |
| 2-S2 | `tsmm_tiled_omp_s2.c` | OMP Ti×Tj + collapse(2) | ✅ |
| 2-S3 | `tsmm_tiled_omp_s3.c` | OMP + private buffer | ✅ |
| 2-S4 | `tsmm_tiled_omp_s4.c` | OMP pk-parallel + reduction | ✅ |
| 5 | `tsmm_avx512_s5.c` | AVX-512 8×16 micro-kernel | 🔄 implementing |
| 6a-h | (future) | Per-shape specialization | 📋 planned |

## Key API (tsmm.h)

- `tsmm_layout_t` — `TSMM_ROW_MAJOR` or `TSMM_COL_MAJOR`
- `tsmm_perf_t` — wall/CPU time, GFLOPS, bandwidth, arithmetic intensity, cache ratios
- `tsmm_shape_t` — m, n, k, label; 8 pre-defined in `tsmm_shapes[]`
- Kernel functions (select via `--kernel` in benchmark):
  - `naive` — serial baseline
  - `tiled_3d` / `tiled_mn` — serial cache tiling
  - `omp_3d` / `omp_mn` — Stage 2 Step 1 (Ti×n strip)
  - `omp_s2_3d` / `omp_s2_mn` — Stage 2 Step 2 (Ti×Tj + collapse)
  - `omp_s3_3d` / `omp_s3_mn` — Stage 2 Step 3 (private buffer)
  - `omp_s4_3d` / `omp_s4_mn` — Stage 2 Step 4 (pk-parallel + reduction)
  - `avx512_s5` — Stage 5 (AVX-512 micro-kernel, serial)
- `tsmm_alloc_matrix` returns 64-byte aligned memory
- BIN_SUFFIX isolates binaries (e.g. `_omp_s4`), but all kernels are linked into every binary

## Build & Run

```bash
make                    # build all (no BLAS)
make test               # build & run correctness
make blas BLAS=openblas # build with BLAS comparison

# Batch submission (pre-builds once, then submits parallel jobs):
bash scripts/run_correctness_all.sh    # 1 job
bash scripts/run_benchmark_all.sh      # 6 jobs (3 kernels × 2 layouts)
bash scripts/run_profile_all.sh        # 6 jobs (perf stat: cache miss, CPI, ...)

# Manual single-job:
sbatch scripts/run_correctness.sh
KERNEL=naive LAYOUT=row sbatch scripts/run_benchmark.sh
KERNEL=tiled_3d sbatch scripts/run_profile.sh
```

Outputs go to `logs/` with descriptive tags (e.g. `logs/bench_naiv_row_12345.out`).

## Instructions for Claude

### 1. Independent benchmarks per optimization

Every optimization strategy gets its OWN benchmark entry point. When a new optimization is added (e.g., tiling, SIMD, multi-threading, prefetch), create:
- A dedicated kernel function in its own source file under `src/`
- A declaration in `include/tsmm.h`
- An entry in `tests/benchmark.c` controllable via `--kernel <name>`
- If the profiling flow needs to differ (e.g., different events, different warmup), create a separate benchmark executable or support it via CLI flags

Do NOT merge different optimization strategies into a single hard-coded measurement loop.

### 2. Three separate submission scripts

Always maintain exactly three submission categories:
| Script | Purpose | Output prefix |
|--------|---------|---------------|
| `run_correctness.sh` | Correctness validation | `check_*` |
| `run_benchmark.sh` | Performance (GFLOPS, BW, AI) | `bench_*` |
| `run_profile.sh` | Hardware counters (cache miss, CPI) | `prof_*` |

Each has a `_all.sh` batch wrapper that pre-builds once and submits jobs with `NO_BUILD=1`. The batch wrappers must `make clean && make` before submitting. Output files use descriptive tags (kernel name + layout), never just job ID.

### 3. Full implementation — Claude writes complete code

From Stage 5 onward, Claude writes the **full implementation** of all kernel logic, including:
- Inner loop bodies, SIMD intrinsics, micro-kernel assembly-level optimization
- Tile loop ordering, boundary handling, fallback paths
- The user reviews and tests the code

Integration work (header declarations, Makefile additions, benchmark registration, batch scripts) is always handled by Claude regardless of stage.

### 4. Performance metrics tracked

Through `tsmm_perf_t` and `tsmm_compute_perf()`:
- Wall time, CPU time
- GFLOPS (total FLOPs = 2×m×n×k)
- Memory bandwidth (theoretical read/write bytes / time)
- Arithmetic intensity (FLOP/byte)
- Reserved: L1/L2/L3 miss ratio (for future PAPI integration)

Hardware profiling via `perf stat` in `run_profile.sh` collects:
- cycles, instructions → CPI
- L1-dcache-loads/misses → L1 miss rate
- LLC-loads/misses → L3 miss rate
- branch-instructions/misses → branch prediction miss rate
- l2_rqsts demand hit/miss → L2 miss rate (Cascade Lake specific)

### 5. Fair comparison rules

When comparing with OpenBLAS/MKL:
- Same number of threads (set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS)
- Same NUMA binding
- Same warmup + timed runs count
- Same input data (fixed seed)
- Take min time across runs

### 6. CS V output convention

CSV header: `kernel,shape,m,n,k,layout,wall_ms,gflops,mem_bw_gbs,ai,total_flops,bytes_read,bytes_written,threads`

Kernel names in the `kernel` column: `naive`, `tiled_3d`, `tiled_mn` (add new names for future optimizations).
