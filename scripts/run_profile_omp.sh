#!/bin/bash
# ============================================================================
# run_profile_omp.sh — 多线程硬件 profiling (perf stat)
#
# 用法同 run_benchmark_omp.sh。
# 每个线程数单独做 perf stat 采集。
# ============================================================================
#SBATCH --job-name=tsmm_omp_prof
#SBATCH --output=logs/omp_prof_%j.out
#SBATCH --error=logs/omp_prof_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

KERNEL="${KERNEL:-tiled_omp_3d}"
LAYOUT="${LAYOUT:-row}"
TILE_M="${TILE_M:-64}"
TILE_N="${TILE_N:-256}"
TILE_K="${TILE_K:-256}"
RUNS="${RUNS:-3}"
THREAD_LIST="${THREAD_LIST:-1 4 16 24 48 96}"
SHAPE="${SHAPE:-all}"

LAYOUT_ARG="--layout 2"
[ "$LAYOUT" = "row" ] && LAYOUT_ARG="--layout 0"
[ "$LAYOUT" = "col" ] && LAYOUT_ARG="--layout 1"

echo "=== TSMM OpenMP Profiling ==="
echo "Kernel: $KERNEL   Layout: $LAYOUT   Shape: $SHAPE"
echo "Threads: $THREAD_LIST"
echo ""

BENCH_BIN="./build/benchmark_omp"
if [ "${NO_BUILD:-0}" != "1" ]; then
    make clean BIN_SUFFIX=_omp && make BIN_SUFFIX=_omp
fi
echo ""

for NTHREADS in $THREAD_LIST; do

    if [ "$NTHREADS" -le 24 ]; then
        NUMA_CMD="numactl --cpunodebind=0 --membind=0"
    elif [ "$NTHREADS" -le 48 ]; then
        NUMA_CMD="numactl --cpunodebind=0-1 --interleave=0-1"
    elif [ "$NTHREADS" -le 72 ]; then
        NUMA_CMD="numactl --cpunodebind=0-2 --interleave=0-2"
    else
        NUMA_CMD="numactl --cpunodebind=0-3 --interleave=all"
    fi

    export OMP_NUM_THREADS=$NTHREADS
    export OMP_PROC_BIND=spread
    export OMP_PLACES=cores
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    TAG="${KERNEL}_${LAYOUT}_t${NTHREADS}"

    echo ""
    echo "===== Threads=$NTHREADS  NUMA: $NUMA_CMD ====="

    # basic events that work universally
    EVENTS="cycles:u,instructions:u"
    EVENTS="$EVENTS,L1-dcache-loads:u,L1-dcache-load-misses:u"
    EVENTS="$EVENTS,LLC-loads:u,LLC-load-misses:u"
    EVENTS="$EVENTS,task-clock"

    perf stat \
        -e "$EVENTS" \
        -o "logs/perf_${TAG}_${SLURM_JOB_ID:-local}.txt" \
        --metric-only \
        $NUMA_CMD $BENCH_BIN \
            --kernel "$KERNEL" \
            --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
            --runs "$RUNS" --threads "$NTHREADS" \
            --shape "$SHAPE" $LAYOUT_ARG 2>&1 || true

    echo "  → logs/perf_${TAG}_*.txt"
done

echo ""
echo "=== Done at $(date) ==="
