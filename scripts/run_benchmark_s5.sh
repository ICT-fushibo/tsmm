#!/bin/bash
# ============================================================================
# run_benchmark_s5.sh — AVX-512 S5 benchmark (serial, single-thread)
#
# 用法:
#   sbatch scripts/run_benchmark_s5.sh
#   KERNEL=avx512_s5 SHAPE=4 sbatch scripts/run_benchmark_s5.sh
# ============================================================================
#SBATCH --job-name=bm_s5
#SBATCH --output=logs/bench_s5_%j.out
#SBATCH --error=logs/bench_s5_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

KERNEL="${KERNEL:-avx512_s5_omp}"
SHAPE="${SHAPE:-all}"
LAYOUT="${LAYOUT:-row}"
RUNS="${RUNS:-10}"
TILE_M="${TILE_M:-64}"
TILE_N="${TILE_N:-256}"
TILE_K="${TILE_K:-256}"
BIN_SUFFIX="${BIN_SUFFIX:-_s5}"
THREAD_LIST="${THREAD_LIST:-1 2 4 8 16 24 32 48 64 96}"

BIN="./build/benchmark${BIN_SUFFIX}"

[ "${NO_BUILD:-0}" != "1" ] && { make BIN_SUFFIX="$BIN_SUFFIX" 2>&1 | tail -2; echo ""; }

SHAPE_ARG=""
[ "$SHAPE" != "all" ] && SHAPE_ARG="--shape $SHAPE"

echo "=== S5 AVX-512 Benchmark ==="
echo "Kernel: $KERNEL  Shape: $SHAPE  Layout: $LAYOUT"
echo "Tile: Ti=$TILE_M Tj=$TILE_N Tk=$TILE_K  Runs: $RUNS"
echo "Threads: $THREAD_LIST"
echo "Node: $(hostname)  Date: $(date)"
echo ""

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

for NT in $THREAD_LIST; do
    if [ "$NT" -le 24 ]; then
        NUMA_CMD="numactl --cpunodebind=0 --membind=0"
    elif [ "$NT" -le 48 ]; then
        NUMA_CMD="numactl --cpunodebind=0-1 --interleave=0-1"
    elif [ "$NT" -le 72 ]; then
        NUMA_CMD="numactl --cpunodebind=0-2 --interleave=0-2"
    else
        NUMA_CMD="numactl --cpunodebind=0-3 --interleave=all"
    fi
    export OMP_NUM_THREADS=$NT

    echo "--- Threads=$NT ---"
    $NUMA_CMD $BIN --kernel "$KERNEL" \
         --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
         --runs "$RUNS" --threads "$NT" \
         --csv \
         $SHAPE_ARG --layout 0
    echo ""
done

echo "=== Done at $(date) ==="
