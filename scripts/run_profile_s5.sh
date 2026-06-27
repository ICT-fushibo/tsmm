#!/bin/bash
#SBATCH --job-name=pf_s5
#SBATCH --output=logs/prof_s5_%j.out
#SBATCH --error=logs/prof_s5_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail
source /public5/soft/modules/module.sh
module purge && module load gcc/10.2.0

KERNEL="${KERNEL:-avx512_s5_omp}"
SHAPE="${SHAPE:-all}"
RUNS="${RUNS:-3}"
TILE_M="${TILE_M:-64}"
TILE_N="${TILE_N:-256}"
TILE_K="${TILE_K:-256}"
BIN_SUFFIX="${BIN_SUFFIX:-_s5}"
THREAD_LIST="${THREAD_LIST:-1 2 4 8 16 24 32 48 64 96}"

BIN="./build/benchmark${BIN_SUFFIX}"
[ "${NO_BUILD:-0}" != "1" ] && { make BIN_SUFFIX="$BIN_SUFFIX" 2>&1 | tail -2; echo ""; }

SHAPE_ARG=""; [ "$SHAPE" != "all" ] && SHAPE_ARG="--shape $SHAPE"

export OMP_PROC_BIND=spread
export OMP_PLACES=cores
PERF_EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,branch-instructions,branch-misses"

echo "=== S5 AVX-512 Profile ==="
echo "Kernel: $KERNEL  Shape: $SHAPE  Threads: $THREAD_LIST"
echo "Node: $(hostname)  Date: $(date)"
echo ""

for NT in $THREAD_LIST; do
    [ "$NT" -le 24 ] && NUMA="numactl --cpunodebind=0 --membind=0"
    [ "$NT" -gt 24 ] && [ "$NT" -le 48 ] && NUMA="numactl --cpunodebind=0-1 --interleave=0-1"
    [ "$NT" -gt 48 ] && NUMA="numactl --cpunodebind=0-3 --interleave=all"
    export OMP_NUM_THREADS=$NT

    echo "--- Threads=$NT ---"
    perf stat -e "$PERF_EVENTS" \
        $NUMA $BIN --kernel "$KERNEL" \
            --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
            --runs "$RUNS" --threads "$NT" --csv \
            $SHAPE_ARG --layout 0 2>&1
    echo ""
done

echo "=== Done at $(date) ==="
