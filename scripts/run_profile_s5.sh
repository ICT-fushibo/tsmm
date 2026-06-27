#!/bin/bash
# ============================================================================
# run_profile_s5.sh — AVX-512 S5 profiling (perf stat: cache miss, CPI, ...)
#
# 用法:
#   sbatch scripts/run_profile_s5.sh
#   KERNEL=avx512_s5 SHAPE=all sbatch scripts/run_profile_s5.sh
# ============================================================================
#SBATCH --job-name=pf_s5
#SBATCH --output=logs/prof_s5_%j.out
#SBATCH --error=logs/prof_s5_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

KERNEL="${KERNEL:-avx512_s5}"
SHAPE="${SHAPE:-all}"
LAYOUT="${LAYOUT:-row}"
RUNS="${RUNS:-3}"
TILE_M="${TILE_M:-64}"
TILE_N="${TILE_N:-256}"
TILE_K="${TILE_K:-256}"
BIN_SUFFIX="${BIN_SUFFIX:-_s5}"

BIN="./build/benchmark${BIN_SUFFIX}"

[ "${NO_BUILD:-0}" != "1" ] && { make BIN_SUFFIX="$BIN_SUFFIX" 2>&1 | tail -2; echo ""; }

SHAPE_ARG=""
[ "$SHAPE" != "all" ] && SHAPE_ARG="--shape $SHAPE"

echo "=== S5 AVX-512 Profile (perf stat) ==="
echo "Kernel: $KERNEL  Shape: $SHAPE  Runs: $RUNS"
echo "Tile: Ti=$TILE_M Tj=$TILE_N Tk=$TILE_K"
echo "Node: $(hostname)  Date: $(date)"
echo ""

export OMP_NUM_THREADS=1

# --- perf stat ---
PERF_EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,branch-instructions,branch-misses"

perf stat -e "$PERF_EVENTS" \
    $BIN --kernel "$KERNEL" \
         --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
         --runs "$RUNS" --threads 1 \
         --csv \
         $SHAPE_ARG --layout 0 2>&1

echo ""
echo "=== Done at $(date) ==="
