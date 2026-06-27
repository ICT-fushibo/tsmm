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
#SBATCH --cpus-per-task=1
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

KERNEL="${KERNEL:-avx512_s5}"
SHAPE="${SHAPE:-all}"
LAYOUT="${LAYOUT:-row}"
RUNS="${RUNS:-10}"
TILE_M="${TILE_M:-64}"
TILE_N="${TILE_N:-256}"
TILE_K="${TILE_K:-256}"
BIN_SUFFIX="${BIN_SUFFIX:-_s5}"

BIN="./build/benchmark${BIN_SUFFIX}"

[ "${NO_BUILD:-0}" != "1" ] && { make BIN_SUFFIX="$BIN_SUFFIX" 2>&1 | tail -2; echo ""; }

SHAPE_ARG=""
[ "$SHAPE" != "all" ] && SHAPE_ARG="--shape $SHAPE"
LAYOUT_ARG="--layout 2"
[ "$LAYOUT" = "row" ] && LAYOUT_ARG="--layout 0"
[ "$LAYOUT" = "col" ] && LAYOUT_ARG="--layout 1"

export OMP_NUM_THREADS=1

echo "=== S5 AVX-512 Benchmark ==="
echo "Kernel: $KERNEL  Shape: $SHAPE  Layout: $LAYOUT"
echo "Tile: Ti=$TILE_M Tj=$TILE_N Tk=$TILE_K  Runs: $RUNS"
echo "Node: $(hostname)  Date: $(date)"
echo ""

$BIN --kernel "$KERNEL" \
     --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
     --runs "$RUNS" --threads 1 \
     --csv \
     $SHAPE_ARG $LAYOUT_ARG

echo ""
echo "=== Done at $(date) ==="
