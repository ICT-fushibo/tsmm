#!/bin/bash
# ============================================================================
# run_benchmark.sh — 性能 benchmark（GFLOPS, 带宽, AI）
#
# 用法：
#   sbatch scripts/run_benchmark.sh                        # all kernels
#   sbatch scripts/run_benchmark.sh --kernel naive          # 只测 naive
#   sbatch scripts/run_benchmark.sh --kernel tiled_3d       # 只测 tiled 3D
#   KERNEL=naive TILE_M=128 sbatch scripts/run_benchmark.sh # 调参
# ============================================================================
#SBATCH --job-name=tsmm_bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

# --- 参数（可通过环境变量覆盖）---
KERNEL="${KERNEL:-all}"         # naive | tiled_3d | tiled_mn | all
TILE_M="${TILE_M:-64}"
TILE_N="${TILE_N:-256}"
TILE_K="${TILE_K:-256}"
RUNS="${RUNS:-10}"
SHAPE="${SHAPE:-all}"           # all | 1-8
LAYOUT="${LAYOUT:-all}"         # all | row | col

echo "=== TSMM Benchmark [${TAG:-default}] ==="
echo "Node: $(hostname)  Date: $(date)"
echo "Kernel: $KERNEL"
echo "Tile: Ti=$TILE_M Tj=$TILE_N Tk=$TILE_K"
echo "Runs: $RUNS"
echo ""

# --- 编译（NO_BUILD=1 时跳过，由批量提交脚本预编译）---
if [ "${NO_BUILD:-0}" != "1" ]; then
    make clean
    make
    echo ""
fi

# --- 线程控制 ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# --- 构建 layout 参数 ---
LAYOUT_ARG="--layout 2"
[ "$LAYOUT" = "row" ] && LAYOUT_ARG="--layout 0"
[ "$LAYOUT" = "col" ] && LAYOUT_ARG="--layout 1"

SHAPE_ARG=""
[ "$SHAPE" != "all" ] && SHAPE_ARG="--shape $SHAPE"

# --- 运行 ---
echo "=== Text output ==="
./build/benchmark \
    --kernel "$KERNEL" \
    --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
    --runs "$RUNS" --threads 1 \
    $SHAPE_ARG $LAYOUT_ARG

echo ""
echo "=== CSV output ==="
./build/benchmark \
    --kernel "$KERNEL" \
    --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
    --runs "$RUNS" --threads 1 \
    --csv \
    $SHAPE_ARG $LAYOUT_ARG

echo ""
echo "=== Done at $(date) ==="
