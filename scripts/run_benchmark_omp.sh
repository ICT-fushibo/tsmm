#!/bin/bash
# ============================================================================
# run_benchmark_omp.sh — 多线程 benchmark (NUMA-aware thread sweep)
#
# 用法:
#   KERNEL=tiled_omp_3d sbatch scripts/run_benchmark_omp.sh
#   bash scripts/run_benchmark_omp.sh   # 交互式运行
#
# 线程序列: 1 2 4 8 16 24 32 48 64 96
# NUMA 绑定:
#   ≤24:  --cpunodebind=0 --membind=0
#   25-48: --cpunodebind=0-1 --interleave=0-1
#   49-72: --cpunodebind=0-2 --interleave=0-2
#   73-96: --cpunodebind=0-3 --interleave=all
# ============================================================================
#SBATCH --job-name=tsmm_omp
#SBATCH --output=logs/omp_%j.out
#SBATCH --error=logs/omp_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

# --- 参数 ---
KERNEL="${KERNEL:-tiled_omp_3d}"   # tiled_omp_3d | tiled_omp_mn
LAYOUT="${LAYOUT:-row}"
TILE_M="${TILE_M:-64}"
TILE_N="${TILE_N:-256}"
TILE_K="${TILE_K:-256}"
RUNS="${RUNS:-10}"
WARMUP="${WARMUP:-3}"
THREAD_LIST="${THREAD_LIST:-1 2 4 8 16 24 32 48 64 96}"
SHAPE="${SHAPE:-all}"

LAYOUT_ARG="--layout 2"
[ "$LAYOUT" = "row" ] && LAYOUT_ARG="--layout 0"
[ "$LAYOUT" = "col" ] && LAYOUT_ARG="--layout 1"
SHAPE_ARG=""
[ "$SHAPE" != "all" ] && SHAPE_ARG="--shape $SHAPE"

echo "=== TSMM OpenMP Benchmark ==="
echo "Kernel: $KERNEL   Layout: $LAYOUT"
echo "Tile: Ti=$TILE_M Tj=$TILE_N Tk=$TILE_K"
echo "Runs: $RUNS (warmup=$WARMUP)"
echo "Threads: $THREAD_LIST"
echo "Node: $(hostname)  Date: $(date)"
echo ""

# --- 编译（BIN_SUFFIX=_omp 隔离，不覆盖旧 binary）---
BENCH_BIN="./build/benchmark_omp"
if [ "${NO_BUILD:-0}" != "1" ]; then
    make clean BIN_SUFFIX=_omp && make BIN_SUFFIX=_omp
fi
echo ""

# --- 线程 sweep ---
for NTHREADS in $THREAD_LIST; do

    # NUMA 绑定
    if [ "$NTHREADS" -le 24 ]; then
        NUMA_CMD="numactl --cpunodebind=0 --membind=0"
    elif [ "$NTHREADS" -le 48 ]; then
        NUMA_CMD="numactl --cpunodebind=0-1 --interleave=0-1"
    elif [ "$NTHREADS" -le 72 ]; then
        NUMA_CMD="numactl --cpunodebind=0-2 --interleave=0-2"
    else
        NUMA_CMD="numactl --cpunodebind=0-3 --interleave=all"
    fi

    # OMP 环境
    export OMP_NUM_THREADS=$NTHREADS
    export OMP_PROC_BIND=spread
    export OMP_PLACES=cores
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    echo ""
    echo "===== Threads=$NTHREADS  NUMA: $NUMA_CMD ====="

    $NUMA_CMD $BENCH_BIN \
        --kernel "$KERNEL" \
        --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
        --runs "$RUNS" --threads "$NTHREADS" \
        --csv \
        $SHAPE_ARG $LAYOUT_ARG

done

echo ""
echo "=== Done at $(date) ==="
