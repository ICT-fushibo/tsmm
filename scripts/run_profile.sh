#!/bin/bash
# ============================================================================
# run_profile.sh — 硬件性能计数器采集（cache miss, CPI, 分支预测等）
#
# 用法：
#   sbatch scripts/run_profile.sh                           # naive + tiled
#   KERNEL=naive sbatch scripts/run_profile.sh               # 只测 naive
#   KERNEL=tiled_3d sbatch scripts/run_profile.sh            # 只测 tiled_3d
#
# 原理：
#   用 Linux perf stat 在 benchmark 外层采集硬件计数器。
#   不需要改代码，perf 利用 PMU (Performance Monitoring Unit) 直接读寄存器。
#
# 采集指标（Intel Cascade Lake / Xeon Platinum 9242）：
#   - cycles, instructions          → CPI = cycles / instructions
#   - L1-dcache-loads, L1-dcache-load-misses → L1 miss rate
#   - LLC-loads, LLC-load-misses    → L3 (LLC) miss rate
#   - branch-instructions, branch-misses → 分支预测 miss rate
#   - cpu-clock, task-clock         → CPU 利用率
#   - fp_arith_inst_retired.256b_packed_double  → AVX-512 FP 利用率 (参考)
# ============================================================================
#SBATCH --job-name=tsmm_profile
#SBATCH --output=logs/profile_%j.out
#SBATCH --error=logs/profile_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

# --- 参数 ---
KERNEL_LIST="${KERNEL:-naive tiled_3d tiled_mn}"   # 空格分隔，逐个测
TILE_M="${TILE_M:-64}"
TILE_N="${TILE_N:-256}"
TILE_K="${TILE_K:-256}"
RUNS="${RUNS:-5}"             # profiling 时 runs 少一点，perf 开销大
SHAPE="${SHAPE:-4}"           # 默认只测 shape 4 (144x144x144, 较快)
LAYOUT="${LAYOUT:-row}"       # 默认只测 row-major

echo "=== TSMM Hardware Profiling [${TAG:-default}] ==="
echo "Node: $(hostname)  Date: $(date)"
echo "Kernels: $KERNEL_LIST"
echo "Shape: $SHAPE   Layout: $LAYOUT   Runs: $RUNS"
echo "Tile: Ti=$TILE_M Tj=$TILE_N Tk=$TILE_K"
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

# --- 布局参数 ---
LAYOUT_ARG="--layout 2"
[ "$LAYOUT" = "row" ] && LAYOUT_ARG="--layout 0"
[ "$LAYOUT" = "col" ] && LAYOUT_ARG="--layout 1"

# --- 检查 perf 是否可用 ---
if ! perf stat -e cycles:u true 2>/dev/null; then
    echo "WARNING: perf stat not available or no user-access to PMU."
    echo "Trying without :u suffix..."
    PERF_SUFFIX=""
else
    PERF_SUFFIX=":u"
fi

# --- 逐 kernel 采集 ---
for KERN in $KERNEL_LIST; do
    echo ""
    echo "======================================================================"
    echo "  Profiling: $KERN"
    echo "======================================================================"

    # perf stat 事件列表
    # 基础
    EVENTS="cycles${PERF_SUFFIX},instructions${PERF_SUFFIX}"
    # L1 数据 cache
    EVENTS="$EVENTS,L1-dcache-loads${PERF_SUFFIX},L1-dcache-load-misses${PERF_SUFFIX}"
    # L3 / LLC
    EVENTS="$EVENTS,LLC-loads${PERF_SUFFIX},LLC-load-misses${PERF_SUFFIX}"
    # 分支预测
    EVENTS="$EVENTS,branch-instructions${PERF_SUFFIX},branch-misses${PERF_SUFFIX}"
    # CPU 时钟
    EVENTS="$EVENTS,task-clock"

    # 可选：L2 cache (Cascade Lake 专用事件)
    EVENTS="$EVENTS,l2_rqsts.demand_data_rd_hit${PERF_SUFFIX},l2_rqsts.demand_data_rd_miss${PERF_SUFFIX}" 2>/dev/null || true

    # 可选：AVX-512 FP 指令 (衡量 SIMD 利用率)
    # EVENTS="$EVENTS,fp_arith_inst_retired.256b_packed_double${PERF_SUFFIX}" 2>/dev/null || true

    perf stat \
        -e "$EVENTS" \
        -o "logs/perf_${TAG:-${KERN}}_${SLURM_JOB_ID:-local}.txt" \
        --metric-only \
        ./build/benchmark \
            --kernel "$KERN" \
            --tile-m "$TILE_M" --tile-n "$TILE_N" --tile-k "$TILE_K" \
            --runs "$RUNS" --threads 1 \
            --shape "$SHAPE" $LAYOUT_ARG 2>&1 || true

    echo ""
    echo "--- perf stat raw output ---"
    cat "logs/perf_${KERN}_${SLURM_JOB_ID:-local}.txt" 2>/dev/null || echo "(no output)"

    echo ""
    echo "======================================================================"
done

echo ""
echo "=== Done at $(date) ==="
echo ""
echo "=== 如何使用这些数据 ==="
echo "  L1 miss rate  = L1-dcache-load-misses / L1-dcache-loads"
echo "  L2 miss rate  = l2_rqsts.demand_data_rd_miss / (hit + miss)"
echo "  L3 miss rate  = LLC-load-misses / LLC-loads"
echo "  CPI           = cycles / instructions"
echo "  Branch MPKI   = branch-misses / (instructions / 1000)"
echo ""
echo "  高 L1 miss rate → tile 太大，放不进 L1"
echo "  高 L2 miss rate → tile 太大，放不进 L2"
echo "  高 L3 miss rate → 数据量超出 LLC，在访问主存"
echo "  高 CPI (>2)    → 内存墙，CPU 在等数据"
echo "  低 CPI (≈0.5)  → 计算瓶颈，优化空间在 SIMD / ILP"
