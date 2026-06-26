#!/bin/bash
# ============================================================================
# run_benchmark.sh — TSMM 多线程 × NUMA 绑定 benchmark 自动化
#
# 用法:
#   # 单线程快速测试（不绑定 NUMA）
#   bash scripts/run_benchmark.sh --quick
#
#   # 完整 sweep（1,2,4,8,16,24,48,96 线程）
#   bash scripts/run_benchmark.sh
#
#   # 指定线程数 + BLAS 对比
#   bash scripts/run_benchmark.sh --threads 24 --blas mkl
#
#   # 提交到 SLURM
#   sbatch scripts/run_benchmark.sh
#
# 输出:
#   logs/bench_<timestamp>_<threads>t.csv     ← 数据（可画图）
#   logs/bench_<timestamp>_<threads>t.log     ← 原始输出
# ============================================================================

set -euo pipefail

# ============================================================================
# SLURM job header (ignored when run directly via bash)
# ============================================================================
#SBATCH --job-name=tsmm_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

# ============================================================================
# 配置
# ============================================================================

# --- 默认参数 ---
QUICK=0
THREAD_LIST=""         # 空 = 用默认序列
BLAS_LIB=""            # openblas | mkl | "" (no BLAS)
SHAPES="all"           # all | 1-8
LAYOUTS="all"          # all | row | col
OUTDIR="logs"
RUNS=20
WARMUP=10

# --- 节点硬件（来自 node_config.mk 中的值，此处硬编码一份供 bash 用）---
NODE_NUMA=4
CORES_PER_NUMA=24

# ============================================================================
# 参数解析
# ============================================================================
while [ $# -gt 0 ]; do
    case "$1" in
        --quick)
            QUICK=1
            RUNS=3
            WARMUP=2
            shift ;;
        --threads)
            THREAD_LIST="$2"
            shift 2 ;;
        --blas)
            BLAS_LIB="$2"
            shift 2 ;;
        --shapes)
            SHAPES="$2"
            shift 2 ;;
        --layouts)
            LAYOUTS="$2"
            shift 2 ;;
        --runs)
            RUNS="$2"
            shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "  --quick          fast test (3 runs, 2 warmup)"
            echo "  --threads N,...  thread counts (default: 1,2,4,8,16,24,48,96)"
            echo "  --blas mkl|openblas  enable BLAS comparison"
            echo "  --shapes 1-8|all   which shapes to test"
            echo "  --layouts row|col|all"
            echo "  --runs N         timed runs per test (default: 20)"
            exit 0 ;;
        *)
            echo "Unknown: $1"
            exit 1 ;;
    esac
done

# --- 默认线程序列 ---
if [ -z "$THREAD_LIST" ]; then
    if [ "$QUICK" = "1" ]; then
        THREAD_LIST="1 48 96"
    else
        THREAD_LIST="1 2 4 8 16 24 48 96"
    fi
fi

# --- 时间戳 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTDIR"
LOG_DIR="$OUTDIR"

# ============================================================================
# 模块加载
# ============================================================================
module_load() {
    local blas="$1"
    module purge 2>/dev/null || true

    case "$blas" in
        openblas)
            echo "=== Loading OpenBLAS ==="
            module load gcc/12.2.0          2>/dev/null || module load gcc 2>/dev/null || true
            module load openblas/0.3.17-ips18 2>/dev/null || true
            ;;
        mkl|intel)
            echo "=== Loading Intel oneAPI ==="
            module load intel/2022.1 2>/dev/null || true
            ;;
        *)
            echo "=== No BLAS module loaded ==="
            module load gcc/12.2.0 2>/dev/null || module load gcc 2>/dev/null || true
            ;;
    esac
    echo "Loaded modules:"
    module list 2>&1 | head -10 || true
}

# ============================================================================
# NUMA 绑定 — 根据线程数选择策略
# ============================================================================
numa_bind() {
    local nthr="$1"

    if [ "$nthr" -le 24 ]; then
        # 单 NUMA node
        echo "numactl --cpunodebind=0 --membind=0"
    elif [ "$nthr" -le 48 ]; then
        # 两个 NUMA node, interleaved
        echo "numactl --cpunodebind=0-1 --interleave=0-1"
    else
        # 全节点 interleaved
        echo "numactl --cpunodebind=0-3 --interleave=all"
    fi
}

# ============================================================================
# OMP 环境变量
# ============================================================================
omp_env() {
    local nthr="$1"
    echo "OMP_PROC_BIND=spread OMP_PLACES=cores"
    echo "OMP_NUM_THREADS=${nthr}"
    echo "MKL_NUM_THREADS=${nthr}"
    echo "OPENBLAS_NUM_THREADS=${nthr}"
    echo "MKL_DYNAMIC=FALSE"
    echo "KMP_AFFINITY=granularity=fine,compact,1,0"
}

# ============================================================================
# 编译
# ============================================================================
do_build() {
    local blas="$1"

    if [ "$blas" = "mkl" ] || [ "$blas" = "intel" ]; then
        make clean -s
        make blas BLAS=mkl -s 2>&1 | tail -3
    elif [ "$blas" = "openblas" ]; then
        make clean -s
        make blas BLAS=openblas -s 2>&1 | tail -3
    else
        make clean -s
        make all -s 2>&1 | tail -3
    fi
    echo "  Binary: $(file build/benchmark 2>/dev/null || echo 'not found')"
}

# ============================================================================
# 运行一组 benchmark
# ============================================================================
run_round() {
    local nthr="$1"
    local blas="$2"
    local label="$3"      # "naive" | "openblas" | "mkl"
    local csv="$4"

    local NUMA=$(numa_bind "$nthr")
    local OMP=$(omp_env "$nthr")

    local LOG="${LOG_DIR}/bench_${TIMESTAMP}_${label}_${nthr}t.log"

    # 构建 benchmark 参数
    local BENCH_ARGS="--runs $RUNS --threads $nthr"
    if [ "$SHAPES" != "all" ]; then
        BENCH_ARGS="$BENCH_ARGS --shape $SHAPES"
    fi
    case "$LAYOUTS" in
        row)  BENCH_ARGS="$BENCH_ARGS --layout 0" ;;
        col)  BENCH_ARGS="$BENCH_ARGS --layout 1" ;;
        *)    BENCH_ARGS="$BENCH_ARGS --layout 2" ;;
    esac
    if [ "$blas" = "" ]; then
        BENCH_ARGS="$BENCH_ARGS --no-blas"
    fi

    echo ""
    echo "--- $label  |  threads=$nthr  |  $(date) ---"
    echo "  NUMA: $NUMA"
    echo "  OMP:  $OMP"
    echo "  Log:  $LOG"

    # 实际运行
    env $OMP $NUMA ./build/benchmark $BENCH_ARGS --csv 2>&1 | tee "$LOG"

    # 提取 CSV 行（以 kernel 名开头的行）
    grep -E '^(tsmm_naive|openblas|mkl),' "$LOG" | while read -r line; do
        echo "$line" >> "$CSV"
    done
}

# ============================================================================
# Main
# ============================================================================

echo "============================================================================"
echo "  TSMM Benchmark Suite"
echo "  Time: $(date)"
echo "  Node: $(hostname)"
echo "  Threads: $THREAD_LIST"
echo "  BLAS:    ${BLAS_LIB:-none}"
echo "  Shapes:  $SHAPES"
echo "  Runs:    $RUNS  (warmup: $WARMUP)"
echo "============================================================================"

# --- 编译 ---
module_load "$BLAS_LIB"
do_build "$BLAS_LIB"

# --- CSV header ---
CSV_FILE="${LOG_DIR}/bench_${TIMESTAMP}_all.csv"
echo "kernel,shape_name,m,n,k,layout,threads,wall_ms,gflops,mem_bw_gbs,ai,peak_pct" > "$CSV_FILE"

# --- 逐线程数测试 ---
for nthr in $THREAD_LIST; do
    echo ""
    echo "====== Threads: $nthr ======"

    # 1. 自己的 tsmm_naive
    run_round "$nthr" "" "tsmm_naive" "$CSV_FILE"

    # 2. BLAS 对比（如果指定）
    if [ "$BLAS_LIB" = "openblas" ]; then
        run_round "$nthr" "openblas" "openblas" "$CSV_FILE"
    elif [ "$BLAS_LIB" = "mkl" ] || [ "$BLAS_LIB" = "intel" ]; then
        run_round "$nthr" "mkl" "mkl" "$CSV_FILE"
    fi
done

# --- 汇总 ---
echo ""
echo "============================================================================"
echo "  Done.  Results:"
echo "    Individual logs: ${LOG_DIR}/bench_${TIMESTAMP}_*.log"
echo "    Combined CSV:    $CSV_FILE"
echo ""
echo "  Quick view:"
head -5 "$CSV_FILE"
echo "  ..."
wc -l "$CSV_FILE"
echo "============================================================================"
