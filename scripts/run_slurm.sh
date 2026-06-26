#!/bin/bash
# ============================================================================
# run_slurm.sh — BSCC-T6 单节点 TSMM benchmark 提交脚本
#
# 用法:
#   sbatch scripts/run_slurm.sh                 # 全量 benchmark
#   sbatch scripts/run_benchmark.sh             # 更精细的 NUMA sweep
# ============================================================================

#SBATCH --job-name=tsmm_bench
#SBATCH --output=logs/tsmm_%j.out
#SBATCH --error=logs/tsmm_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0                              # all memory

# ============================================================================
# 环境
# ============================================================================
echo "=== SLURM Job: $SLURM_JOB_ID ==="
echo "Node:  $(hostname)"
echo "Date:  $(date)"
echo "CPUs:  $SLURM_CPUS_PER_TASK"
echo ""

# --- 初始化模块系统 ---
source /public5/soft/modules/module.sh

# --- 加载模块 ---
module purge
module load gcc/10.2.0
# module load openblas/0.3.17-ips18   # 需要 BLAS 对比时取消注释
# module load intel/2022.1             # 需要 MKL 对比时取消注释

echo "Loaded modules:"
module list 2>&1
echo ""

gcc --version | head -1
echo ""

# ============================================================================
# 编译（当前仅 tsmm_naive，不需要 BLAS）
# ============================================================================
echo "=== Building ==="
make clean
make
echo ""

# ============================================================================
# 1. 正确性验证
# ============================================================================
echo "=== Correctness Tests ==="
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
./build/test_correctness
echo ""

# ============================================================================
# 2. 性能 benchmark — 串行 baseline
# ============================================================================
echo "=== Serial Baseline (1 thread) ==="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    numactl --cpunodebind=0 --membind=0 \
    ./build/benchmark --runs 20 --threads 1 --csv 2>&1 | tee logs/baseline_serial.csv
echo ""

# ============================================================================
# 3. 多线程 sweep (如果跑全量，用 run_benchmark.sh 更合适)
# ============================================================================
echo "=== Done at $(date) ==="
echo ""
echo "For full thread/NUMA sweep, use:"
echo "  sbatch scripts/run_benchmark.sh --blas openblas"
echo "  sbatch scripts/run_benchmark.sh --blas mkl"
