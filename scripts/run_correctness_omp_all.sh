#!/bin/bash
# ============================================================================
# run_correctness_omp_all.sh — OMP 全 kernel 全线程正确性验证（单 job）
#
# 单 job 内遍历 4 个线程数，测试全部 11 个 kernel × 7 shape × 2 layout。
# test_correctness_omp_s4 已链接全部 kernel，不需要多 binary。
#
# 用法:  bash scripts/run_correctness_omp_all.sh
#
# 或直接提交 SLURM job:
#   sbatch scripts/run_correctness_omp_all.sh
# ============================================================================
#SBATCH --job-name=tsmm_chk_all
#SBATCH --output=logs/check_all_%j.out
#SBATCH --error=logs/check_all_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

echo "=== OMP Correctness — All kernels, Multi-thread ==="
echo "Node: $(hostname)  Date: $(date)"
echo ""

BIN_SUFFIX="${BIN_SUFFIX:-_s5}"
BIN="./build/test_correctness${BIN_SUFFIX}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

THREADS="${THREADS:-1 2 4 8 16 24 32 48 64 96}"
KERNEL_FILTER="${KERNEL:-}"
SHAPE_FILTER="${SHAPE:-}"

OK=0
FAIL=0

for NT in $THREADS; do
    echo "=========================================="
    echo "  Threads = $NT"
    echo "=========================================="
    export OMP_NUM_THREADS=$NT

    ARGS="--threads $NT"
    [ -n "$KERNEL_FILTER" ] && ARGS="$ARGS --kernel $KERNEL_FILTER"
    $BIN $ARGS $SHAPE_FILTER
    rc=$?

    echo ""
    if [ $rc -eq 0 ]; then
        echo "  Threads=$NT : ALL PASS"
        OK=$((OK + 1))
    else
        echo "  Threads=$NT : FAILURES DETECTED"
        FAIL=$((FAIL + 1))
    fi
    echo ""
done

echo "=== Summary ==="
echo "Pass: $OK / $((OK + FAIL))"
echo "Done at $(date)"

exit $FAIL
