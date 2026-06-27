#!/bin/bash
# ============================================================================
# run_correctness_omp.sh — 单个 OMP kernel 的正确性验证 (SLURM job)
#
# 环境变量:
#   BIN_SUFFIX     binary suffix (e.g. _omp_s4)
#   KERNEL         kernel name (e.g. omp_s4_3d)
#   THREADS        thread count (1, 48, 96)
#   TAG            log tag
#   NO_BUILD       skip build if set to 1
#
# 用法:
#   KERNEL=omp_s4_3d THREADS=96 sbatch scripts/run_correctness_omp.sh
# ============================================================================
#SBATCH --job-name=tsmm_check
#SBATCH --output=logs/check_omp_%j.out
#SBATCH --error=logs/check_omp_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

BIN_SUFFIX="${BIN_SUFFIX:-_omp}"
KERNEL="${KERNEL:-omp_3d}"
THREADS="${THREADS:-1}"

[ "${NO_BUILD:-0}" != "1" ] && { make BIN_SUFFIX="$BIN_SUFFIX"; echo ""; }

BIN="./build/test_correctness${BIN_SUFFIX}"

echo "=== Correctness: $KERNEL  threads=$THREADS ==="
echo "Node: $(hostname)  Date: $(date)"
echo "Binary: $BIN"
echo ""

# 单核或所有核都需要正确性一致
export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

$BIN --kernel "$KERNEL" --threads "$THREADS"

echo ""
echo "=== Done at $(date) ==="
