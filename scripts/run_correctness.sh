#!/bin/bash
# ============================================================================
# run_correctness.sh — 正确性验证
#   sbatch scripts/run_correctness.sh
# ============================================================================
#SBATCH --job-name=tsmm_check
#SBATCH --output=logs/check_%j.out
#SBATCH --error=logs/check_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

echo "=== Correctness Test [${TAG:-default}] ==="
echo "Node: $(hostname)  Date: $(date)"
echo ""

if [ "${NO_BUILD:-0}" != "1" ]; then
    make clean
    make
    echo ""
fi

export OMP_NUM_THREADS=1
./build/test_correctness
echo ""
echo "=== Done at $(date) ==="
