#!/bin/bash
# ============================================================================
# run_huge.sh — Huge pages 对比测试 (S6 + S8)
# ============================================================================
#SBATCH --job-name=huge_test
#SBATCH --output=logs/huge_%j.out
#SBATCH --error=logs/huge_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail
source /public5/soft/modules/module.sh
module purge && module load gcc/10.2.0

BIN_SUFFIX="_s6h"
make BIN_SUFFIX="$BIN_SUFFIX" 2>&1 | tail -2
BIN="./build/benchmark${BIN_SUFFIX}"

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "=== Huge Pages Test ==="
echo "Node: $(hostname)  Date: $(date)"
echo ""
grep -E "HugePages_(Total|Free)" /proc/meminfo 2>/dev/null || echo "(huge page info not available)"
echo ""

for SH in 6 8; do
    for KERN in s4_3d s6f s6h; do
        for NT in 96; do
            [ "$NT" -le 24 ] && NUMA="numactl --cpunodebind=0 --membind=0"
            [ "$NT" -gt 24 ] && [ "$NT" -le 48 ] && NUMA="numactl --cpunodebind=0-1 --interleave=0-1"
            [ "$NT" -gt 48 ] && [ "$NT" -le 72 ] && NUMA="numactl --cpunodebind=0-2 --interleave=0-2"
            [ "$NT" -gt 72 ] && NUMA="numactl --cpunodebind=0-3 --interleave=all"
            export OMP_NUM_THREADS=$NT

            echo "--- S${SH} $KERN t=${NT} ---"
            $NUMA $BIN --kernel "$KERN" --shape "$SH" --runs 5 --threads "$NT" --csv --layout 0
            echo ""
        done
    done
done

echo "=== Done at $(date) ==="
