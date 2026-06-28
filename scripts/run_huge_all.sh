#!/bin/bash
# ============================================================================
# run_huge_all.sh — Huge pages 全线程 sweep (S6 + S8, 3 kernels each)
# ============================================================================

echo "=== Pre-building ==="
make BIN_SUFFIX=_s6h 2>&1 | tail -2
echo ""

for SH in 6 8; do
    for KERN in s4_3d s6f s6h; do
        case "$KERN" in
            s4_3d) KT="s4d" ;;
            s6f)   KT="s6f" ;;
            s6h)   KT="s6h" ;;
        esac
        TAG="huge_S${SH}_${KT}"
        echo "  -> S${SH} $KERN"

        sbatch --job-name="${TAG}" \
               --output="logs/${TAG}_%j.out" --error="logs/${TAG}_%j.err" \
               --export=ALL,BIN_SUFFIX=_s6h,KERNEL="$KERN",SHAPE="$SH",LAYOUT=row,NO_BUILD=1 \
               scripts/run_benchmark_s5.sh
    done
done

echo ""
echo "=== Submitted 6 jobs ==="
echo "Compare with previous results in result/stage2s6f / stage2s6h"
