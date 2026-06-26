#!/bin/bash
# ============================================================================
# run_profile_all.sh — 硬件 profiling 批量提交（cache miss, CPI, ...）
#
# 先编译一次，再提交 3 kernel × 2 layout = 6 个任务并行跑。
# 用法:  bash scripts/run_profile_all.sh
# ============================================================================


echo "=== Submitting 6 profiling jobs ==="

for KERN in naive tiled_3d tiled_mn; do
    case "$KERN" in
        naive)    KTAG="naiv" ;;
        tiled_3d) KTAG="t3d"  ;;
        tiled_mn) KTAG="tmn"  ;;
    esac
    for LAYOUT in row col; do
        TAG="prof_${KTAG}_${LAYOUT}"
        JOB_NAME="pf_${KTAG}_${LAYOUT:0:1}"

        echo "  → $TAG"

        sbatch \
            --job-name="$JOB_NAME" \
            --output="logs/${TAG}_%j.out" \
            --error="logs/${TAG}_%j.err" \
            --export=ALL,KERNEL="$KERN",LAYOUT="$LAYOUT",TAG="$TAG",NO_BUILD=1 \
            scripts/run_profile.sh
    done
done

echo ""
echo "Submitted 6 profiling jobs.  Monitor:  squeue -u \$USER"
