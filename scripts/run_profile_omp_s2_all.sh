#!/bin/bash
# ============================================================================
# run_profile_omp_s2_all.sh — Step 2 (collapse) profiling 批量提交
#
# 用法:  bash scripts/run_profile_omp_s2_all.sh
#        （登录节点先 make BIN_SUFFIX=_omp_s2）
# ============================================================================

echo "=== Submitting Step 2 profiling jobs (row-major only) ==="

for KERN in omp_s2_3d omp_s2_mn; do
    case "$KERN" in
        omp_s2_3d) KTAG="s2_3d" ;;
        omp_s2_mn) KTAG="s2_mn" ;;
    esac
    TAG="prof_${KTAG}_row"
    JOB_NAME="pf_${KTAG}_r"

    echo "  -> $TAG  (sweeps 1,4,16,24,48,96 threads)"

    sbatch \
        --job-name="$JOB_NAME" \
        --output="logs/${TAG}_%j.out" \
        --error="logs/${TAG}_%j.err" \
        --export=ALL,BIN_SUFFIX=_omp_s2,KERNEL="$KERN",LAYOUT="row",TAG="$TAG",NO_BUILD=1 \
        scripts/run_profile_omp.sh
done

echo ""
echo "Submitted 2 jobs.  Monitor: squeue -u \$USER"
echo "Results in:  logs/prof_s2_3d_row_*.out  logs/prof_s2_mn_row_*.out"
