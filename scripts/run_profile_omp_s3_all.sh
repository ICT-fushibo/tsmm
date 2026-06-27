#!/bin/bash
# ============================================================================
# run_profile_omp_s3_all.sh — Step 3 (private buffer) profiling 批量提交
#
# 用法:  bash scripts/run_profile_omp_s3_all.sh
#        （登录节点先 make BIN_SUFFIX=_omp_s3）
# ============================================================================

echo "=== Submitting Step 3 profiling jobs (row-major only) ==="

for KERN in omp_s3_3d omp_s3_mn; do
    case "$KERN" in
        omp_s3_3d) KTAG="s3_3d" ;;
        omp_s3_mn) KTAG="s3_mn" ;;
    esac
    TAG="prof_${KTAG}_row"
    JOB_NAME="pf_${KTAG}_r"

    echo "  -> $TAG  (sweeps 1,4,16,24,48,96 threads)"

    sbatch \
        --job-name="$JOB_NAME" \
        --output="logs/${TAG}_%j.out" \
        --error="logs/${TAG}_%j.err" \
        --export=ALL,BIN_SUFFIX=_omp_s3,KERNEL="$KERN",LAYOUT="row",TAG="$TAG",NO_BUILD=1 \
        scripts/run_profile_omp.sh
done

echo ""
echo "Submitted 2 jobs.  Monitor: squeue -u \$USER"
