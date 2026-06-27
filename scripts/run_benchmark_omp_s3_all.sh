#!/bin/bash
# ============================================================================
# run_benchmark_omp_s3_all.sh — Step 3 (private buffer) benchmark 批量提交
#
# 用法:  bash scripts/run_benchmark_omp_s3_all.sh
#        （登录节点先 make BIN_SUFFIX=_omp_s3）
# ============================================================================

echo "=== Submitting Step 3 benchmark jobs (row-major only) ==="

for KERN in omp_s3_3d omp_s3_mn; do
    case "$KERN" in
        omp_s3_3d) KTAG="s3_3d" ;;
        omp_s3_mn) KTAG="s3_mn" ;;
    esac
    TAG="bench_${KTAG}_row"
    JOB_NAME="bm_${KTAG}_r"

    echo "  -> $TAG  (sweeps 1..96 threads)"

    sbatch \
        --job-name="$JOB_NAME" \
        --output="logs/${TAG}_%j.out" \
        --error="logs/${TAG}_%j.err" \
        --export=ALL,BIN_SUFFIX=_omp_s3,KERNEL="$KERN",LAYOUT="row",TAG="$TAG",NO_BUILD=1 \
        scripts/run_benchmark_omp.sh
done

echo ""
echo "Submitted 2 jobs.  Monitor: squeue -u \$USER"
