#!/bin/bash
# ============================================================================
# run_profile_omp_all.sh — 多线程 profiling 批量提交
#
# 先编译，再提交 4 kernel × 2 layout = 8 个任务。
# 每个任务内部 sweep 6 个线程数 (1,4,16,24,48,96)。
# ============================================================================

echo "=== Pre-building (BIN_SUFFIX=_omp, isolated from old binary) ==="

echo "=== Submitting OMP profiling jobs (row-major only) ==="

for KERN in tiled_omp_3d tiled_omp_mn; do
    case "$KERN" in
        tiled_omp_3d) KTAG="o3d" ;;
        tiled_omp_mn) KTAG="omn" ;;
    esac
    TAG="prof_${KTAG}_row"
    JOB_NAME="pf_${KTAG}_r"

    echo "  → $TAG  (sweeps 1,4,16,24,48,96 threads)"

    sbatch \
        --job-name="$JOB_NAME" \
        --output="logs/${TAG}_%j.out" \
        --error="logs/${TAG}_%j.err" \
        --export=ALL,KERNEL="$KERN",LAYOUT="row",TAG="$TAG",NO_BUILD=1 \
        scripts/run_profile_omp.sh
done

echo ""
echo "Submitted 2 jobs.  Monitor: squeue -u \$USER"
echo "Outputs:  logs/prof_o3d_row_*.out  logs/prof_omn_row_*.out"
