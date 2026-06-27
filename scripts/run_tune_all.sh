#!/bin/bash
# ============================================================================
# run_tune_all.sh — 提交 3 个 kernel 的 Ti/Tj/Tk 网格搜索 job
#
# 每个 job 内部 sweep 8 shapes × 3 threads (24/48/96)。
# 搜索组合: 3D=150种, MN=30种 tile，每种跑 3 runs 取 min。
#
# 用法:  bash scripts/run_tune_all.sh
# ============================================================================

echo "=== Submitting 3 tuning jobs (one per kernel) ==="

for KERN in omp_s3_3d omp_s3_mn omp_s4_3d; do
    case "$KERN" in
        omp_s3_3d) KT="s3d" ;;
        omp_s3_mn) KT="s3m" ;;
        omp_s4_3d) KT="s4d" ;;
    esac

    JOB_NAME="tune_${KT}"
    echo "  -> $KERN  ($JOB_NAME)"

    sbatch \
        --job-name="$JOB_NAME" \
        --output="logs/tune_${KT}_%j.out" \
        --error="logs/tune_${KT}_%j.err" \
        --export=ALL,KERNEL="$KERN",NO_BUILD=1 \
        scripts/run_tune_by_kernel.sh
done

echo ""
echo "=== Submitted 3 jobs ==="
echo ""
echo "Results per kernel:  logs/tune_s3d_*.out  logs/tune_s3m_*.out  logs/tune_s4d_*.out"
echo "Collect all BEST lines:  grep '^BEST,' logs/tune_s*_*.out"
