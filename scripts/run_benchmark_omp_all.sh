#!/bin/bash
# ============================================================================
# run_benchmark_omp_all.sh — 多线程 benchmark 批量提交
#
# 先编译，再提交 4 kernel × 2 layout = 8 个任务。
# 每个任务内部 sweep 10 个线程数 (1,2,4,8,16,24,32,48,64,96)。
# ============================================================================

echo "=== Pre-building (BIN_SUFFIX=_omp, isolated from old binary) ==="


echo "=== Submitting OMP benchmark jobs (row-major only) ==="

for KERN in tiled_omp_3d tiled_omp_mn; do
    case "$KERN" in
        tiled_omp_3d) KTAG="o3d" ;;
        tiled_omp_mn) KTAG="omn" ;;
    esac
    TAG="bench_${KTAG}_row"
    JOB_NAME="bm_${KTAG}_r"

    echo "  → $TAG  (sweeps 1..96 threads)"

    sbatch \
        --job-name="$JOB_NAME" \
        --output="logs/${TAG}_%j.out" \
        --error="logs/${TAG}_%j.err" \
        --export=ALL,KERNEL="$KERN",LAYOUT="row",TAG="$TAG",NO_BUILD=1 \
        scripts/run_benchmark_omp.sh
done

echo ""
echo "Submitted 2 jobs (each sweeps 10 thread counts).  Monitor: squeue -u \$USER"
echo "Outputs:  logs/bench_o3d_row_*.out  logs/bench_omn_row_*.out"
