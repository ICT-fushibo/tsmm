#!/bin/bash
# ============================================================================
# run_benchmark_all.sh — 性能 benchmark 批量提交
#
# 先编译一次，再提交 3 kernel × 2 layout = 6 个任务并行跑。
# 用法:  bash scripts/run_benchmark_all.sh
# ============================================================================

echo "=== Submitting 6 benchmark jobs ==="

for KERN in naive tiled_3d tiled_mn; do
    case "$KERN" in
        naive)    KTAG="naiv" ;;
        tiled_3d) KTAG="t3d"  ;;
        tiled_mn) KTAG="tmn"  ;;
    esac
    for LAYOUT in row col; do
        TAG="bench_${KTAG}_${LAYOUT}"
        JOB_NAME="bm_${KTAG}_${LAYOUT:0:1}"

        echo "  → $TAG"

        sbatch \
            --job-name="$JOB_NAME" \
            --output="logs/${TAG}_%j.out" \
            --error="logs/${TAG}_%j.err" \
            --export=ALL,KERNEL="$KERN",LAYOUT="$LAYOUT",TAG="$TAG",NO_BUILD=1 \
            scripts/run_benchmark.sh
    done
done

echo ""
echo "Submitted 6 benchmark jobs.  Monitor:  squeue -u \$USER"
echo ""
echo "Output files:"
echo "  logs/bench_naiv_row_*.out    logs/bench_naiv_col_*.out"
echo "  logs/bench_t3d_row_*.out     logs/bench_t3d_col_*.out"
echo "  logs/bench_tmn_row_*.out     logs/bench_tmn_col_*.out"
