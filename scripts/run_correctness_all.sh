#!/bin/bash
# ============================================================================
# run_correctness_all.sh — 正确性验证提交
#
# 预编译一次，提交 1 个任务。
# 用法:  bash scripts/run_correctness_all.sh
# ============================================================================


echo "=== Submitting correctness check ==="

TAG="check_all"

sbatch \
    --job-name="tsmm_check" \
    --output="logs/${TAG}_%j.out" \
    --error="logs/${TAG}_%j.err" \
    --export=ALL,TAG="$TAG",NO_BUILD=1 \
    scripts/run_correctness.sh

echo "Submitted.  Output:  logs/check_all_<jobid>.out"
echo "Monitor:  squeue -u \$USER"
