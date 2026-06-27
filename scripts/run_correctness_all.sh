#!/bin/bash
# ============================================================================
# run_correctness_all.sh — 正确性验证提交
#
# 用法:  bash scripts/run_correctness_all.sh
# ============================================================================

echo "=== Building (BIN_SUFFIX=_omp) ==="
make BIN_SUFFIX=_omp
echo ""

echo "=== Submitting correctness check ==="

TAG="check_omp"

sbatch \
    --job-name="tsmm_ck_omp" \
    --output="logs/${TAG}_%j.out" \
    --error="logs/${TAG}_%j.err" \
    --export=ALL,TAG="$TAG",NO_BUILD=1,BIN_SUFFIX="_omp" \
    scripts/run_correctness.sh

echo "Submitted.  Output:  logs/check_omp_<jobid>.out"
