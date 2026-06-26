#!/bin/bash
# ============================================================================
# run_correctness_all.sh — 正确性验证提交 (OMP version, isolated binary)
#
# 用法:  bash scripts/run_correctness_all.sh
# ============================================================================

echo "=== Pre-building (BIN_SUFFIX=_omp, isolated from old binary) ==="

echo "=== Submitting correctness check ==="

TAG="check_omp"

sbatch \
    --job-name="tsmm_ck_omp" \
    --output="logs/${TAG}_%j.out" \
    --error="logs/${TAG}_%j.err" \
    --export=ALL,TAG="$TAG",NO_BUILD=1,BIN_SUFFIX="_omp" \
    scripts/run_correctness.sh

echo "Submitted.  Output:  logs/check_omp_<jobid>.out"
echo "Monitor:  squeue -u \$USER"
