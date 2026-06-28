#!/bin/bash
# ============================================================================
# run_s6_all.sh — S6a/S6f/S6h 正确性 + bench + profile 批量提交
#
# Per-kernel per-shape 专用: s6a→S1, s6f→S6, s6h→S8
# 用法:  bash scripts/run_s6_all.sh
# ============================================================================

KERNELS="s6c:3 s6d:4 s6e:5 s6g:7"
BIN_SUFFIX="_s6h"  # all kernels linked, any suffix works

echo "=== Submitting S6 per-shape jobs ==="
echo ""

for entry in $KERNELS; do
    KERN="${entry%%:*}"
    SH="${entry##*:}"
    TAG="${KERN}"

    # 1. Correctness (target kernel only, target shape only, full thread sweep)
    echo "  -> correctness $KERN (S${SH})"
    sbatch --job-name="chk_${TAG}" \
           --output="logs/check_${TAG}_%j.out" --error="logs/check_${TAG}_%j.err" \
           --export=ALL,BIN_SUFFIX="$BIN_SUFFIX",KERNEL="$KERN",SHAPE="$SH" \
           scripts/run_correctness_omp_all.sh

    # 2. Benchmark (target shape only, full thread sweep)
    echo "  -> benchmark  $KERN (S${SH})"
    sbatch --job-name="bm_${TAG}" \
           --output="logs/bench_${TAG}_%j.out" --error="logs/bench_${TAG}_%j.err" \
           --export=ALL,BIN_SUFFIX="$BIN_SUFFIX",KERNEL="$KERN",SHAPE="$SH",LAYOUT=row,NO_BUILD=1 \
           scripts/run_benchmark_s5.sh

    # 3. Profile (target shape only, full thread sweep, perf stat)
    echo "  -> profile    $KERN (S${SH})"
    sbatch --job-name="pf_${TAG}" \
           --output="logs/prof_${TAG}_%j.out" --error="logs/prof_${TAG}_%j.err" \
           --export=ALL,BIN_SUFFIX="$BIN_SUFFIX",KERNEL="$KERN",SHAPE="$SH",NO_BUILD=1 \
           scripts/run_profile_s5.sh
    echo ""
done

echo "=== Submitted 12 jobs ==="
echo "Monitor: squeue -u \$USER | grep -E 'chk_|bm_|pf_'"
