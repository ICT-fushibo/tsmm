#!/bin/bash
# ============================================================================
# run_s5_all.sh — AVX-512 S5 正确性 + benchmark + profile 批量提交
#
# 预编译一次，提交 3 个 job。
# 用法:  bash scripts/run_s5_all.sh
# ============================================================================

echo "=== Pre-building (BIN_SUFFIX=_s5) ==="
make BIN_SUFFIX=_s5 2>&1 | tail -2
echo ""

echo "=== Submitting S5 jobs ==="

# 1. Correctness (serial, 1 thread)
echo "  -> correctness"
sbatch --job-name="chk_s5" \
       --output="logs/check_s5_%j.out" --error="logs/check_s5_%j.err" \
       --export=ALL,BIN_SUFFIX=_s5,KERNEL=avx512_s5,THREADS=1 \
       scripts/run_correctness_omp.sh

# 2. Benchmark (all shapes, row+col)
echo "  -> benchmark"
sbatch --job-name="bm_s5" \
       --output="logs/bench_s5_%j.out" --error="logs/bench_s5_%j.err" \
       --export=ALL,KERNEL=avx512_s5,SHAPE=all,NO_BUILD=1 \
       scripts/run_benchmark_s5.sh

# 3. Profile (all shapes, row only, perf stat)
echo "  -> profile"
sbatch --job-name="pf_s5" \
       --output="logs/prof_s5_%j.out" --error="logs/prof_s5_%j.err" \
       --export=ALL,KERNEL=avx512_s5,SHAPE=all,NO_BUILD=1 \
       scripts/run_profile_s5.sh

echo ""
echo "=== Submitted 3 jobs ==="
echo "Monitor:  squeue -u \$USER"
