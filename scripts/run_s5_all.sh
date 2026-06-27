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

# 1. Correctness (single job: serial + all thread counts OMP)
echo "  -> correctness (serial + 1..96 threads)"
sbatch --job-name="chk_s5_all" \
       --output="logs/check_s5_all_%j.out" --error="logs/check_s5_all_%j.err" \
       --export=ALL,BIN_SUFFIX=_s5,KERNEL=avx512_s5_omp \
       scripts/run_correctness_omp_all.sh

# 2. Benchmark (all shapes, row only, sweep 1/24/48/96 threads)
echo "  -> benchmark"
sbatch --job-name="bm_s5" \
       --output="logs/bench_s5_%j.out" --error="logs/bench_s5_%j.err" \
       --export=ALL,KERNEL=avx512_s5_omp,SHAPE=all,NO_BUILD=1 \
       scripts/run_benchmark_s5.sh

# 3. Profile (all shapes, sweep 1/24/48/96 threads)
echo "  -> profile"
sbatch --job-name="pf_s5" \
       --output="logs/prof_s5_%j.out" --error="logs/prof_s5_%j.err" \
       --export=ALL,KERNEL=avx512_s5_omp,SHAPE=all,NO_BUILD=1 \
       scripts/run_profile_s5.sh

echo ""
echo "=== Submitted 3 jobs ==="
echo "Monitor:  squeue -u \$USER"
