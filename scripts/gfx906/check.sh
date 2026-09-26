#!/usr/bin/env bash
# Quick round check: correctness of the fused/changed ops vs CPU, tg speed on 1 and 2 GPUs, kernel profile.
#
# usage: scripts/gfx906/check.sh [model.gguf] [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BUILD_DIR=${2:-build-dpp}
BIN=${BUILD_DIR}/bin

echo "=== correctness vs CPU"
for OP in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION MUL_MAT_VEC_FUSION_MERGED; do
    R=$("${BIN}/test-backend-ops" -b ROCm0 -o "${OP}" 2>&1 | tee "/tmp/check-${OP}.log" | grep -E "tests passed" | tail -n 1)
    echo "${OP}: ${R}"
    grep -m 5 "FAIL" "/tmp/check-${OP}.log"
done

echo
echo "=== tg128"
for SM in none layer; do
    "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm "${SM}" -p 0 -n 128 -r 5 -d 0,4096 -o md 2>/dev/null | grep tg128
done

echo
echo "=== kernel profile (1 GPU)"
if command -v rocprofv3 >/dev/null; then
    rm -rf /tmp/check-prof
    rocprofv3 --kernel-trace --stats --output-format csv -d /tmp/check-prof -o prof -- \
        "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -sm none -fa 1 -p 0 -n 32 -r 1 > /tmp/check-prof.log 2>&1
    STATS=$(find /tmp/check-prof -name "*kernel_stats.csv" | head -n 1)
    [ -n "${STATS}" ] && python3 scripts/gfx906/prof-top.py "${STATS}" 25
fi
