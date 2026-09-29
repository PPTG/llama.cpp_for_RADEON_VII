#!/usr/bin/env bash
# Final comparison: upstream build vs this fork, token generation on 1 GPU and 2 GPUs (layer split).
# Run it alone, no other GPU jobs.
#
# usage: scripts/gfx906/final-bench.sh [model.gguf] [upstream-build-dir] [fork-build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
UPSTREAM=${2:-build-nodpp}
FORK=${3:-build-gfx906}
OUT=results-gfx906/final-$(date +%Y%m%d-%H%M%S).md
mkdir -p results-gfx906

run() {
    local name=$1 build=$2 sm=$3
    echo "### ${name}, -sm ${sm}" | tee -a "${OUT}"
    "${build}/bin/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm "${sm}" -p 512 -n 128 -r 5 -d 0,4096,16384 -o md 2>/dev/null \
        | tee -a "${OUT}"
    echo | tee -a "${OUT}"
}

# upstream had no staged copy and no tuned MMVQ, the fork defaults are on
for SM in none layer; do
    GGML_CUDA_STAGED_COPY=0 GGML_HIP_MMVQ_VARIANT=0 GGML_CUDA_MMVQ_Q8_CACHE=0 run "upstream" "${UPSTREAM}" "${SM}"
    run "fork" "${FORK}" "${SM}"
done

echo "saved to ${OUT}"
