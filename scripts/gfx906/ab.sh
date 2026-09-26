#!/usr/bin/env bash
# A/B of the fork options on 1 and 2 GPUs, one at a time, interleaved to average out drift.
# Run it alone: other GPU jobs at the same time skew the numbers.
#
# usage: scripts/gfx906/ab.sh [model.gguf] [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BIN=${2:-build-dpp}/bin

bench() {
    local sm=$1; shift
    env "$@" "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm "${sm}" -p 0 -n 128 -r 5 -o csv 2>/dev/null \
        | tail -n 1 | awk -F, '{gsub(/"/, "", $(NF-1)); printf "%7.2f", $(NF-1)}'
}

CONFIGS=(
    "all on (default)|"
    "no gate_up fusion|GGML_CUDA_FUSE_GATE_UP=0"
    "no Q8 cache|GGML_CUDA_MMVQ_Q8_CACHE=0"
    "upstream MMVQ|GGML_HIP_MMVQ_VARIANT=0"
    "MMVQ variant 3|GGML_HIP_MMVQ_VARIANT=3"
)

printf "%-22s %8s %8s\n" "config" "1 GPU" "2 GPU"
for ROUND in 1 2; do
    for C in "${CONFIGS[@]}"; do
        NAME=${C%%|*}
        ENVS=${C#*|}
        # shellcheck disable=SC2086
        printf "%-22s %8s %8s\n" "${NAME}" "$(bench none ${ENVS})" "$(bench layer ${ENVS})"
    done
    echo "---"
done
