#!/usr/bin/env bash
# A/B of the fork options on 1 and 2 GPUs, one at a time, interleaved to average out drift.
# Run it alone: other GPU jobs at the same time skew the numbers.
#
# usage: scripts/gfx906/ab.sh [model.gguf] [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
BIN=${2:-build-gfx906}/bin

bench() {
    local sm=$1; shift
    env "$@" "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm "${sm}" -p 0 -n 128 -r 5 -o csv 2>/dev/null \
        | tail -n 1 | awk -F, '{gsub(/"/, "", $(NF-1)); printf "%7.2f", $(NF-1)}'
}

CONFIGS=(
    "all on (default)|"
    "no QKV multi MMVQ|GGML_CUDA_FUSE_QKV=0"
    "no glu+q8 fusion|GGML_CUDA_FUSE_GLU_Q8=0"
    "no multi norm|GGML_CUDA_FUSE_NORM_MULTI=0"
    "no fusions at all|GGML_CUDA_DISABLE_FUSION=1"
    "upstream MMVQ|GGML_HIP_MMVQ_VARIANT=0"
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
