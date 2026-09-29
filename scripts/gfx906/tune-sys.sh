#!/usr/bin/env bash
# Runtime/driver knobs that affect kernel launch latency and clocks, tg128 on 1 GPU and 2 GPUs.
# Needs root for the perflevel test; the perflevel is set back to auto at exit.
# Run it alone, no other GPU jobs.
#
# usage: scripts/gfx906/tune-sys.sh [model.gguf] [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
BIN=${2:-build-gfx906}/bin
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH="${ROCM_PATH}/bin:${PATH}"

bench() {
    local sm=$1; shift
    env "$@" "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm "${sm}" -p 0 -n 128 -r 5 -o csv 2>/dev/null \
        | tail -n 1 | awk -F, '{gsub(/"/, "", $(NF-1)); printf "%7.2f", $(NF-1)}'
}

row() {
    local name=$1; shift
    printf "%-36s %8s %8s\n" "${name}" "$(bench none "$@")" "$(bench layer "$@")"
}

restore() {
    rocm-smi --setperflevel auto > /dev/null 2>&1
}
trap restore EXIT

printf "%-36s %8s %8s\n" "config" "1 GPU" "2 GPU"
row "baseline"
row "NORM_BLOCK_THRESHOLD=4096"             GGML_CUDA_NORM_BLOCK_THRESHOLD=4096
row "HIP_FORCE_DEV_KERNARG=1"               HIP_FORCE_DEV_KERNARG=1
row "DEBUG_CLR_GRAPH_PACKET_CAPTURE=1"      DEBUG_CLR_GRAPH_PACKET_CAPTURE=1
row "DEBUG_CLR_GRAPH_PACKET_CAPTURE=0"      DEBUG_CLR_GRAPH_PACKET_CAPTURE=0

if rocm-smi --setperflevel high > /dev/null 2>&1; then
    row "perflevel high"
    row "perflevel high + DEV_KERNARG"      HIP_FORCE_DEV_KERNARG=1
    restore
else
    echo "perflevel high: could not set (need root?)"
fi
row "baseline again"
