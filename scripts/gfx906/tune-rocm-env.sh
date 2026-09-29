#!/usr/bin/env bash
# ROCm runtime settings (HSA/HIP environment variables) vs token generation: the token time is bound by ~940 small
# kernels and ~60 copies between the GPUs per token, so the launch/copy/sync paths of the runtime matter.
#
# usage: scripts/gfx906/tune-rocm-env.sh [model.gguf] [build-dir]
# env:   DEPTHS="0,15000" REPS=5
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
BIN=${2:-build-gfx906}/bin
DEPTHS=${DEPTHS:-"0,15000"}
REPS=${REPS:-5}

bench() {
    local name=$1; shift
    printf '%-44s' "${name}"
    env LLAMA_KV_SPLIT_HEADS=1 "$@" "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer -ctk q8_0 -ctv q8_0 \
        -p 0 -n 32 -r "${REPS}" -d "${DEPTHS}" -o csv 2>/dev/null \
        | tail -n +2 | awk -F, '{gsub(/"/, "", $(NF-1)); gsub(/"/, "", $NF); printf "  %8.2f ± %5.2f", $(NF-1), $NF}'
    echo
}

echo "tg32 t/s at depths ${DEPTHS} (2 GPUs, -sm layer, LLAMA_KV_SPLIT_HEADS=1, q8_0 KV)"
bench "default"
bench "HSA_ENABLE_SDMA=0"              HSA_ENABLE_SDMA=0
bench "HIP_FORCE_DEV_KERNARG=1"        HIP_FORCE_DEV_KERNARG=1
bench "HSA_ENABLE_INTERRUPT=0"         HSA_ENABLE_INTERRUPT=0
bench "GPU_MAX_HW_QUEUES=1"            GPU_MAX_HW_QUEUES=1
bench "GPU_MAX_HW_QUEUES=2"            GPU_MAX_HW_QUEUES=2
bench "SDMA=0 + DEV_KERNARG=1 + INTERRUPT=0" HSA_ENABLE_SDMA=0 HIP_FORCE_DEV_KERNARG=1 HSA_ENABLE_INTERRUPT=0
