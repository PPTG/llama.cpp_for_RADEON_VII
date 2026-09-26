#!/usr/bin/env bash
# 2 GPU token generation: compare split modes and copy/AllReduce variants, and check that the output text matches 1 GPU.
#
# usage: scripts/gfx906/multi-gpu.sh [model.gguf] [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BUILD_DIR=${2:-build-dpp}
BIN=${BUILD_DIR}/bin

BENCH_ARGS=(-m "${MODEL}" -ngl 99 -fa 1 -p 0 -n 128 -r 5 -d 0,4096)

run_bench() {
    local name=$1; shift
    echo "=== ${name}"
    env "$@" "${BIN}/llama-bench" "${BENCH_ARGS[@]}" -sm "${SM}" -o md 2>/dev/null | grep -E "tg128"
}

SM=none   run_bench "1 GPU (reference)"
SM=layer  run_bench "layer (staged copy, default)"
SM=layer  run_bench "layer, peer copy"             GGML_CUDA_STAGED_COPY=0
SM=layer  run_bench "layer, peer copy + P2P"       GGML_CUDA_STAGED_COPY=0 GGML_CUDA_P2P=1
SM=tensor run_bench "tensor + P2P"             GGML_CUDA_P2P=1
SM=tensor run_bench "tensor + P2P + AR_P2P"    GGML_CUDA_P2P=1 GGML_CUDA_AR_P2P=1

# correctness: greedy output of each new variant must match the same split mode without it (same math, other transport)
echo "=== output check (greedy, 64 tokens)"
PROMPT="Explain in a few sentences how a GPU executes a matrix multiplication."
gen() {
    env "$@" "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -no-cnv --temp 0 -n 64 -p "${PROMPT}" \
        --no-display-prompt 2>/dev/null
}
check() {
    local ref_desc=$1 new_desc=$2
    # shellcheck disable=SC2086
    local ref; ref=$(gen ${ref_desc})
    # shellcheck disable=SC2086
    local out; out=$(gen ${new_desc})
    if [ -n "${out}" ] && [ "${out}" == "${ref}" ]; then
        echo "OK   [${new_desc}] == [${ref_desc}]"
    else
        echo "DIFF [${new_desc}] != [${ref_desc}]"
        echo "--- ref: $(echo "${ref}" | head -c 300)"
        echo "--- new: $(echo "${out}" | head -c 300)"
    fi
}
check "LLAMA_ARG_SPLIT_MODE=layer GGML_CUDA_STAGED_COPY=0" "LLAMA_ARG_SPLIT_MODE=layer"
check "LLAMA_ARG_SPLIT_MODE=tensor GGML_CUDA_P2P=1" "LLAMA_ARG_SPLIT_MODE=tensor GGML_CUDA_P2P=1 GGML_CUDA_AR_P2P=1"
