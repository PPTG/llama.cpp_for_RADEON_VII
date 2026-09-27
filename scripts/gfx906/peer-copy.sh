#!/usr/bin/env bash
# GGML_CUDA_PEER_COPY=1: small copies between the GPUs by a kernel that writes into the memory of the other GPU, instead
# of a D2H + H2D copy through pinned host memory. Text check and token generation, with and without LLAMA_KV_SPLIT_HEADS.
#
# usage: scripts/gfx906/peer-copy.sh [model.gguf] [build-dir]
# env:   DEPTHS="0,15000"
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BIN=${2:-build-dpp}/bin
DEPTHS=${DEPTHS:-"0,15000"}
PROMPT="Explain in a few sentences how a GPU executes a matrix multiplication."

echo "=== peer access"
GGML_CUDA_PEER_COPY=1 "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm layer -no-cnv --temp 0 -n 1 -p "hi" 2>&1 \
    | grep -iE "peer|error|failed" | head -5

echo
echo "=== text (greedy, 48 tokens, must be the same with and without GGML_CUDA_PEER_COPY)"
for S in 0 1; do
    for P in 0 1; do
        printf 'split=%s peer=%s: ' "${S}" "${P}"
        LLAMA_KV_SPLIT_HEADS=${S} GGML_CUDA_PEER_COPY=${P} "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm layer \
            -ctk q8_0 -ctv q8_0 -no-cnv --temp 0 -n 48 -p "${PROMPT}" --no-display-prompt 2>/dev/null | tr '\n' ' ' | head -c 120
        echo
    done
done

echo
echo "=== tg32 (mean of 3)"
for S in 0 1; do
    for P in 0 1; do
        echo "--- LLAMA_KV_SPLIT_HEADS=${S} GGML_CUDA_PEER_COPY=${P}"
        LLAMA_KV_SPLIT_HEADS=${S} GGML_CUDA_PEER_COPY=${P} "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer \
            -ctk q8_0 -ctv q8_0 -p 0 -n 32 -r 3 -d "${DEPTHS}" 2>&1 | grep -E "tg32|error|abort"
    done
done
