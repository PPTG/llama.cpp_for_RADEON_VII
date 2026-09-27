#!/usr/bin/env bash
# LLAMA_KV_SPLIT_HEADS=1: the KV heads of the global (non-SWA) layers are split over the two GPUs, the attention of each
# half runs on its GPU at the same time. Text check and token generation at long context, 2 GPU layer split, q8_0 KV.
#
# usage: scripts/gfx906/split-heads.sh [model.gguf] [build-dir]
# env:   DEPTHS="0,15000" (at most 32768, the prefill takes long)
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BIN=${2:-build-dpp}/bin
DEPTHS=${DEPTHS:-"0,15000,32768"}
PROMPT="Explain in a few sentences how a GPU executes a matrix multiplication."

echo "=== KV cache placement"
LLAMA_KV_SPLIT_HEADS=1 "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm layer -ctk q8_0 -ctv q8_0 -no-cnv \
    --temp 0 -n 1 -p "hi" 2>&1 | grep -E "KV heads|KV buffer size|error|failed" | head -12

echo
echo "=== text (greedy, 48 tokens; the split sums the attention parts in another order, small differences are possible)"
for S in 0 1; do
    printf 'split=%s: ' "${S}"
    LLAMA_KV_SPLIT_HEADS=${S} "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm layer -ctk q8_0 -ctv q8_0 -no-cnv \
        --temp 0 -n 48 -p "${PROMPT}" --no-display-prompt 2>/dev/null | tr '\n' ' ' | head -c 150
    echo
done

echo
echo "=== tg32"
for S in 0 1; do
    echo "--- LLAMA_KV_SPLIT_HEADS=${S}"
    LLAMA_KV_SPLIT_HEADS=${S} "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer -ctk q8_0 -ctv q8_0 -p 0 -n 32 -r 3 \
        -d "${DEPTHS}" 2>&1 | grep -E "tg32|error|abort"
done
