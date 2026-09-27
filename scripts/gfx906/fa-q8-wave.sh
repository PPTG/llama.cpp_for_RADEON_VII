#!/usr/bin/env bash
# FA kernel for the Gemma 4 global layers (q8_0 KV cache, head size 512, GQA 8, token generation) that computes all 8
# Q heads per wave (GGML_CUDA_FA_Q8_WAVE, default on) vs the tile kernel (GGML_CUDA_FA_Q8_WAVE=0):
# correctness vs CPU, time per call, text and tg32 with the model (2 GPU layer split, LLAMA_KV_SPLIT_HEADS=1).
#
# usage: scripts/gfx906/fa-q8-wave.sh [build-dir] [model.gguf]
# env:   DEPTHS="0,15000,32768"
set -uo pipefail

cd "$(dirname "$0")/../.."

BIN=${1:-build-dpp}/bin
MODEL=${2:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
DEPTHS=${DEPTHS:-"0,15000,32768"}
PROMPT="Explain in a few sentences how a GPU executes a matrix multiplication."

echo "=== correctness (head 512, q8_0 KV, CPU reference)"
for W in 1 0; do
    printf 'GGML_CUDA_FA_Q8_WAVE=%s: ' "${W}"
    GGML_CUDA_FA_Q8_WAVE=${W} "${BIN}/test-backend-ops" -b ROCm0 -o FLASH_ATTN_EXT -p "hsk=512,.*type_K=q8_0" 2>&1 \
        | grep -E "FAIL|tests passed" | tr '\n' ' '
    echo
done

echo
echo "=== time per call (nh = KV heads, perm = model KV layout)"
for W in 0 1; do
    echo "--- GGML_CUDA_FA_Q8_WAVE=${W}"
    GGML_CUDA_FA_Q8_WAVE=${W} "${BIN}/test-backend-ops" perf -b ROCm0 -o FLASH_ATTN_EXT -p "hsk=512,.*nr23=\[8,1\],.*nb=1,.*type_K=q8_0" 2>&1 \
        | grep -E "FLASH_ATTN_EXT\(" \
        | sed -E 's/FLASH_ATTN_EXT\(hsk=([0-9]+),hsv=[0-9]+,nh=([0-9]+),nr23=\[([0-9]+),1\],kv=([0-9]+),.*permute=\[([0-9,]+)\].*\): +[0-9]+ runs - +([0-9.]+ us\/run).*/nh=\2 kv=\4 perm=\5: \6/' \
        | sort -u -t: -k1,1 | sort -t= -k3 -n
done

if [ ! -f "${MODEL}" ]; then
    echo "no model at ${MODEL}, skipping the model tests"
    exit 0
fi

echo
echo "=== text (greedy, 48 tokens, LLAMA_KV_SPLIT_HEADS=1)"
for W in 0 1; do
    printf 'wave=%s: ' "${W}"
    GGML_CUDA_FA_Q8_WAVE=${W} LLAMA_KV_SPLIT_HEADS=1 "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm layer \
        -ctk q8_0 -ctv q8_0 -no-cnv --temp 0 -n 48 -p "${PROMPT}" --no-display-prompt 2>/dev/null | tr '\n' ' ' | head -c 150
    echo
done

echo
echo "=== tg32 (LLAMA_KV_SPLIT_HEADS=1)"
for W in 0 1; do
    echo "--- GGML_CUDA_FA_Q8_WAVE=${W}"
    GGML_CUDA_FA_Q8_WAVE=${W} LLAMA_KV_SPLIT_HEADS=1 "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer \
        -ctk q8_0 -ctv q8_0 -p 0 -n 32 -r 3 -d "${DEPTHS}" 2>&1 | grep -E "tg32|error|abort"
done
