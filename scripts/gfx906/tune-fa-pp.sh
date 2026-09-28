#!/usr/bin/env bash
# Prompt processing variants of the FA tile kernel (GGML_CUDA_FA_PP_CFG for head size 512, GGML_CUDA_FA_PP_CFG_256 for
# head size 256, 0 = default config): correctness, us per call for the Gemma 4 26B A4B shapes (ubatch 1024, q8_0 KV)
# and llama-bench pp at depth.
#
# usage: scripts/gfx906/tune-fa-pp.sh [model.gguf] [build-dir]
# env:   CFG512="0 11 14 15 16 17" CFG256="0 11 12 13 15 16" BENCH=1 DEPTH=30000
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BIN=${2:-build-dpp}/bin
CFG512=${CFG512:-"0 11 14 15 16 17"}
CFG256=${CFG256:-"0 11 12 13 15 16"}
BENCH=${BENCH:-1}
DEPTH=${DEPTH:-30000}

check() { # env var, value, head size
    env "$1=$2" "${BIN}/test-backend-ops" -b ROCm0 -o FLASH_ATTN_EXT -p "hsk=$3,.*kv=1100" 2>&1 | grep -cE "FAIL"
}

perf() { # env var, value, head size
    env "$1=$2" "${BIN}/test-backend-ops" perf -b ROCm0 -o FLASH_ATTN_EXT -p "hsk=$3,.*nb=1024" 2>&1 \
        | grep -E "FLASH_ATTN_EXT\(" \
        | sed -E 's/.*kv=([0-9]+),nb=([0-9]+).*: +[0-9]+ runs - +([0-9.]+) us\/run.*/kv \1: \3 us/' | tr '\n' ' '
}

bench() { # env var, value
    env "$1=$2" LLAMA_KV_SPLIT_HEADS=1 "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer -ctk q8_0 -ctv q8_0 \
        -p 1024 -n 0 -ub 1024 -r 1 -d "${DEPTH}" 2>/dev/null | grep -E "pp1024" | awk -F'|' '{print $(NF-1)}' | tr -d ' '
}

echo "=== head 512 (GGML_CUDA_FA_PP_CFG): FAIL count, us per call, llama-bench pp1024 @ d${DEPTH}"
for C in ${CFG512}; do
    printf '%3s  fail %s  %s' "${C}" "$(check GGML_CUDA_FA_PP_CFG "${C}" 512)" "$(perf GGML_CUDA_FA_PP_CFG "${C}" 512)"
    [ "${BENCH}" = 1 ] && [ -f "${MODEL}" ] && printf ' pp %s' "$(bench GGML_CUDA_FA_PP_CFG "${C}")"
    echo
done

echo "=== head 256 (GGML_CUDA_FA_PP_CFG_256)"
for C in ${CFG256}; do
    printf '%3s  fail %s  %s' "${C}" "$(check GGML_CUDA_FA_PP_CFG_256 "${C}" 256)" "$(perf GGML_CUDA_FA_PP_CFG_256 "${C}" 256)"
    [ "${BENCH}" = 1 ] && [ -f "${MODEL}" ] && printf ' pp %s' "$(bench GGML_CUDA_FA_PP_CFG_256 "${C}")"
    echo
done
