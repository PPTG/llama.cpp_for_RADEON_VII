#!/usr/bin/env bash
# Tuning variants of the FA tile kernel that reads a q8_0 KV cache (GGML_CUDA_FA_Q8_CFG=0..5): correctness vs CPU and
# time for the Gemma 4 token generation shapes (SWA head 256 GQA 2, global head 512 GQA 8). No model needed.
#
# usage: scripts/gfx906/fa-q8-cfg.sh [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

BIN=${1:-build-dpp}/bin

for CFG in 0 1 2 3 4 5; do
    echo "=== GGML_CUDA_FA_Q8_CFG=${CFG}"
    GGML_CUDA_FA_Q8_CFG=${CFG} "${BIN}/test-backend-ops" -b ROCm0 -o FLASH_ATTN_EXT -p "hsk=(256|512),.*nr23=\[(2|8),1\],.*nb=1,.*type_K=q8_0" 2>&1 \
        | grep -E "FAIL|tests passed" | sed 's/^/  correctness: /'
    GGML_CUDA_FA_Q8_CFG=${CFG} "${BIN}/test-backend-ops" perf -b ROCm0 -o FLASH_ATTN_EXT -p "hsk=(256|512),hsv=.*nb=1,.*type_K=q8_0" 2>&1 \
        | grep -E "FLASH_ATTN_EXT\(" \
        | sed -E 's/FLASH_ATTN_EXT\(hsk=([0-9]+),hsv=[0-9]+,nh=([0-9]+),nr23=\[([0-9]+),1\],kv=([0-9]+),.*type_K=([a-z0-9_]+).*\): +[0-9]+ runs - +([0-9.]+ us\/run).*/hs=\1 nh=\2 gqa=\3 kv=\4 \5: \6/' \
        | grep -E "gqa=(2|8) "
done
