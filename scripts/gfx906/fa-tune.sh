#!/usr/bin/env bash
# Flash attention kernels at the token generation shapes of Gemma 4 (SWA head 256, global head 512 up to 128k),
# f16 vs q8_0 KV, with the FA tuning knobs of this fork. Fast (no model needed).
#
# usage: scripts/gfx906/fa-tune.sh [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

BIN=${1:-build-dpp}/bin

run() {
    echo "=== ${1}"
    shift
    env "$@" "${BIN}/test-backend-ops" perf -b ROCm0 -o FLASH_ATTN_EXT -p "hsk=(256|512),hsv=.*nb=1," 2>&1 \
        | grep -E "FLASH_ATTN_EXT\(" | sed -E 's/FLASH_ATTN_EXT\(hsk=([0-9]+),hsv=[0-9]+,nh=([0-9]+),nr23=\[([0-9]+),1\],kv=([0-9]+),.*type_K=([a-z0-9_]+).*\): +[0-9]+ runs - +([0-9.]+ us\/run).*/hs=\1 nh=\2 gqa=\3 kv=\4 \5: \6/'
}

run "default"
run "q8_0 head 256 on the tile kernel (GGML_CUDA_FA_Q8_VEC=0)" GGML_CUDA_FA_Q8_VEC=0
run "old combine kernel (GGML_CUDA_FA_COMBINE_SPLIT=0)"        GGML_CUDA_FA_COMBINE_SPLIT=0
