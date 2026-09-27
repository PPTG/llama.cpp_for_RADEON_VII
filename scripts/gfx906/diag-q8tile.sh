#!/usr/bin/env bash
# Narrow down the wrong text with a q8_0 KV cache when the Gemma 4 SWA layers (head 256) use the FA tile kernel
# (GGML_CUDA_FA_Q8_VEC=0). Each line switches off one suspect. Greedy, 48 tokens, 1 GPU.
#
# usage: scripts/gfx906/diag-q8tile.sh [model.gguf] [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BIN=${2:-build-dpp}/bin
PROMPT="Explain in a few sentences how a GPU executes a matrix multiplication."

gen() {
    env "$@" "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm none -ctk q8_0 -ctv q8_0 -no-cnv --temp 0 -n 48 \
        -p "${PROMPT}" --no-display-prompt 2>/dev/null | tr '\n' ' ' | head -c 110
    echo
}

printf '%-44s ' "vector kernel (default)";                      gen
printf '%-44s ' "tile";                                         gen GGML_CUDA_FA_Q8_VEC=0
printf '%-44s ' "tile, f16 conversion";                         gen GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_FA_TILE_Q8=0
printf '%-44s ' "tile, no fusions";                             gen GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_DISABLE_FUSION=1
printf '%-44s ' "tile, rotation not fused";                     gen GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_FUSE_FWHT=0
printf '%-44s ' "tile, no rotation";                            gen GGML_CUDA_FA_Q8_VEC=0 LLAMA_ATTN_ROT_DISABLE=1
printf '%-44s ' "tile, old combine";                            gen GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_FA_COMBINE_SPLIT=0
printf '%-44s ' "tile, no HIP graphs";                          gen GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_DISABLE_GRAPHS=1
