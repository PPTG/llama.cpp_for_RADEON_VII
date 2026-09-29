#!/usr/bin/env bash
# Text with a q8_0 KV cache for the FA kernel variants of the Gemma 4 SWA layers (head 256) and with single features
# switched off. Greedy, 48 tokens, 1 GPU. Different kernels round differently, so only "tile" and
# "tile, f16 conversion" must match exactly.
#
# usage: scripts/gfx906/diag-q8tile.sh [model.gguf] [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
BIN=${2:-build-gfx906}/bin
PROMPT="Explain in a few sentences how a GPU executes a matrix multiplication."

gen() {
    env "$@" "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm none -ctk q8_0 -ctv q8_0 -no-cnv --temp 0 -n 48 \
        -p "${PROMPT}" --no-display-prompt 2>/dev/null | tr '\n' ' ' | head -c 110
    echo
}

printf '%-44s ' "vector kernel";                                gen GGML_CUDA_FA_Q8_VEC=1
printf '%-44s ' "tile";                                         gen GGML_CUDA_FA_Q8_VEC=0
printf '%-44s ' "tile, f16 conversion";                         gen GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_FA_TILE_Q8=0
printf '%-44s ' "tile, no fusions";                             gen GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_DISABLE_FUSION=1
printf '%-44s ' "tile, rotation not fused";                     gen GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_FUSE_FWHT=0
printf '%-44s ' "tile, no rotation";                            gen GGML_CUDA_FA_Q8_VEC=0 LLAMA_ATTN_ROT_DISABLE=1
printf '%-44s ' "tile, old combine";                            gen GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_FA_COMBINE_SPLIT=0
printf '%-44s ' "tile, no HIP graphs";                          gen GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_DISABLE_GRAPHS=1
