#!/usr/bin/env bash
# diff-runs.sh over a set of variants (split heads on/off, second stream, KV type, FA kernel): which ones decode
# non-deterministically and where they start to diverge.
#
# usage: scripts/gfx906/diff-matrix.sh [model.gguf] [build-dir]
# env:   CTX=1024, FIRST=1 (see diff-runs.sh)
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
BUILD=${2:-build-gfx906}

v() { # name, KV args, env...
    local name=$1 kv=$2; shift 2
    printf '%-22s ' "${name}"
    env "$@" KV="${kv}" scripts/gfx906/diff-runs.sh "${MODEL}" "${BUILD}" | sed -n '1,2p' | tr '\n' ' '
    echo
}

Q8="-ctk q8_0 -ctv q8_0"
F16="-ctk f16 -ctv f16"
v split-q8             "${Q8}"  GGML_CUDA_FA_Q8_WAVE=1 LLAMA_KV_SPLIT_HEADS=1
v split-q8-again       "${Q8}"  GGML_CUDA_FA_Q8_WAVE=1 LLAMA_KV_SPLIT_HEADS=1
v split-q8-stream0     "${Q8}"  GGML_CUDA_FA_Q8_WAVE=1 LLAMA_KV_SPLIT_HEADS=1 LLAMA_KV_SPLIT_STREAM=0
v split-q8-tile        "${Q8}"  GGML_CUDA_FA_Q8_WAVE=0 LLAMA_KV_SPLIT_HEADS=1
v split-f16            "${F16}" LLAMA_KV_SPLIT_HEADS=1
v nosplit-q8           "${Q8}"  GGML_CUDA_FA_Q8_WAVE=1
v nosplit-f16          "${F16}"
