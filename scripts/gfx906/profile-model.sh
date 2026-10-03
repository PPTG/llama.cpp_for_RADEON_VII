#!/usr/bin/env bash
# Everything to look for gains on one model: tensor types and bytes per token, kernel profile of the token generation
# (2 GPUs, split heads) and of the prompt processing. Output also in results-gfx906/profile-<model>.txt.
# Run it alone, no other GPU jobs.
#
# usage: scripts/gfx906/profile-model.sh <model.gguf> [build-dir]
# env:   TG_DEPTHS="512 30000", PROMPT=4096, UB=1024, TOP=40
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:?usage: profile-model.sh <model.gguf> [build-dir]}
BUILD=${2:-build-gfx906}
TG_DEPTHS=${TG_DEPTHS:-"512 30000"}
export LLAMA_KV_SPLIT_HEADS=${LLAMA_KV_SPLIT_HEADS:-1}
export TOP=${TOP:-40}
mkdir -p results-gfx906
OUT=results-gfx906/profile-$(basename "${MODEL}" .gguf).txt

{
    echo "##### tensor types"
    python3 scripts/gfx906/gguf-types.py "${MODEL}"
    for d in ${TG_DEPTHS}; do
        echo
        echo "##### token generation, depth ${d}"
        DEPTH=${d} scripts/gfx906/profile-tg.sh "${MODEL}" "${BUILD}"
    done
    echo
    echo "##### prompt processing"
    PROMPT=${PROMPT:-4096} UB=${UB:-1024} scripts/gfx906/profile-pp.sh "${MODEL}" "${BUILD}"
} 2>&1 | tee "${OUT}"
echo "saved: ${OUT}"
