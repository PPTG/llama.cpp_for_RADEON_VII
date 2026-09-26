#!/usr/bin/env bash
# Token generation benchmark for comparing builds on gfx906.
#
# usage: scripts/gfx906/bench-tg.sh <model.gguf> <build-dir> [<build-dir> ...]
#   e.g. scripts/gfx906/bench-tg.sh ~/models/gemma-4-26b-a4b-Q4_K_M.gguf build-upstream build-gfx906
set -euo pipefail

MODEL=$1
shift

for BUILD_DIR in "$@"; do
    echo "=== ${BUILD_DIR}"
    "${BUILD_DIR}/bin/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -p 0 -n 128 -r 5 -d 0,4096 -o md
done
