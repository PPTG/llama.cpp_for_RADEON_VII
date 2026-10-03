#!/usr/bin/env bash
# Token generation with GGML_CUDA_FA_Q8_WAVE_ROWS (KV rows per wave of the head 512 FlashAttention, 64 = default):
# fewer rows split a short context over more blocks. 2 GPUs, split heads, q8_0 KV. Run it alone.
#
# usage: scripts/gfx906/tune-fa-q8-rows.sh [model.gguf] [build-dir]
# env:   ROWS="64 32 16 8", DEPTHS="512,4096,16384,30000"
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
BIN=${2:-build-gfx906}/bin
ROWS=${ROWS:-"64 32 16 8"}
DEPTHS=${DEPTHS:-"512,4096,16384,30000"}
export LLAMA_KV_SPLIT_HEADS=${LLAMA_KV_SPLIT_HEADS:-1}

printf "%-6s" "rows"
for d in ${DEPTHS//,/ }; do printf " %9s" "d${d}"; done
echo "   (tg32 t/s)"
for r in ${ROWS}; do
    printf "%-6s" "${r}"
    GGML_CUDA_FA_Q8_WAVE_ROWS=${r} "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer -ctk q8_0 -ctv q8_0 \
        -p 0 -n 32 -d "${DEPTHS}" -r 3 -o csv 2>/dev/null | python3 -c '
import csv, sys
print("".join(" %9.2f" % float(r["avg_ts"]) for r in csv.DictReader(sys.stdin)))'
done
