#!/usr/bin/env bash
# CPU sampling with a grammar (tool calls, JSON) and penalties: the chain on the top (top_k + margin) logits
# (LLAMA_SAMPLING_FAST_TOP_K, default on) vs all 262144 logits. The text must be the same (same seed), the sampling
# time per token lower.
#
# usage: scripts/gfx906/sampling-fast.sh [model.gguf] [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BIN=${2:-build-dpp}/bin
PROMPT="Describe three GPUs as a JSON array of objects with the fields name, vendor, memory_gb and year."

for GRAMMAR in "" "--grammar-file grammars/json_arr.gbnf"; do
    echo "=== ${GRAMMAR:-no grammar}"
    for F in 0 1; do
        OUT=$(LLAMA_SAMPLING_STATS=1 LLAMA_SAMPLING_FAST_TOP_K=${F} "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm layer -no-cnv \
            -n 160 --seed 7 --temp 0.8 --top-k 40 --top-p 0.95 --min-p 0.05 --frequency-penalty 0.8 --repeat-penalty 1.1 \
            ${GRAMMAR} -p "${PROMPT}" --no-display-prompt 2> /tmp/sampling-fast-${F}.log)
        echo "${OUT}" > /tmp/sampling-fast-${F}.txt
        printf 'fast=%s: %s\n' "${F}" "$(grep -E 'sampling time|eval time' /tmp/sampling-fast-${F}.log | grep -v prompt | sed -E 's/.*: +//' | tr '\n' ' ')"
        grep -E "sampling stats" /tmp/sampling-fast-${F}.log | sed -E 's/.*sampling stats: /        /'
    done
    if cmp -s /tmp/sampling-fast-0.txt /tmp/sampling-fast-1.txt; then
        echo "OK   same text"
    else
        echo "DIFF text differs:"
        head -c 300 /tmp/sampling-fast-0.txt; echo
        head -c 300 /tmp/sampling-fast-1.txt; echo
    fi
done
