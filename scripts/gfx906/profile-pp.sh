#!/usr/bin/env bash
# Kernel profile of the prompt processing: rocprofv3 kernel trace of llama-bench without warmup, kernels per prompt
# token sorted by time (scripts/gfx906/prof-tg.py with ALL=1). With DEPTH > 0 the trace covers the fill of the context
# and the prompt, i.e. the average over the depths 0..DEPTH+PROMPT.
#
# usage: scripts/gfx906/profile-pp.sh [model.gguf] [build-dir]
# env:   PROMPT=4096 DEPTH=0 SM=layer KV=q8_0 UB=512 TOP=30; LLAMA_KV_SPLIT_HEADS etc. are passed on
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BIN=${2:-build-dpp}/bin
PROMPT=${PROMPT:-4096}
DEPTH=${DEPTH:-0}
SM=${SM:-layer}
KV=${KV:-q8_0}
UB=${UB:-512}
TOP=${TOP:-30}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH="${ROCM_PATH}/bin:${PATH}"
OUT=/tmp/profile-pp

rm -rf "${OUT}"
echo "=== rocprofv3: pp${PROMPT} at depth ${DEPTH}, ubatch ${UB}, -sm ${SM}, KV ${KV}, LLAMA_KV_SPLIT_HEADS=${LLAMA_KV_SPLIT_HEADS:-0}"
rocprofv3 --kernel-trace --output-format csv -d "${OUT}" -o prof -- \
    "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm "${SM}" -ctk "${KV}" -ctv "${KV}" -ub "${UB}" \
    -p "${PROMPT}" -n 0 -r 1 -d "${DEPTH}" --no-warmup > "${OUT}.log" 2>&1
grep -E "pp${PROMPT}" "${OUT}.log"

TRACE=$(find "${OUT}" -name "*kernel_trace.csv" | head -n 1)
if [ -z "${TRACE}" ]; then
    echo "no kernel trace, last lines of ${OUT}.log:"
    tail -n 20 "${OUT}.log"
    exit 1
fi
ALL=1 python3 scripts/gfx906/prof-tg.py "${TRACE}" $((PROMPT + DEPTH)) "${TOP}"
