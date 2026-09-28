#!/usr/bin/env bash
# Kernel profile of the token generation: rocprofv3 kernel trace of llama-bench (32 tokens after a short prefill that
# marks where the token generation starts), then the kernels per token sorted by time (scripts/gfx906/prof-tg.py).
#
# usage: scripts/gfx906/profile-tg.sh [model.gguf] [build-dir]
# env:   DEPTH=512 (context before the 32 tokens), SM=layer, KV=q8_0, TOP=40, SEQ=<kernel>, GAPS=1 (see prof-tg.py); LLAMA_KV_SPLIT_HEADS etc. are passed on
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BIN=${2:-build-dpp}/bin
DEPTH=${DEPTH:-512}
SM=${SM:-layer}
KV=${KV:-q8_0}
TOP=${TOP:-40}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH="${ROCM_PATH}/bin:${PATH}"
OUT=/tmp/profile-tg

rm -rf "${OUT}"
echo "=== rocprofv3: tg32 at depth ${DEPTH}, -sm ${SM}, KV ${KV}, LLAMA_KV_SPLIT_HEADS=${LLAMA_KV_SPLIT_HEADS:-0}"
rocprofv3 --kernel-trace --output-format csv -d "${OUT}" -o prof -- \
    "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm "${SM}" -ctk "${KV}" -ctv "${KV}" -p 0 -n 32 -r 1 -d "${DEPTH}" \
    > "${OUT}.log" 2>&1
grep -E "tg32" "${OUT}.log"

TRACE=$(find "${OUT}" -name "*kernel_trace.csv" | head -n 1)
if [ -z "${TRACE}" ]; then
    echo "no kernel trace, last lines of ${OUT}.log:"
    tail -n 20 "${OUT}.log"
    exit 1
fi
python3 scripts/gfx906/prof-tg.py "${TRACE}" 32 "${TOP}"
