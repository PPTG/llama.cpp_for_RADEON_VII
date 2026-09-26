#!/usr/bin/env bash
# Sweep MMVQ variants (GGML_HIP_MMVQ_VARIANT) and check the Q8_1 cache (GGML_CUDA_MMVQ_Q8_CACHE).
#
# usage: scripts/gfx906/tune-mmvq.sh [model.gguf] [build-dir] [split-mode]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BUILD_DIR=${2:-build-dpp}
SM=${3:-layer}
BIN=${BUILD_DIR}/bin

bench() {
    env "$@" "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm "${SM}" -p 0 -n 128 -r 5 -o csv 2>/dev/null \
        | tail -n 1 | awk -F, '{gsub(/"/, "", $(NF-1)); printf "%s", $(NF-1)}'
}

echo "=== correctness of the MMVQ variants vs CPU (MUL_MAT, MUL_MAT_ID)"
for V in 1 2 3 4 5 6 7 8; do
    R1=$(GGML_HIP_MMVQ_VARIANT=$V "${BIN}/test-backend-ops" -b ROCm0 -o MUL_MAT    2>&1 | grep -E "tests passed" | tail -n 1)
    R2=$(GGML_HIP_MMVQ_VARIANT=$V "${BIN}/test-backend-ops" -b ROCm0 -o MUL_MAT_ID 2>&1 | grep -E "tests passed" | tail -n 1)
    echo "variant ${V}: MUL_MAT ${R1} | MUL_MAT_ID ${R2}"
done

echo
echo "=== tg128 t/s, split mode ${SM}"
echo "Q8 cache off, default kernel: $(bench GGML_CUDA_MMVQ_Q8_CACHE=0)"
echo "Q8 cache on,  default kernel: $(bench)"
for V in 1 2 3 4 5 6 7 8; do
    echo "Q8 cache on,  variant ${V}:     $(bench GGML_HIP_MMVQ_VARIANT=$V)"
done

echo
echo "=== output check: Q8 cache must not change the text (greedy, 64 tokens)"
PROMPT="Explain in a few sentences how a GPU executes a matrix multiplication."
gen() {
    env "$@" "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm "${SM}" -no-cnv --temp 0 -n 64 -p "${PROMPT}" \
        --no-display-prompt 2>/dev/null
}
REF=$(gen GGML_CUDA_MMVQ_Q8_CACHE=0)
OUT=$(gen)
if [ -n "${OUT}" ] && [ "${OUT}" == "${REF}" ]; then
    echo "OK   Q8 cache on == off"
else
    echo "DIFF Q8 cache on != off"
    echo "--- off: $(echo "${REF}" | head -c 300)"
    echo "--- on:  $(echo "${OUT}" | head -c 300)"
fi
