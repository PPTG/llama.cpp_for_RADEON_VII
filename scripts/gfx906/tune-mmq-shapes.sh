#!/usr/bin/env bash
# MMQ tile configurations for q4_0 on gfx906 (GGML_HIP_MMQ_Q4_0_VARIANT, ggml/src/ggml-cuda/mmq-config-gcn.cuh) vs the
# prompt processing matrix shapes of Gemma 4 26B A4B (ubatch 512 / 1024) and llama-bench pp4096. Every variant is a
# rebuild of the MMQ files: the variant goes into the untracked ggml/src/ggml-cuda/mmq-config-gcn-local.h, which is
# removed at the end (the build is left at the default variant).
#
# usage: scripts/gfx906/tune-mmq-shapes.sh [model.gguf] [build-dir]
# env:   VARIANTS="0 1 2 3 4 5 6" BENCH=1 (llama-bench pp4096, 2 GPUs, split heads, ub 1024) CHECK=1 (correctness)
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
BUILD=${2:-build-gfx906}
BIN=${BUILD}/bin
VARIANTS=${VARIANTS:-"0 1 2 3 4 5 6"}
BENCH=${BENCH:-1}
CHECK=${CHECK:-1}
LOCAL=ggml/src/ggml-cuda/mmq-config-gcn-local.h
OUT=results-gfx906/tune-mmq
mkdir -p "${OUT}"

FILTER_MM="type_a=q4_0,type_b=f32,m=(4096|2048|2112|2816),n=(512|1024),k=(2816|2112|4096),bs"
FILTER_ID="type_a=q4_0,type_b=f32,n_mats=128,n_used=8,b=.,m=(1408|2816),n=(512|1024),k=(2816|704)"

build() {
    echo "#define GGML_HIP_MMQ_Q4_0_VARIANT $1" > "${LOCAL}"
    touch ggml/src/ggml-cuda/mmq-config-gcn.cuh
    cmake --build "${BUILD}" --target test-backend-ops llama-bench -j"$(nproc)" > "${OUT}/build-$1.log" 2>&1
}

trap 'rm -f "${LOCAL}"' EXIT

for V in ${VARIANTS}; do
    echo "=== variant ${V}"
    if ! build "${V}"; then
        echo "build failed, see ${OUT}/build-${V}.log"
        continue
    fi
    if [ "${CHECK}" = 1 ]; then
        {
            "${BIN}/test-backend-ops" -b ROCm0 -o MUL_MAT    -p "type_a=q4_0,type_b=f32" 2>&1
            "${BIN}/test-backend-ops" -b ROCm0 -o MUL_MAT_ID -p "type_a=q4_0,type_b=f32" 2>&1
        } > "${OUT}/check-${V}.log"
        echo "correctness: $(grep -c 'OK' "${OUT}/check-${V}.log") OK, $(grep -c 'FAIL' "${OUT}/check-${V}.log") FAIL"
    fi
    {
        "${BIN}/test-backend-ops" perf -b ROCm0 -o MUL_MAT    -p "${FILTER_MM}" 2>&1
        "${BIN}/test-backend-ops" perf -b ROCm0 -o MUL_MAT_ID -p "${FILTER_ID}" 2>&1
    } | grep -E "MUL_MAT(_ID)?\(" > "${OUT}/perf-${V}.log"
    sed -E 's/^ *(MUL_MAT(_ID)?)\(.*m=([0-9]+),n=([0-9]+),k=([0-9]+).*: +[0-9]+ runs - +([0-9.]+) us\/run.*/\1 \3x\5 n\4 \6/' \
        "${OUT}/perf-${V}.log" | awk '{printf "  %-10s %-10s %-6s %10.1f us\n", $1, $2, $3, $4}'
    if [ "${BENCH}" = 1 ] && [ -f "${MODEL}" ]; then
        LLAMA_KV_SPLIT_HEADS=1 "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer -ctk q8_0 -ctv q8_0 \
            -p 4096 -n 0 -ub 1024 -r 3 2>/dev/null | grep -E "pp4096" | awk -F'|' '{print "  llama-bench pp4096:" $(NF-1)}'
    fi
done

echo "=== back to the default variant"
rm -f "${LOCAL}"
touch ggml/src/ggml-cuda/mmq-config-gcn.cuh
cmake --build "${BUILD}" -j"$(nproc)" > "${OUT}/build-default.log" 2>&1 && echo "rebuilt" || echo "rebuild failed, see ${OUT}/build-default.log"
