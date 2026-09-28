#!/usr/bin/env bash
# MMVQ geometry variants (GGML_HIP_MMVQ_VARIANT) per matrix shape of Gemma 4 26B A4B token generation (q4_0):
# us per call for every variant, to pick the variant per shape (e.g. the short rows of the MoE down experts, K = 704).
#
# usage: scripts/gfx906/tune-mmvq-shapes.sh [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

BIN=${1:-build-dpp}/bin
VARIANTS=${VARIANTS:-"1 2 3 4 5 6 7 8 9 10 11"}

# gemma 4: attention / shared MLP (dense, K = 2816 / 2112 / 4096) and the experts (8 of 128, K = 2816 and 704)
FILTER_MM="type_a=q4_0,type_b=f32,m=(4096|2048|2112|2816),n=1,k=(2816|2112|4096),bs"
FILTER_ID="type_a=q4_0,type_b=f32,n_mats=128,n_used=8,b=.,m=(1408|2816),n=1,k=(2816|704)"

run() {
    local v=$1
    {
        GGML_HIP_MMVQ_VARIANT=$v "${BIN}/test-backend-ops" perf -b ROCm0 -o MUL_MAT    -p "${FILTER_MM}" 2>&1
        GGML_HIP_MMVQ_VARIANT=$v "${BIN}/test-backend-ops" perf -b ROCm0 -o MUL_MAT_ID -p "${FILTER_ID}" 2>&1
    } | grep -E "MUL_MAT(_ID)?\(" \
      | sed -E 's/^ *MUL_MAT_ID\(/moe(/; s/^ *MUL_MAT\(/dense(/' \
      | sed -E 's/^([a-z]+)\(.*m=([0-9]+),n=1,k=([0-9]+).*: +[0-9]+ runs - +([0-9.]+) us\/run.*/\1 \2x\3 \4/'
}

printf '%-8s' "variant"
run 8 | awk '{printf "%14s", $1 "-" $2}'
echo
for V in ${VARIANTS}; do
    printf '%-8s' "${V}"
    run "${V}" | awk '{printf "%14s", $3}'
    echo
done
echo "(us per call, lower is better; default variant is 8)"
