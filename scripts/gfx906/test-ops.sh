#!/usr/bin/env bash
# Correctness check of the ops touched by the gfx906 changes against the CPU backend.
#
# usage: scripts/gfx906/test-ops.sh [build-dir]
set -euo pipefail

BUILD_DIR=${1:-build-gfx906}

for OP in MUL_MAT MUL_MAT_ID RMS_NORM NORM SOFT_MAX FLASH_ATTN_EXT ARGSORT TOP_K SUM_ROWS GROUP_NORM L2_NORM; do
    echo "=== ${OP}"
    "${BUILD_DIR}/bin/test-backend-ops" -b ROCm0 -o "${OP}" 2>&1 | tail -n 3
done
