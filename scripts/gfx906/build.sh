#!/usr/bin/env bash
# Build llama.cpp for Radeon VII / Radeon Pro VII / MI50 (gfx906) with HIP.
#
# usage: scripts/gfx906/build.sh [build-dir] [extra cmake args...]
#   e.g. scripts/gfx906/build.sh build-nw4 -DGGML_HIP_MMVQ_GCN_NWARPS=4
#        scripts/gfx906/build.sh build-nodpp -DGGML_HIP_NO_DPP_REDUCE=ON
set -euo pipefail

BUILD_DIR=${1:-build-gfx906}
shift || true

ROCM_PATH=${ROCM_PATH:-/opt/rocm}

HIPCXX="$(${ROCM_PATH}/bin/hipconfig -l)/clang" HIP_PATH="$(${ROCM_PATH}/bin/hipconfig -R)" \
cmake -S . -B "${BUILD_DIR}" \
    -DGGML_HIP=ON \
    -DGPU_TARGETS=gfx906 \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    "$@"

cmake --build "${BUILD_DIR}" --config Release -j"$(nproc)" --target llama-bench llama-cli llama-server llama-completion test-backend-ops
