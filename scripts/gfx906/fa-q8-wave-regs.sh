#!/usr/bin/env bash
# VGPRs, spills and occupancy of the FA q8_0 head 512 kernels (flash_attn_q8_wave, tile q8_0) with the installed
# ROCm compiler: spills in the inner loops cost memory bandwidth, the register allocation differs between versions.
#
# usage: scripts/gfx906/fa-q8-wave-regs.sh
set -uo pipefail

cd "$(dirname "$0")/../.."

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
OUT=$(mktemp -d)

"${ROCM_PATH}/bin/hipcc" -x hip --offload-arch=gfx906 -O3 -std=c++17 -DGGML_USE_HIP -DGGML_CUDA_FA \
    -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
    -c ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq512-dv512.cu -o "${OUT}/t.o" \
    -Rpass-analysis=kernel-resource-usage 2>&1 \
    | grep -E "Function Name: .*(flash_attn_q8_wave|flash_attn_tileILi512ELi512ELi1ELi8ELb0EL9ggml_type8ELi3E)" -A8 \
    | grep -E "Function Name|VGPRs|Scratch|Occupancy" | sed -E 's/.*remark: //; s/ \[-Rpass-analysis=kernel-resource-usage\]//'
rm -rf "${OUT}"
