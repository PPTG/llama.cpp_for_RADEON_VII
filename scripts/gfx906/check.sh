#!/usr/bin/env bash
# Quick round check: correctness of the fused/changed ops vs CPU, tg speed on 1 and 2 GPUs, kernel profile.
#
# usage: scripts/gfx906/check.sh [model.gguf] [build-dir]
# env: SKIP_TESTS=1, SKIP_TEXT=1, SKIP_BENCH=1, SKIP_PROFILE=1
set -uo pipefail

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH="${ROCM_PATH}/bin:${PATH}"

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BUILD_DIR=${2:-build-dpp}
BIN=${BUILD_DIR}/bin

if [ -z "${SKIP_TESTS:-}" ]; then
echo "=== correctness vs CPU"
for OP in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION MUL_MAT_VEC_FUSION_MERGED MUL_MAT_MULTI GLU_MUL_MAT RMS_NORM_MULTI RMS_NORM_SCALE_MUL RMS_NORM_MUL_ADD MOE_REDUCE FWHT_FUSED FLASH_ATTN_EXT; do
    R=$("${BIN}/test-backend-ops" -b ROCm0 -o "${OP}" 2>&1 | tee "/tmp/check-${OP}.log" | grep -E "tests passed" | tail -n 1)
    echo "${OP}: ${R}"
    grep -m 5 "FAIL" "/tmp/check-${OP}.log"
done
fi

if [ -z "${SKIP_TEXT:-}" ]; then
echo
echo "=== output check (greedy, 64 tokens, 1 GPU)"
PROMPT="Explain in a few sentences how a GPU executes a matrix multiplication."
gen() {
    env "$@" "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm none -no-cnv --temp 0 -n 64 -p "${PROMPT}" \
        --no-display-prompt 2>/dev/null
}
OUT=$(gen)
# these fork fusions compute exactly the same values as the unfused ops, the text must not change
for OFF in GGML_CUDA_FUSE_QKV=0 GGML_CUDA_FUSE_GLU_Q8=0 GGML_CUDA_FUSE_NORM_MULTI=0; do
    REF=$(gen ${OFF})
    if [ -n "${OUT}" ] && [ "${OUT}" == "${REF}" ]; then
        echo "OK   default == ${OFF}"
    else
        echo "DIFF default != ${OFF}"
        echo "--- ${OFF}: $(echo "${REF}" | head -c 300)"
        echo "--- default: $(echo "${OUT}" | head -c 300)"
    fi
done
# q8_0 KV cache: FA tile reading q8_0 directly must give the same text as converting to f16 first
genq8() {
    env "$@" "${BIN}/llama-completion" -m "${MODEL}" -ngl 99 -fa on -sm none -ctk q8_0 -ctv q8_0 -no-cnv --temp 0 -n 64 \
        -p "${PROMPT}" --no-display-prompt 2>/dev/null
}
Q8NEW=$(genq8)
Q8OLD=$(genq8 GGML_CUDA_FA_TILE_Q8=0)
if [ -n "${Q8NEW}" ] && [ "${Q8NEW}" == "${Q8OLD}" ]; then
    echo "OK   q8_0 KV: FA tile q8_0 == f16 conversion"
else
    echo "DIFF q8_0 KV: FA tile q8_0 != f16 conversion"
    echo "--- conversion: $(echo "${Q8OLD}" | head -c 300)"
    echo "--- q8_0:       $(echo "${Q8NEW}" | head -c 300)"
fi

# experimental q8_0 head 256 on the tile kernel: q8_0 read directly vs f16 conversion vs vector kernel (default)
Q8T=$(genq8 GGML_CUDA_FA_Q8_VEC=0)
Q8TC=$(genq8 GGML_CUDA_FA_Q8_VEC=0 GGML_CUDA_FA_TILE_Q8=0)
echo "INFO q8_0 KV, head 256 on tile (GGML_CUDA_FA_Q8_VEC=0), first 150 chars:"
echo "--- vector (default):  $(echo "${Q8NEW}" | head -c 150)"
echo "--- tile, q8_0:        $(echo "${Q8T}" | head -c 150)"
echo "--- tile, f16 conv.:   $(echo "${Q8TC}" | head -c 150)"
# q8_0 KV cache: the Hadamard rotation fused into the cache store / attn_output quantization must not change the text
Q8FWHT=$(genq8 GGML_CUDA_FUSE_FWHT=0)
if [ -n "${Q8NEW}" ] && [ "${Q8NEW}" == "${Q8FWHT}" ]; then
    echo "OK   q8_0 KV: default == GGML_CUDA_FUSE_FWHT=0"
else
    echo "DIFF q8_0 KV: default != GGML_CUDA_FUSE_FWHT=0"
    echo "--- GGML_CUDA_FUSE_FWHT=0: $(echo "${Q8FWHT}" | head -c 300)"
    echo "--- default:               $(echo "${Q8NEW}" | head -c 300)"
fi

NOFUSE=$(gen GGML_CUDA_DISABLE_FUSION=1)
if [ "${OUT}" == "${NOFUSE}" ]; then
    echo "OK   all fusions == no fusions"
else
    echo "INFO all fusions != no fusions (upstream fusions may round differently), first 200 chars:"
    echo "--- no fusion: $(echo "${NOFUSE}" | head -c 200)"
    echo "--- fusion:    $(echo "${OUT}" | head -c 200)"
fi
fi

if [ -z "${SKIP_BENCH:-}" ]; then
echo
echo "=== tg128"
for SM in none layer; do
    "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm "${SM}" -p 0 -n 128 -r 5 -d 0,4096 -o md 2>/dev/null | grep tg128
done
fi

[ -n "${SKIP_PROFILE:-}" ] && exit 0
echo
echo "=== kernel profile (1 GPU)"
if ! command -v rocprofv3 >/dev/null; then
    echo "rocprofv3 not found (PATH=${PATH}), set ROCM_PATH"
    exit 1
fi
rm -rf /tmp/check-prof
rocprofv3 --kernel-trace --stats --output-format csv -d /tmp/check-prof -o prof -- \
    "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -sm none -fa 1 -p 0 -n 32 -r 1 > /tmp/check-prof.log 2>&1
STATS=$(find /tmp/check-prof -name "*kernel_stats.csv" | head -n 1)
if [ -n "${STATS}" ]; then
    python3 scripts/gfx906/prof-top.py "${STATS}" 25
else
    echo "no kernel stats produced, last lines of /tmp/check-prof.log:"
    tail -n 20 /tmp/check-prof.log
fi
