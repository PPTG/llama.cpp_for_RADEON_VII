#!/usr/bin/env bash
# Token generation at long context (2 GPU layer split), f16 vs q8_0 KV cache (with and without the fused
# Hadamard rotation, and without rotation), plus a kernel profile at the deepest depth.
# The depth is filled with a prompt before each test (a few minutes at 15k). Run it alone.
#
# usage: scripts/gfx906/long-ctx.sh [model.gguf] [build-dir]
# env:   DEPTHS="0,4096,8192,15000"  PROF_DEPTH=15000  SKIP_PROFILE=1
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BIN=${2:-build-dpp}/bin
DEPTHS=${DEPTHS:-"0,4096,8192,15000"}
PROF_DEPTH=${PROF_DEPTH:-${DEPTHS##*,}}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH="${ROCM_PATH}/bin:${PATH}"

run() {
    local name=$1 kv=$2; shift 2
    echo "=== ${name}"
    env "$@" "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer -ctk "${kv}" -ctv "${kv}" -p 0 -n 32 -r 1 -d "${DEPTHS}" -o md 2>&1 \
        | grep -E "tg32|error|failed"
}

run "KV f16"                                        f16
run "KV q8_0 (default)"                             q8_0
run "KV q8_0, rotation not fused (FUSE_FWHT=0)"     q8_0 GGML_CUDA_FUSE_FWHT=0
run "KV q8_0, no rotation (LLAMA_ATTN_ROT_DISABLE)" q8_0 LLAMA_ATTN_ROT_DISABLE=1

[ -n "${SKIP_PROFILE:-}" ] && exit 0

echo
echo "=== kernel profile, tg at depth ${PROF_DEPTH}, q8_0 KV, only the token generation after the prefill"
rm -rf /tmp/long-prof
rocprofv3 --kernel-trace --stats --output-format csv -d /tmp/long-prof -o prof -- \
    "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer -ctk q8_0 -ctv q8_0 -p 0 -n 32 -r 1 -d "${PROF_DEPTH}" > /tmp/long-prof.log 2>&1
TRACE=$(find /tmp/long-prof -name "*kernel_trace.csv" | head -n 1)
if [ -n "${TRACE}" ]; then
    python3 scripts/gfx906/prof-tg.py "${TRACE}" 32 30
else
    echo "no kernel stats produced, last lines of /tmp/long-prof.log:"
    tail -n 20 /tmp/long-prof.log
fi
