#!/usr/bin/env bash
# Token generation at long context (2 GPU layer split), f16 vs q8_0 KV cache, plus a kernel profile at 64k.
# The depth is filled with a prompt before each test, so this takes a while (~30 min). Run it alone.
#
# usage: scripts/gfx906/long-ctx.sh [model.gguf] [build-dir]
# env:   DEPTHS="0,32768,65536,120000"  SKIP_PROFILE=1
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BIN=${2:-build-dpp}/bin
DEPTHS=${DEPTHS:-"0,32768,65536,120000"}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH="${ROCM_PATH}/bin:${PATH}"

run() {
    local name=$1 kv=$2; shift 2
    echo "=== ${name}"
    env "$@" "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer -ctk "${kv}" -ctv "${kv}" -p 0 -n 32 -r 1 -d "${DEPTHS}" -o md 2>&1 \
        | grep -E "tg32|error|failed"
}

run "KV f16"                                   f16
run "KV q8_0, FA tile reads q8_0 (new)"        q8_0
run "KV q8_0, FA tile converts to f16 (old)"   q8_0 GGML_CUDA_FA_TILE_Q8=0

[ -n "${SKIP_PROFILE:-}" ] && exit 0

echo
echo "=== kernel profile, tg at depth 65536, f16 KV (prefill kernels are included, tg FA kernels have ncols1 = 1)"
rm -rf /tmp/long-prof
rocprofv3 --kernel-trace --stats --output-format csv -d /tmp/long-prof -o prof -- \
    "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer -p 0 -n 32 -r 1 -d 65536 > /tmp/long-prof.log 2>&1
STATS=$(find /tmp/long-prof -name "*kernel_stats.csv" | head -n 1)
if [ -n "${STATS}" ]; then
    python3 scripts/gfx906/prof-top.py "${STATS}" 25
else
    echo "no kernel stats produced, last lines of /tmp/long-prof.log:"
    tail -n 20 /tmp/long-prof.log
fi
