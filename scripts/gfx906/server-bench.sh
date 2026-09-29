#!/usr/bin/env bash
# llama-server token generation for several server configs (same prompt, 512 generated tokens, sampling on).
# Run it alone, no other GPU jobs and no other server on the port.
#
# usage: scripts/gfx906/server-bench.sh [model.gguf] [build-dir] [port]
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
BIN=${2:-build-gfx906}/bin
PORT=${3:-8099}

CONFIGS=(
    "default|"
    "-bs|-bs"
    "-np 1 -c 32768|-np 1 -c 32768"
    "-bs -np 1 -c 32768|-bs -np 1 -c 32768"
)

REQ='{"prompt": "Write a long, detailed story about a lighthouse keeper who finds a strange machine on the shore.", "n_predict": 512, "ignore_eos": true, "cache_prompt": false, "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05}'

run_config() {
    local name=$1 args=$2
    # shellcheck disable=SC2086
    "${BIN}/llama-server" -m "${MODEL}" -ngl 99 -fa on --port "${PORT}" ${args} > /tmp/server-bench.log 2>&1 &
    local pid=$!

    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1 && break
        sleep 1
    done

    local res=()
    for _ in 1 2 3; do
        res+=("$(curl -s "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' -d "${REQ}" \
            | python3 -c 'import json,sys; print("%.2f" % json.load(sys.stdin)["timings"]["predicted_per_second"])' 2>/dev/null || echo "err")")
    done

    kill "${pid}" 2>/dev/null
    wait "${pid}" 2>/dev/null
    printf "%-24s tg t/s: %s\n" "${name}" "${res[*]}"
    grep -m1 -i "backend sampling\|not compatible" /tmp/server-bench.log
}

for C in "${CONFIGS[@]}"; do
    run_config "${C%%|*}" "${C#*|}"
done
