#!/usr/bin/env bash
# llama-server token generation without and with MTP (speculative decoding with the multi-token prediction head):
# t/s and accepted drafts for a story (low acceptance) and code (high acceptance), 512 tokens, 3 runs each.
# Run it alone, no other GPU jobs and no other server on the port.
#
# usage: scripts/gfx906/mtp-bench.sh <model.gguf> <mtp.gguf> [build-dir] [port]
# env:   DRAFT="2 3 4" (--spec-draft-n-max values), TEMP=0.0, EXTRA="..." (more server args)
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:?usage: mtp-bench.sh <model.gguf> <mtp.gguf> [build-dir] [port]}
MTP=${2:?usage: mtp-bench.sh <model.gguf> <mtp.gguf> [build-dir] [port]}
BIN=${3:-build-gfx906}/bin
PORT=${4:-8099}
DRAFT=${DRAFT:-"2 3 4"}
TEMP=${TEMP:-0.0}
EXTRA=${EXTRA:-}
LOG=results-gfx906/mtp-bench
mkdir -p "${LOG}"

export LLAMA_KV_SPLIT_HEADS=${LLAMA_KV_SPLIT_HEADS:-1}
BASE="-m ${MODEL} -ngl 99 -sm layer -fa on -ctk q8_0 -ctv q8_0 -c 16384 -np 1 --port ${PORT} ${EXTRA}"

PROMPTS=(
    "story|Write a long, detailed story about a lighthouse keeper who finds a strange machine on the shore."
    "code|Write a complete Python module with a class that parses, validates and pretty-prints JSON configuration files, with docstrings and unit tests."
)

request() { # prompt text
    python3 - "$1" "${TEMP}" << 'PY'
import json, sys
print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}], "max_tokens": 512, "ignore_eos": True,
                  "temperature": float(sys.argv[2]), "cache_prompt": False}))
PY
}

run_config() { # name, extra server args
    local name=$1 args=$2
    # shellcheck disable=SC2086
    "${BIN}/llama-server" ${BASE} ${args} > "${LOG}/server-${name}.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1 && break
        sleep 1
    done

    for p in "${PROMPTS[@]}"; do
        local kind=${p%%|*} text=${p#*|}
        local req; req=$(request "${text}")
        local out=()
        for _ in 1 2 3; do
            out+=("$(curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' -d "${req}" \
                | python3 -c '
import json, sys
t = json.load(sys.stdin)["timings"]
acc = " %3.0f%%" % (100.0 * t["draft_n_accepted"] / t["draft_n"]) if t.get("draft_n") else ""
print("%6.1f%s" % (t["predicted_per_second"], acc))' 2>/dev/null || echo "   err")")
        done
        printf '%-12s %-6s %s\n' "${name}" "${kind}" "${out[*]}"
    done

    kill "${pid}" 2>/dev/null
    wait "${pid}" 2>/dev/null
}

echo "t/s per run (accepted drafts), temperature ${TEMP}, 512 tokens"
run_config "no-mtp" ""
for n in ${DRAFT}; do
    run_config "mtp-${n}" "--spec-type draft-mtp -md ${MTP} --spec-draft-n-max ${n}"
done
echo "server logs: ${LOG}/"
