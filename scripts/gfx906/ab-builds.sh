#!/usr/bin/env bash
# A/B of two builds (e.g. compiler flags): tg32 and pp4096 at d0 and d30000 on 2 GPUs with split heads,
# interleaved over rounds to average out drift. Run it alone: other GPU jobs skew the numbers.
#
# usage: scripts/gfx906/ab-builds.sh <build-a> <build-b> [model.gguf]
#   e.g. scripts/gfx906/build.sh build-ilp -DGGML_HIP_MAX_ILP=ON
#        scripts/gfx906/ab-builds.sh build-gfx906 build-ilp
# env:   ROUNDS=3, DEPTHS="0 30000", EXTRA="..." (more llama-bench args)
set -uo pipefail

cd "$(dirname "$0")/../.."

A=${1:?usage: ab-builds.sh <build-a> <build-b> [model.gguf]}
B=${2:?usage: ab-builds.sh <build-a> <build-b> [model.gguf]}
MODEL=${3:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
ROUNDS=${ROUNDS:-3}
DEPTHS=${DEPTHS:-"0 30000"}
EXTRA=${EXTRA:-}

export LLAMA_KV_SPLIT_HEADS=${LLAMA_KV_SPLIT_HEADS:-1}

bench() { # build-dir, -p N, -n N, depth -> t/s
    # shellcheck disable=SC2086
    "$1/bin/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer -ub 1024 -ctk q8_0 -ctv q8_0 \
        -p "$2" -n "$3" -d "$4" -r 3 -o csv ${EXTRA} 2>/dev/null | python3 -c '
import csv, sys
rows = list(csv.DictReader(sys.stdin))
print("%8.2f" % float(rows[-1]["avg_ts"]) if rows else "     err")'
}

printf "%-14s %-8s %s\n" "test" "build" "t/s per round"
for d in ${DEPTHS}; do
    for t in "tg32|0|32" "pp4096|4096|0"; do
        IFS='|' read -r name p n <<< "${t}"
        ra=() rb=()
        for _ in $(seq 1 "${ROUNDS}"); do
            ra+=("$(bench "${A}" "${p}" "${n}" "${d}")")
            rb+=("$(bench "${B}" "${p}" "${n}" "${d}")")
        done
        printf "%-14s %-8s %s\n" "${name} d${d}" "A" "${ra[*]}"
        printf "%-14s %-8s %s\n" "${name} d${d}" "B" "${rb[*]}"
    done
done
echo "A = ${A}, B = ${B}"
