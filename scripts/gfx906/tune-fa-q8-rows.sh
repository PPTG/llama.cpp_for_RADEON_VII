#!/usr/bin/env bash
# Token generation with GGML_CUDA_FA_Q8_WAVE_ROWS (KV rows per wave of the head 512 FlashAttention, 64 = default):
# fewer rows split a short context over more blocks. 2 GPUs, split heads, q8_0 KV. Run it alone.
# The values are interleaved over rounds (tg on 2 GPUs drifts between processes); prints the median and the range.
#
# usage: scripts/gfx906/tune-fa-q8-rows.sh [model.gguf] [build-dir]
# env:   ROWS="64 32 16", DEPTHS="512,4096,16384,30000", ROUNDS=4,
#        PREFIX="numactl -N 0 -m 0" (command in front of llama-bench, e.g. to pin it to the NUMA node of the GPUs)
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
BIN=${2:-build-gfx906}/bin
ROWS=${ROWS:-"64 32 16"}
DEPTHS=${DEPTHS:-"512,4096,16384,30000"}
ROUNDS=${ROUNDS:-4}
PREFIX=${PREFIX:-}
export LLAMA_KV_SPLIT_HEADS=${LLAMA_KV_SPLIT_HEADS:-1}

RES=$(mktemp)
for round in $(seq 1 "${ROUNDS}"); do
    for r in ${ROWS}; do
        # shellcheck disable=SC2086
        GGML_CUDA_FA_Q8_WAVE_ROWS=${r} ${PREFIX} "${BIN}/llama-bench" -m "${MODEL}" -ngl 99 -fa 1 -sm layer \
            -ctk q8_0 -ctv q8_0 -p 0 -n 32 -d "${DEPTHS}" -r 2 -o csv 2>/dev/null | python3 -c '
import csv, sys
for row in csv.DictReader(sys.stdin):
    print(sys.argv[1], row["n_depth"], row["avg_ts"])' "${r}" >> "${RES}"
    done
    echo "round ${round}/${ROUNDS} done" >&2
done

python3 - "${RES}" << 'PY'
import collections, statistics, sys
v = collections.defaultdict(list)
depths = []
for line in open(sys.argv[1]):
    r, d, ts = line.split()
    v[(r, d)].append(float(ts))
    if d not in depths:
        depths.append(d)
rows = list(dict.fromkeys(r for r, _ in v))
print("tg32 t/s: median (min-max) over rounds")
print("%-5s" % "rows" + "".join(" %20s" % ("d" + d) for d in depths))
for r in rows:
    print("%-5s" % r + "".join(" %20s" % ("%.1f (%.1f-%.1f)" % (statistics.median(v[(r, d)]), min(v[(r, d)]), max(v[(r, d)])))
                               for d in depths))
PY
rm -f "${RES}"
