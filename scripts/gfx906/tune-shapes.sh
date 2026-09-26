#!/usr/bin/env bash
# Time MMVQ (batch size 1) for every Gemma 4 tg matrix shape and every GGML_HIP_MMVQ_VARIANT.
# Prints us per op, lower is better. Run it alone, no other GPU jobs.
#
# usage: scripts/gfx906/tune-shapes.sh [build-dir]
set -uo pipefail

cd "$(dirname "$0")/../.."

BIN=${1:-build-dpp}/bin
VARIANTS=${VARIANTS:-"0 1 2 3 4 5 6 7 8 10 11"}
TMP=$(mktemp -d)

for V in ${VARIANTS}; do
    echo "variant ${V}..." >&2
    GGML_HIP_MMVQ_VARIANT=${V} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} "${BIN}/test-backend-ops" perf -b ROCm0 \
        -o "MUL_MAT.*" -p "type_a=q4_0,type_b=f32,.*n=1,k=(2816|4096|8192|2112|704)," 2>/dev/null \
        | grep "us/run" > "${TMP}/v${V}.txt"
done

python3 - "${TMP}" ${VARIANTS} <<'PY'
import re, sys
tmp, variants = sys.argv[1], sys.argv[2:]
rows = {}
order = []
for v in variants:
    for line in open(f"{tmp}/v{v}.txt"):
        op = "ID " if "MUL_MAT_ID" in line else "MM "
        m = re.search(r",m=(\d+),n=1,k=(\d+)", line)
        t = re.search(r"([\d.]+) us/run", line)
        if not m or not t:
            continue
        key = f"{op}{m.group(1)}x{m.group(2)}"
        if key not in rows:
            rows[key] = {}
            order.append(key)
        rows[key][v] = float(t.group(1))
print(f"{'shape (rows x k)':20}" + "".join(f"{'v' + v:>9}" for v in variants) + "   best")
for key in order:
    r = rows[key]
    best = min(r, key=r.get)
    print(f"{key:20}" + "".join(f"{r.get(v, float('nan')):9.1f}" for v in variants) + f"   v{best}")
PY
rm -rf "${TMP}"
