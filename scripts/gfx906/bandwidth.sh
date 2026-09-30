#!/usr/bin/env bash
# Copy bandwidth / latency host <-> GPU and GPU <-> GPU (scripts/gfx906/bandwidth.hip), once with the default placement
# and once bound to every NUMA node (the host buffer and thread on that node), to see which socket the GPUs hang off
# and what the socket links (QPI / UPI) deliver.
#
# usage: scripts/gfx906/bandwidth.sh [buffer MB] [small copy bytes]
set -euo pipefail

cd "$(dirname "$0")/../.."

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
OUT=results-gfx906/bandwidth
mkdir -p "${OUT}"

"${ROCM_PATH}/bin/hipcc" -O2 -std=c++17 scripts/gfx906/bandwidth.hip -o "${OUT}/bandwidth"

echo "=== GPU NUMA nodes"
for d in /sys/class/drm/card*/device; do
    [ -f "$d/numa_node" ] || continue
    vendor=$(cat "$d/vendor" 2>/dev/null || true)
    [ "${vendor}" = "0x1002" ] || continue
    echo "$(basename "$(readlink -f "$d")"): node $(cat "$d/numa_node")"
done

echo
echo "=== default placement"
"${OUT}/bandwidth" "$@"

if command -v numactl > /dev/null; then
    for node in $(numactl -H | awk '/^node [0-9]+ cpus:/ { print $2 }'); do
        echo
        echo "=== bound to NUMA node ${node}"
        numactl --cpunodebind="${node}" --membind="${node}" "${OUT}/bandwidth" "$@"
    done
fi
