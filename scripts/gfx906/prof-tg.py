#!/usr/bin/env python3
# Token generation part of a rocprofv3 kernel_trace.csv: kernels that start after the last prefill kernel
# (mul_mat_q, the batched matmul), in us per token.
# usage: scripts/gfx906/prof-tg.py <kernel_trace.csv> <n tokens> [n kernels]
# env:   SEQ=<kernel name part> also prints the kernels around one call of that kernel (duration, idle gap before it)
import collections
import csv
import os
import sys

if len(sys.argv) < 3:
    sys.exit("usage: prof-tg.py <kernel_trace.csv> <n tokens> [n kernels]  (scripts/gfx906/profile-tg.sh records the trace)")

rows = list(csv.DictReader(open(sys.argv[1])))
n_tok = int(sys.argv[2])
n_top = int(sys.argv[3]) if len(sys.argv) > 3 else 30

def col(*names):
    for k in rows[0].keys():
        if k in names:
            return k
    sys.exit(f"no column {names} in {list(rows[0].keys())}")

c_name  = col("Kernel_Name", "KernelName", "Name")
c_start = col("Start_Timestamp", "BeginNs", "Start")
c_end   = col("End_Timestamp", "EndNs", "End")

def short(name):
    name = name.replace("void ", "")
    return name.split(">(")[0] + ">" if "<" in name else name.split("(")[0]

last_pp = max((int(r[c_start]) for r in rows if short(r[c_name]).startswith("mul_mat_q<")), default=0)
tg = [r for r in rows if int(r[c_start]) > last_pp]
if not tg:
    sys.exit("no kernels after the last prefill kernel")

stats = collections.defaultdict(lambda: [0, 0])
for r in tg:
    s = stats[short(r[c_name])]
    s[0] += int(r[c_end]) - int(r[c_start])
    s[1] += 1

total = sum(s[0] for s in stats.values())
wall  = max(int(r[c_end]) for r in tg) - min(int(r[c_start]) for r in tg)
print(f"tg: {n_tok} tokens, {len(tg)/n_tok:.0f} kernels/token, kernel time {total/1e3/n_tok:.0f} us/token, "
      f"wall {wall/1e3/n_tok:.0f} us/token")
print(f"{'%':>6} {'us/tok':>8} {'calls/tok':>9} {'avg us':>8}  kernel")
for name, (ns, calls) in sorted(stats.items(), key=lambda kv: -kv[1][0])[:n_top]:
    print(f"{100*ns/total:6.2f} {ns/1e3/n_tok:8.1f} {calls/n_tok:9.1f} {ns/1e3/calls:8.2f}  {name[:110]}")

seq = os.environ.get("SEQ")
if seq:
    c_agent = next((k for k in rows[0].keys() if k in ("Agent_Id", "Agent_ID", "gpu-id", "GPU_ID")), None)
    tg.sort(key=lambda r: int(r[c_start]))
    hits = [i for i, r in enumerate(tg) if seq in short(r[c_name])]
    if not hits:
        sys.exit(f"no kernel matching {seq}")
    i0 = hits[len(hits) // 2]
    agent = tg[i0][c_agent] if c_agent else None
    near = [r for r in tg if agent is None or r[c_agent] == agent]
    j0 = next(j for j, r in enumerate(near) if r is tg[i0])
    print(f"\nkernels around one {seq} call (agent {agent}): duration and idle gap before the kernel, us")
    for j in range(max(0, j0 - 6), min(len(near), j0 + 7)):
        r = near[j]
        dur = (int(r[c_end]) - int(r[c_start])) / 1e3
        gap = (int(r[c_start]) - int(near[j - 1][c_end])) / 1e3 if j > 0 else 0.0
        print(f"{'>' if j == j0 else ' '} {dur:8.2f} {gap:8.2f}  {short(r[c_name])[:100]}")
