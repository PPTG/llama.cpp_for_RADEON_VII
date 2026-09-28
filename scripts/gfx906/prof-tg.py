#!/usr/bin/env python3
# Token generation part of a rocprofv3 kernel_trace.csv: kernels that start after the last prefill kernel
# (mul_mat_q, the batched matmul), in us per token.
# usage: scripts/gfx906/prof-tg.py <kernel_trace.csv> <n tokens> [n kernels]
# env:   SEQ=<kernel name part> also prints the kernels around one call of that kernel (duration, idle gap before it)
#        GAPS=1 prints the idle time of each GPU between kernels, by the kernel that follows the gap, and the time when
#        no GPU runs a kernel (the real loss of a split over GPUs: copies through the host, syncs, launches)
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

if os.environ.get("GAPS"):
    c_agent = next((k for k in rows[0].keys() if k in ("Agent_Id", "Agent_ID", "gpu-id", "GPU_ID")), None)
    by_agent = collections.defaultdict(list)
    for r in sorted(tg, key=lambda r: int(r[c_start])):
        by_agent[r[c_agent] if c_agent else "all"].append(r)
    print("\nidle gaps between kernels of the same GPU (gaps > 1 ms, e.g. waiting for the other GPU, are left out)")
    for agent, rs in sorted(by_agent.items()):
        gaps = collections.defaultdict(lambda: [0, 0, 0])
        for prev, r in zip(rs, rs[1:]):
            gap = int(r[c_start]) - int(prev[c_end])
            if 0 < gap < 1000000:
                g = gaps[(short(prev[c_name])[:45], short(r[c_name])[:45])]
                g[0] += gap
                g[1] += 1
                g[2] = max(g[2], gap)
        tot = sum(g[0] for g in gaps.values())
        print(f"{agent}: idle {tot/1e3/n_tok:.0f} us/token")
        print(f"{'us/tok':>8} {'n/tok':>6} {'avg us':>7} {'max us':>7}  after -> before")
        for (a, b), (ns, n, mx) in sorted(gaps.items(), key=lambda kv: -kv[1][0])[:12]:
            print(f"{ns/1e3/n_tok:8.1f} {n/n_tok:6.1f} {ns/1e3/n:7.2f} {mx/1e3:7.2f}  {a} -> {b}")

    # time when no GPU runs anything, attributed to the kernels around the gap
    ev = sorted(tg, key=lambda r: int(r[c_start]))
    both = collections.defaultdict(lambda: [0, 0, 0])
    end_max, last = int(ev[0][c_end]), ev[0]
    for r in ev[1:]:
        st = int(r[c_start])
        if st > end_max:
            gap = st - end_max
            ag = lambda x: (x[c_agent].replace("Agent ", "") + ":") if c_agent else ""
            g = both[(ag(last) + short(last[c_name])[:42], ag(r) + short(r[c_name])[:42])]
            g[0] += gap
            g[1] += 1
            g[2] = max(g[2], gap)
        if int(r[c_end]) > end_max:
            end_max, last = int(r[c_end]), r
    tot = sum(g[0] for g in both.values())
    print(f"\nno GPU busy: {tot/1e3/n_tok:.0f} us/token of wall {wall/1e3/n_tok:.0f} us/token")
    print(f"{'us/tok':>8} {'n/tok':>6} {'avg us':>7} {'max us':>7}  last kernel before -> first kernel after (agent:)")
    for (a, b), (ns, n, mx) in sorted(both.items(), key=lambda kv: -kv[1][0])[:15]:
        print(f"{ns/1e3/n_tok:8.1f} {n/n_tok:6.1f} {ns/1e3/n:7.2f} {mx/1e3:7.2f}  {a} -> {b}")
