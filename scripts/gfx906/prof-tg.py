#!/usr/bin/env python3
# Token generation part of a rocprofv3 kernel_trace.csv: kernels that start after the last prefill kernel
# (mul_mat_q, the batched matmul), in us per token.
# usage: scripts/gfx906/prof-tg.py <kernel_trace.csv> <n tokens> [n kernels]
import collections
import csv
import sys

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
