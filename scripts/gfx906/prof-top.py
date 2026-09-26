#!/usr/bin/env python3
# Print top kernels from a rocprofv3 kernel_stats.csv.
# usage: scripts/gfx906/prof-top.py <kernel_stats.csv> [n]
import csv
import re
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
rows.sort(key=lambda r: -float(r["TotalDurationNs"]))
total = sum(float(r["TotalDurationNs"]) for r in rows)
print(f"total kernel time: {total/1e6:.1f} ms")
print(f"{'%':>6} {'total ms':>9} {'calls':>7} {'avg us':>8}  kernel")
for r in rows[:n]:
    name = re.sub(r"\(.*$", "", r["Name"].replace("void ", ""), count=0) if "<" not in r["Name"] else r["Name"].replace("void ", "").split(">(")[0] + ">"
    print(f"{float(r['Percentage']):6.2f} {float(r['TotalDurationNs'])/1e6:9.2f} {int(r['Calls']):7d} {float(r['AverageNs'])/1e3:8.2f}  {name[:110]}")
