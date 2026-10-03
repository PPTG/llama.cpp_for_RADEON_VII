#!/usr/bin/env python3
# Tensor types of a GGUF model: count and size per tensor kind and type, and the bytes one generated token reads
# (dense tensors in full, MoE expert tensors by expert_used_count / expert_count). Reads the header only, no numpy.
# usage: scripts/gfx906/gguf-types.py <model.gguf>
import collections
import os
import struct
import sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0"}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

path = sys.argv[1]
f = open(path, "rb")


def rd(fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def rd_str():
    return f.read(rd("Q")).decode("utf-8", "replace")


def rd_val(t):
    if t == 8:
        return rd_str()
    if t == 9:
        et, n = rd("I"), rd("Q")
        return [rd_val(et) for _ in range(n)]
    return rd(SCALAR[t])


if f.read(4) != b"GGUF":
    sys.exit("not a GGUF file")
version, n_tensors, n_kv = rd("I"), rd("Q"), rd("Q")
meta = {}
for _ in range(n_kv):
    k = rd_str()
    meta[k] = rd_val(rd("I"))
infos = []
for _ in range(n_tensors):
    name = rd_str()
    dims = [rd("Q") for _ in range(rd("I"))]
    infos.append((name, rd("I"), rd("Q")))
align = meta.get("general.alignment", 32)
data_start = (f.tell() + align - 1) // align * align
end = os.path.getsize(path) - data_start

# size of a tensor = distance to the next tensor (includes the alignment padding, a few bytes)
offs = sorted(o for _, _, o in infos) + [end]
size = {o: offs[i + 1] - o for i, o in enumerate(offs[:-1])}


def meta_int(suffix):
    for k, v in meta.items():
        if k.endswith(suffix) and isinstance(v, int):
            return v
    return 0


n_exp, n_used = meta_int(".expert_count"), meta_int(".expert_used_count")
frac = n_used / n_exp if n_exp else 1.0

tied = not any(name == "output.weight" for name, _, _ in infos)
c = collections.defaultdict(lambda: [0, 0, 0.0])
for name, t, o in infos:
    kind = ".".join(name.split(".")[2:-1]) if name.startswith("blk.") else name.rsplit(".", 1)[0]
    e = c[(kind, TYPES.get(t, str(t)))]
    b = size[o]
    e[0] += 1
    e[1] += b
    # experts: only the used ones are read per token; embeddings: one row, except a token embedding tied to the output
    if "_exps" in name:
        e[2] += b * frac
    elif kind.endswith("token_embd"):
        e[2] += b if kind == "token_embd" and tied else 0.0
    else:
        e[2] += b

print(f"experts {n_used}/{n_exp}, output head {'tied to token_embd' if tied else 'output.weight'}")
print(f"{'tensor':28s} {'type':8s} {'n':>4s} {'MiB':>9s} {'MiB/token':>10s}")
tot = collections.defaultdict(lambda: [0, 0.0])
for (kind, ty), (n, b, bt) in sorted(c.items(), key=lambda x: (-x[1][2], -x[1][1])):
    print(f"{kind:28s} {ty:8s} {n:4d} {b / 2**20:9.1f} {bt / 2**20:10.1f}")
    tot[ty][0] += b
    tot[ty][1] += bt
print("\nper type:")
for ty, (b, bt) in sorted(tot.items(), key=lambda x: -x[1][1]):
    print(f"{ty:8s} {b / 2**20:9.1f} MiB  {bt / 2**20:8.1f} MiB/token")
print(f"total    {sum(v[0] for v in tot.values()) / 2**20:9.1f} MiB  {sum(v[1] for v in tot.values()) / 2**20:8.1f} MiB/token")
