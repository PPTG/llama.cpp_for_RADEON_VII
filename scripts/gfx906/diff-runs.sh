#!/usr/bin/env bash
# Runs llama-perplexity twice with the same settings (one token per decode) and compares the logits token by token:
# prints the first differing positions and the number of differing tokens per block of positions. For finding where
# a non-deterministic decode starts to diverge. Extra env (e.g. LLAMA_KV_SPLIT_HEADS=1) is passed through.
#
# usage: scripts/gfx906/diff-runs.sh [model.gguf] [build-dir]
# env:   CTX=1024, FIRST=1 (first token with logits, LLAMA_PPL_FIRST; logits read after each decode from there on),
#        KV="-ctk q8_0 -ctv q8_0", BLOCK=64
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
BIN=${2:-build-gfx906}/bin
CTX=${CTX:-1024}
FIRST=${FIRST:-1}
KV=${KV:-"-ctk q8_0 -ctv q8_0"}
BLOCK=${BLOCK:-64}
TEXT=${TEXT:-wikitext-2-raw/wiki.test.raw}
OUT=$(mktemp -d)

for r in a b; do
    # shellcheck disable=SC2086
    LLAMA_PPL_FIRST=${FIRST} "${BIN}/llama-perplexity" -m "${MODEL}" -f "${TEXT}" -c "${CTX}" --chunks 1 -ngl 99 -fa on \
        -sm layer ${KV} -b 1 -ub 1 --kl-divergence-base "${OUT}/${r}.kld" > "${OUT}/${r}.log" 2>&1 \
        || { tail -5 "${OUT}/${r}.log"; exit 1; }
done

python3 - "${OUT}/a.kld" "${OUT}/b.kld" "${FIRST}" "${BLOCK}" << 'PY'
import struct, sys
a, b = open(sys.argv[1], 'rb').read(), open(sys.argv[2], 'rb').read()
first, block = int(sys.argv[3]), int(sys.argv[4])
n_ctx, = struct.unpack_from('<i', a, 8)
n_vocab, n_chunk = struct.unpack_from('<ii', a, 12)
off = 20 + 4*n_chunk*n_ctx
nv = 2*((n_vocab + 1)//2) + 4
sz = 2*nv
n = (len(a) - off) // sz
diff = [a[off + i*sz:off + (i+1)*sz] != b[off + i*sz:off + (i+1)*sz] for i in range(n)]
pos = [first + i for i in range(n) if diff[i]]
print("tokens compared: %d, differing: %d" % (n, len(pos)))
print("first differing positions:", pos[:20])
for s in range(0, n, block):
    c = sum(diff[s:s + block])
    if c:
        print("positions %5d-%5d: %d differ" % (first + s, first + min(s + block, n) - 1, c))
PY
rm -rf "${OUT}"
