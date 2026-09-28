#!/usr/bin/env bash
# Model quality baseline: KL divergence and top token agreement of the fork configurations vs a reference with an f16
# KV cache (llama-perplexity --kl-divergence on wikitext-2). The variants decode one token at a time (-b 1 -ub 1), so
# the token generation kernels are measured (FA q8_0 kernels, LLAMA_KV_SPLIT_HEADS), not the prompt processing ones.
# Same numbers on every run: keep the output as the baseline and compare after changes.
#
# usage: scripts/gfx906/quality.sh [model.gguf] [build-dir]
# env:   CTX=4096 CHUNKS=2 (tokens per chunk and number of chunks; one variant takes ~CTX*CHUNKS*10 ms)
#        VARIANTS="f16 q8-plain q8-tile q8 q8-split q8-b8" (subset to run)
#        SANITY=1 (only the plain perplexity on the CPU vs the GPUs, see below)
# variants: f16 KV | q8_0 KV without the fork kernels and fusions (conversion to f16, no fused Hadamard rotation) |
#           q8_0 KV, FA tile kernel reading q8_0 | default (head 512 wave kernel) | default + LLAMA_KV_SPLIT_HEADS=1 |
#           default with 8 tokens per decode (the kernels of the speculative decoding verification)
# results per model in results-gfx906/quality/<model name>/
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-models/gemma-4-26B-A4B-it-qat-uncensored-heretic-UDmerge-Q4_K_XL.gguf}
BUILD_DIR=${2:-build-dpp}
BIN=${BUILD_DIR}/bin
CTX=${CTX:-4096}
CHUNKS=${CHUNKS:-2}
VARIANTS=${VARIANTS:-"f16 q8-plain q8-tile q8 q8-split q8-b8"}
TEXT=wikitext-2-raw/wiki.test.raw
OUT=results-gfx906/quality/$(basename "${MODEL}" .gguf)
mkdir -p "${OUT}"

if [ ! -x "${BIN}/llama-perplexity" ]; then
    cmake --build "${BUILD_DIR}" --config Release -j"$(nproc)" --target llama-perplexity || exit 1
fi
if [ ! -f "${TEXT}" ]; then
    sh scripts/get-wikitext-2.sh || exit 1
fi

# all fork kernels and fusions off (as far as they can be switched off)
PLAIN=(GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_FUSE_QKV=0 GGML_CUDA_FUSE_NORM_MULTI=0 GGML_CUDA_FUSE_GLU_Q8=0
       GGML_CUDA_FUSE_FWHT=0 GGML_CUDA_FA_TILE_Q8=0 GGML_CUDA_FA_Q8_WAVE=0 GGML_CUDA_FA_COMBINE_SPLIT=0
       GGML_CUDA_STAGED_COPY=0 GGML_SCHED_GPU_SPLIT_SYNC=1)

# SANITY=1: plain perplexity of short chunks on the CPU (upstream code), on the GPUs with the fork changes off and on
# the GPUs by default. If the GPU numbers differ from the CPU one, a GPU kernel is wrong.
if [ -n "${SANITY:-}" ]; then
    S=(-m "${MODEL}" -f "${TEXT}" -c 512 --chunks "${SANITY_CHUNKS:-8}" -ctk f16 -ctv f16)
    ppl() {
        local name=$1; shift
        local log="${OUT}/sanity-${name}.log"
        env "$@" > "${log}" 2>&1
        printf '%-12s %s\n' "${name}" "$(grep -E 'Final estimate' "${log}" | sed -E 's/.*Final estimate: //')"
    }
    echo "=== sanity: perplexity, 512 tokens per chunk, f16 KV"
    ppl cpu         "${BIN}/llama-perplexity" "${S[@]}" -ngl 0
    ppl gpu-plain   "${PLAIN[@]}" "${BIN}/llama-perplexity" "${S[@]}" -ngl 99 -fa on -sm layer
    ppl gpu-no-fa   "${PLAIN[@]}" "${BIN}/llama-perplexity" "${S[@]}" -ngl 99 -fa off -sm layer
    ppl gpu         "${BIN}/llama-perplexity" "${S[@]}" -ngl 99 -fa on -sm layer
    ppl gpu-1       "${BIN}/llama-perplexity" "${S[@]}" -ngl 99 -fa on -sm none
    exit 0
fi

COMMON=(-m "${MODEL}" -f "${TEXT}" -c "${CTX}" --chunks "${CHUNKS}" -ngl 99 -fa on -sm layer)
BASE="${OUT}/base-c${CTX}-n${CHUNKS}.kld"

if [ ! -f "${BASE}" ]; then
    echo "=== reference logits (f16 KV cache, batched): ${BASE}"
    "${BIN}/llama-perplexity" "${COMMON[@]}" -ctk f16 -ctv f16 --kl-divergence-base "${BASE}" > "${OUT}/base.log" 2>&1 \
        || { tail -20 "${OUT}/base.log"; exit 1; }
    grep -E "Final estimate" "${OUT}/base.log"
fi

run() {
    local name=$1; shift
    local log="${OUT}/${name}-c${CTX}-n${CHUNKS}.log"
    env "$@" "${BIN}/llama-perplexity" "${COMMON[@]}" "${KV[@]}" -b "${NB:-1}" -ub "${NB:-1}" \
        --kl-divergence-base "${BASE}" --kl-divergence > "${log}" 2>&1 || { echo "${name}: FAILED, see ${log}"; tail -5 "${log}"; return; }
    printf '%-9s PPL %s | KLD %s | same top %s | max KLD %s\n' "${name}" \
        "$(grep -m1 'Mean PPL(Q) ' "${log}" | sed -E 's/.*: +//')" \
        "$(grep -m1 'Mean    KLD' "${log}" | sed -E 's/.*KLD: +//')" \
        "$(grep -m1 'Same top p:' "${log}" | sed -E 's/.*: +//')" \
        "$(grep -m1 'Maximum KLD' "${log}" | sed -E 's/.*: +//')"
}

echo "=== vs reference (CTX=${CTX}, CHUNKS=${CHUNKS}, one token per decode); lower KLD / higher same top = closer to f16"
for V in ${VARIANTS}; do
    case ${V} in
        f16)      KV=(-ctk f16 -ctv f16);   run f16      GGML_CUDA_FA_Q8_WAVE=1 ;;
        q8-plain) KV=(-ctk q8_0 -ctv q8_0); run q8-plain "${PLAIN[@]}" ;;
        q8-tile)  KV=(-ctk q8_0 -ctv q8_0); run q8-tile  GGML_CUDA_FA_Q8_WAVE=0 ;;
        q8)       KV=(-ctk q8_0 -ctv q8_0); run q8       GGML_CUDA_FA_Q8_WAVE=1 ;;
        q8-split) KV=(-ctk q8_0 -ctv q8_0); run q8-split GGML_CUDA_FA_Q8_WAVE=1 LLAMA_KV_SPLIT_HEADS=1 ;;
        q8-b8)    KV=(-ctk q8_0 -ctv q8_0); NB=8 run q8-b8 GGML_CUDA_FA_Q8_WAVE=1 ;;
        *) echo "unknown variant ${V}" ;;
    esac
done
