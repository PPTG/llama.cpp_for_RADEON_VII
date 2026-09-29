#!/usr/bin/env bash
# All-in-one for gfx906: build variants, check correctness, benchmark token generation, profile kernels.
# Results go to results-gfx906/<date>/ and are packed into results-gfx906-<date>.tar.gz.
#
# usage (from repo root):
#   scripts/gfx906/run-all.sh [model.gguf]
#
# env:
#   VARIANTS="nodpp dpp nw1 nw4"  build variants to test (default: all)
#   SKIP_BUILD=1                   reuse existing build dirs
#   SKIP_TESTS=1                   skip test-backend-ops
#   SKIP_BENCH=1                   skip llama-bench
#   SKIP_PROFILE=1                 skip rocprofv3
set -uo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-${MODEL:-models/gemma-4-26B-A4B-it-Q4_0.gguf}}
VARIANTS=${VARIANTS:-"nodpp dpp nw1 nw4"}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH="${ROCM_PATH}/bin:${PATH}"

STAMP=$(date +%Y%m%d-%H%M%S)
OUT=results-gfx906/${STAMP}
mkdir -p "${OUT}"

log() { echo -e "\n### $*" | tee -a "${OUT}/summary.txt"; }

if [ ! -f "${MODEL}" ]; then
    echo "model not found: ${MODEL}"
    exit 1
fi

variant_flags() {
    case "$1" in
        nodpp) echo "-DGGML_HIP_NO_DPP_REDUCE=ON" ;;
        dpp)   echo "" ;;
        nw1)   echo "-DGGML_HIP_MMVQ_GCN_NWARPS=1" ;;
        nw4)   echo "-DGGML_HIP_MMVQ_GCN_NWARPS=4" ;;
        nw8)   echo "-DGGML_HIP_MMVQ_GCN_NWARPS=8" ;;
        *)     echo "unknown variant: $1" >&2; exit 1 ;;
    esac
}

# system info
{
    echo "git: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
    echo "model: ${MODEL}"
    echo "variants: ${VARIANTS}"
    hipconfig --version 2>/dev/null && echo
    rocminfo 2>/dev/null | grep -E "Marketing Name|Name: +gfx" | sort -u
    rocm-smi --showclocks --showpower --showmeminfo vram 2>/dev/null
    uname -a
} > "${OUT}/sysinfo.txt" 2>&1
cat "${OUT}/sysinfo.txt"

# 1. build
if [ -z "${SKIP_BUILD:-}" ]; then
    CCACHE_ARGS=()
    if command -v ccache >/dev/null; then
        CCACHE_ARGS=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache)
    fi
    for V in ${VARIANTS}; do
        log "build ${V}"
        # shellcheck disable=SC2046
        if ! scripts/gfx906/build.sh "build-${V}" $(variant_flags "${V}") "${CCACHE_ARGS[@]}" > "${OUT}/build-${V}.log" 2>&1; then
            echo "BUILD FAILED for ${V}, see ${OUT}/build-${V}.log" | tee -a "${OUT}/summary.txt"
            tail -n 30 "${OUT}/build-${V}.log"
            exit 1
        fi
    done
fi

# 2. correctness vs CPU (only the DPP build, nw* only change MMVQ so check MUL_MAT for them)
if [ -z "${SKIP_TESTS:-}" ]; then
    for V in ${VARIANTS}; do
        case "${V}" in
            dpp)   OPS="MUL_MAT MUL_MAT_ID RMS_NORM NORM SOFT_MAX FLASH_ATTN_EXT ARGSORT TOP_K SUM_ROWS GROUP_NORM L2_NORM" ;;
            nw*)   OPS="MUL_MAT MUL_MAT_ID" ;;
            *)     continue ;;
        esac
        log "test-backend-ops ${V}"
        for OP in ${OPS}; do
            RES=$("build-${V}/bin/test-backend-ops" -b ROCm0 -o "${OP}" 2>&1 | tee -a "${OUT}/test-${V}.log" | grep -E "tests passed|Backend ROCm0:" | tail -n 1)
            echo "${V} ${OP}: ${RES}" | tee -a "${OUT}/summary.txt"
        done
        if grep -q "FAIL" "${OUT}/test-${V}.log"; then
            echo "!!! ${V}: some tests FAILED, see ${OUT}/test-${V}.log" | tee -a "${OUT}/summary.txt"
        fi
    done
fi

# 3. token generation benchmark
if [ -z "${SKIP_BENCH:-}" ]; then
    log "llama-bench tg"
    for V in ${VARIANTS}; do
        echo "=== ${V}" | tee -a "${OUT}/summary.txt"
        "build-${V}/bin/llama-bench" -m "${MODEL}" -ngl 99 -sm none -mg 0 -fa 1 -p 0 -n 128 -r 5 -d 0,4096 -o md 2> "${OUT}/bench-${V}.err" \
            | tee "${OUT}/bench-${V}.md" | tee -a "${OUT}/summary.txt"
    done
fi

# 4. kernel profile of the default build
if [ -z "${SKIP_PROFILE:-}" ]; then
    PV=dpp
    [[ " ${VARIANTS} " == *" dpp "* ]] || PV=$(echo "${VARIANTS}" | awk '{print $1}')
    log "rocprofv3 (${PV})"
    if command -v rocprofv3 >/dev/null; then
        rocprofv3 --kernel-trace --stats --output-format csv -d "${OUT}/prof" -o prof -- \
            "build-${PV}/bin/llama-bench" -m "${MODEL}" -ngl 99 -sm none -mg 0 -fa 1 -p 0 -n 32 -r 1 > "${OUT}/prof.log" 2>&1
        STATS=$(find "${OUT}/prof" -name "*kernel_stats.csv" | head -n 1)
        if [ -n "${STATS}" ]; then
            python3 scripts/gfx906/prof-top.py "${STATS}" 30 | tee -a "${OUT}/summary.txt"
        else
            echo "no kernel stats produced, see ${OUT}/prof.log" | tee -a "${OUT}/summary.txt"
            tail -n 20 "${OUT}/prof.log" | tee -a "${OUT}/summary.txt"
        fi
        # drop the big per-dispatch trace, keep stats
        find "${OUT}/prof" -name "*kernel_trace.csv" -size +20M -delete
    else
        echo "rocprofv3 not found, skipped" | tee -a "${OUT}/summary.txt"
    fi
fi

tar czf "results-gfx906-${STAMP}.tar.gz" -C results-gfx906 "${STAMP}"
log "done: ${OUT}/summary.txt, archive: results-gfx906-${STAMP}.tar.gz"
