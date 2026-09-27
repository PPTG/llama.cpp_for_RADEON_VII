llama.cpp for Radeon VII / Radeon Pro VII (gfx906)

This fork carries HIP/ROCm changes aimed at token generation speed on gfx906 (Radeon VII, Radeon Pro VII, Instinct MI50/MI60).

## Quick start

```bash
scripts/gfx906/run-all.sh models/<model>.gguf
```

Builds the variants (`nodpp` = upstream reductions, `dpp`, `nw1`, `nw4`), runs `test-backend-ops` against the CPU, runs `llama-bench` and profiles kernels with `rocprofv3`. Output: `results-gfx906/<date>/summary.txt` and a `.tar.gz` archive.

## Results

2x Radeon Pro VII (PCIe, no bridge), ROCm 7.1, Gemma 4 26B A4B Q4_0 (UD Q4_K_XL, merged gate_up), `llama-bench -fa 1`,
upstream = 271158c6e, fork = fc331626b, all fork options at their defaults (`scripts/gfx906/final-bench.sh`):

| test           | 1 GPU upstream | 1 GPU fork | gain | 2 GPU layer upstream | 2 GPU layer fork | gain |
|----------------|---------------:|-----------:|-----:|---------------------:|-----------------:|-----:|
| tg128          |  96.6 | 115.9 | +20% |  82.8 | 107.9 | +30% |
| tg128 @ d4096  |  88.5 | 106.9 | +21% |  75.2 |  98.1 | +31% |
| tg128 @ d16384 |  86.2 | 101.8 | +18% |  75.9 |  92.7 | +22% |
| pp512          |  1398 |  1393 |   0% |  1394 |  1415 |  +2% |

Where the tg gain comes from (1 GPU / 2 GPU, from the A/B runs):

- DPP warp reductions: +3-7%
- staged copy between GPUs (2 GPU only): +14%
- MMVQ 1 warp x 8 rows on GCN: +4-8%
- Q8_1 activation cache (Q/K/V and gate/up quantized once): +1.5-3%

Round 2 (kernel count reduction, `rocm-smi --setperflevel high`, tg128, `scripts/gfx906/ab.sh`):

| Config                                   | 1 GPU | 2 GPU layer |
|------------------------------------------|------:|------------:|
| all on (default, 2514c9eb7)              | 124.7 | 115.2 |
| without Q/K/V multi MMVQ                 | 119.7 | 110.4 |
| without any fusion (`GGML_CUDA_DISABLE_FUSION=1`) | 93.2 | 86.6 |
| upstream MMVQ geometry                   | 105.6 |  99.0 |

New in round 2:

- Q/K/V (and shared gate/up) as one MMVQ launch: +4-5%
- rms_norm + scale + mul fused (MoE router input)
- expert scale looked up in the MoE weighted reduction (no repeat + get_rows)
- `rocm-smi --setperflevel high`: +5-9% on 2 GPUs (the idle GPU of the layer split does not clock down), no change on 1 GPU

Round 3 (0aa034c90, `rocm-smi --setperflevel high`, tg128, mean of 2 `scripts/gfx906/ab.sh` rounds):

| Config                                   | 1 GPU | 2 GPU layer |
|------------------------------------------|------:|------------:|
| all on (default)                         | 127.1 | 117.6 |
| without Q/K/V multi MMVQ                 | 118.2 | 110.7 |
| without glu + q8_1 fusion                | 127.9 | 115.8 |
| without multi output rms_norm            | 125.9 | 114.1 |
| without any fusion                       |  93.3 |  88.9 |
| upstream MMVQ geometry                   | 107.6 |  99.8 |

Q/K/V as one launch is the biggest single fusion (+6-8%). glu + q8_1 is neutral on 1 GPU (fewer launches, same work) and kept on since it is bit exact. All fork fusions give the same text as the unfused ops (`scripts/gfx906/check.sh`).

Round 4 (long context, q8_0 KV cache, 2 GPU layer split, `rocm-smi --setperflevel high`):

- glu + q8_1 fusion fixed: it never ran before (`ggml_can_fuse` requires equal shapes). tg128 f16 KV: 114.8 -> 122.9 t/s.
- q8_0 KV with head size 256 (SWA layers) on the FA tile kernel that reads q8_0 directly: 44 -> 34 us per call.
- Hadamard rotation of the quantized KV cache fused into the cache store and the attn_output quantization: +5% tg at depth 0.
- FA combine kernel with more blocks: -14 us per global layer.

| tg32 (`scripts/gfx906/long-ctx.sh`, before the glu fix) | d0 | d4096 | d8192 | d15000 |
|---------------------------------------------------------|---:|------:|------:|-------:|
| f16 KV                                                   | 112.6 | 104.4 | 101.3 | 98.1 |
| q8_0 KV, round 3                                         |  97.8 |  90.7 |  87.0 | 85.4 |
| q8_0 KV, round 4                                         | 101.0 |  95.6 |  95.7 | 95.9 |

After the glu fix and the tuned q8_0 FA variants (`GGML_CUDA_FA_Q8_CFG` defaults): q8_0 KV tg32 104.0 (d0), 99.0 (d15000),
96.2 (d32768); the SWA layers on the tile kernel are +10% at d32768 over the vector kernel (87.1).

Both GPUs at the same time (2 GPU layer split, q8_0 KV, tg32, mean of 3, `scripts/gfx906/split-heads.sh`):
`LLAMA_KV_SPLIT_HEADS=1` puts the second KV head of the 5 global layers (and its half of their KV cache) on the other
GPU, which computes that half of the attention while the device of the layer computes the other half.

| tg32              | d0    | d15000 | d32768 | d65536 |
|-------------------|------:|-------:|-------:|-------:|
| layer split       | 110.0 | ~97    | ~96.5  | 85.5   |
| + split KV heads  | 106.8 | 98.9   | 97.4   | 91.8   |

On the HP DL580 test machine the two GPUs are on different CPU sockets (NUMA nodes 0 and 3): no peer access, every copy
between the GPUs goes through host memory (`GGML_CUDA_PEER_COPY` stays off), and binding the process with `numactl` to
either node made no difference (the socket interconnect has the same speed between all nodes) (d0 / d15000: 108.5 / 99.4 default, 106.6 / 99.8 node 0, 109.0 / 99.9 node 3).

The gain grows with the context (the attention of a global layer is ~130 us at 15k, ~780 us at 128k), the fixed cost of
the extra device switches is ~3% at short context. `-sm tensor` (all matrices split, 2 AllReduce per layer) was slower:
93.0 / 80.3 t/s at d0 / d32768. Token generation is bound by ~900 small kernels per token (~5 us minimum each), which a
split over two GPUs does not reduce, only big kernels (long context attention) gain from a second GPU.

FA of the global layers with the q8_0 KV cache (head 512, GQA 8, token generation), `GGML_CUDA_FA_Q8_WAVE` kernel vs
the tile kernel (`scripts/gfx906/fa-q8-wave.sh`, us per call; nh = KV heads, 1 = one GPU with `LLAMA_KV_SPLIT_HEADS=1`):

| KV rows | nh | tile | wave |
|--------:|---:|-----:|-----:|
|    4096 |  2 |   46 |   46 |
|   16384 |  1 |   79 |   71 |
|   16384 |  2 |  119 |  114 |
|   65536 |  1 |  222 |  181 |
|   65536 |  2 |  404 |  299 |
|  131072 |  2 |  775 |  540 |

tg32 (2 GPUs, `LLAMA_KV_SPLIT_HEADS=1`) d0 / d15000: 107.5 / 101.1 tile, 109.1 / 103.4 wave. ROCm 7.1: 128 VGPRs,
2 waves per SIMD, no spills (`scripts/gfx906/fa-q8-wave-regs.sh`).

`-sm tensor` (with `GGML_CUDA_P2P=1`) was 91 t/s before the MMVQ work: over PCIe without a bridge the per layer AllReduce costs more than it saves.

llama-server (2 GPU layer split, 512 tokens with sampling, `scripts/gfx906/server-bench.sh`):

| Server args             | tg t/s |
|-------------------------|-------:|
| default                 |  96.4  |
| `-bs`                   | 102.7  |
| `-np 1 -c 32768`        |  98.7  |
| `-bs -np 1 -c 32768`    | 103.1  |

Recommended: `rocm-smi --setperflevel high` and `llama-server -m model.gguf -ngl 99 -fa on -bs`. Backend sampling runs the whole default sampler chain on the GPU. The server falls back to CPU sampling for requests with a grammar (JSON schema, tool calls), with logprobs, or with DRY/XTC/typical/top-n-sigma enabled.

Compare the multi GPU variants and check their output: `scripts/gfx906/multi-gpu.sh`.

## Build

```bash
scripts/gfx906/build.sh                      # -> build-gfx906/
```

Build options specific to this fork:

| CMake option                    | Default | Description |
|---------------------------------|---------|-------------|
| `GGML_HIP_NO_DPP_REDUCE`        | `OFF`   | Disable the DPP based warp reductions (fall back to upstream `__shfl_xor`). |
| `GGML_HIP_MMVQ_GCN_NWARPS`      | `2`     | Warps per block for the quantized mat-vec kernel (MMVQ) at batch size 1 on GCN/CDNA. Try 1, 4, 8. |

Runtime options:

| Env variable               | Description |
|----------------------------|-------------|
| `GGML_CUDA_STAGED_COPY=0`   | Default on for HIP: copy tensors between GPUs through pinned host memory with async D2H + H2D (no host stall), instead of the HIP runtime peer copy. `0` goes back to the peer copy. |
| `GGML_CUDA_PEER_COPY=1`     | Experimental: copies up to 1 MB between the GPUs (layer split boundaries, `LLAMA_KV_SPLIT_HEADS`) by a kernel that writes into the memory of the other GPU (peer access over PCIe) instead of D2H + H2D through pinned host memory. Test with `scripts/gfx906/peer-copy.sh`. |
| `GGML_CUDA_AR_P2P=1`        | `-sm tensor` with 2 GPUs: the AllReduce kernel writes directly into the VRAM of the peer and polls local VRAM, instead of staging through host memory. Needs P2P over PCIe. |
| `GGML_CUDA_MMVQ_Q8_CACHE=0` | Default on for HIP: quantize the activations (src1) once for consecutive MMVQ ops that use the same src1 (Q/K/V, gate/up). `0` quantizes for every op. |
| `GGML_HIP_MMVQ_VARIANT=<0..11>` | MMVQ geometry for batch size 1, warps per block x rows per block: 1: 1x1, 2: 1x2, 3: 1x4, 4: 2x1, 5: 2x2, 6: 4x1, 7: 4x2, 8: 1x8, 9: 1x16, 10: 2x4, 11: 2x8. Unset: 8 on GCN (+10% tg on Pro VII). 0: upstream heuristics. Sweep with `scripts/gfx906/tune-mmvq.sh`. |
| `GGML_CUDA_FUSE_GATE_UP=1`  | Fuse merged gate_up weights (e.g. `ffn_gate_up_exps`) + glu into one MMVQ kernel, and add the alloc deps that let mm + glu fusions happen. Default off for HIP: ~5% slower on gfx906 (2x registers). |
| `GGML_CUDA_NORM_BLOCK_THRESHOLD=<n>` | RMS norm rows shorter than n use 256 threads per block, longer rows 1024 (default n = 1024). |
| `GGML_CUDA_FUSE_QKV=0`      | Default on for HIP: consecutive mul_mat with the same src1 (Q, K, V) run as one MMVQ launch; graph_optimize moves them next to each other. |
| `GGML_CUDA_FUSE_GLU_Q8=0`   | Default on for HIP: a glu that only feeds a MMVQ mul_mat (ffn_down) is computed and quantized to q8_1 in one kernel. |
| `GGML_CUDA_FUSE_NORM_MULTI=0` | Default on for HIP: up to 3 rms_norm -> [scale ->] mul chains on the same input run as one kernel; graph_optimize moves them next to each other. |
| `GGML_CUDA_FA_TILE_Q8=0`    | Default on: the FA tile kernel (head size 256/512, token generation) reads a q8_0 K/V cache directly instead of converting the whole cache to f16 on every call. |
| `GGML_CUDA_FUSE_FWHT=0`    | Default on for HIP: with a quantized KV cache llama.cpp rotates Q/K/V and the attention output with a Hadamard transform (fwht). The rotation of K and V runs in the kernel that stores them into a q8_0 cache, the rotation of the attention output in the q8_1 quantization for `attn_output`. `LLAMA_ATTN_ROT_DISABLE=1` turns the rotation off (llama.cpp option, slightly lower KV quality). |
| `GGML_CUDA_FA_Q8_VEC=1`    | q8_0 KV cache, head size 256, token generation: use the FA vector kernel. The default on HIP is the tile kernel, which reads q8_0 directly (Gemma 4 SWA layers 48 -> 39 us or less, GQA 16 2-3x faster). Compare with `scripts/gfx906/fa-tune.sh`. |
| `GGML_CUDA_FA_COMBINE_SPLIT=0` | Default on: the kernel that combines the partial FA results of long KV caches runs 64 values per block (more blocks for few heads) instead of one block per head. Same result. |
| `GGML_CUDA_FA_Q8_CFG=0/2/3` | Tuning variants (2: 128 K columns per step, 3: occupancy 3) of the FA tile kernel that reads q8_0, for head size 512 with GQA 8 and head size 256 with GQA 2. Default on HIP: 3 for head 512 (occupancy 3; 128k: 985 -> 846 us, with the prefetch 782 us), 0 for head 256 with the prefetch (34 -> 24 us; 2 without it). 0 = upstream config. Compare with `scripts/gfx906/fa-q8-cfg.sh`. |
| `LLAMA_SAMPLING_FAST_TOP_K=0` | Default on: CPU sampling (llama-server with a grammar/tool calls, reasoning budget, or without `-bs`) runs the sampler chain on the top (top_k + penalty_last_n + number of negative logit biases) logits plus the tokens with a positive bias instead of all 262144: same tokens, ~0.3 -> ~0.1 ms per token. A token rejected by the grammar is resampled with the grammar applied to the top 1024 + top_k + margin logits when the k-th logit after the grammar is not smaller than the smallest of them (a JSON string: 18 -> 0.8 ms per token), else to all logits. |
| `LLAMA_SAMPLING_STATS=1` | Prints the time of the sampling steps (first try, grammar check, resampling, accept) with the performance summary. `scripts/gfx906/sampling-fast.sh` (160 tokens, penalties, json_arr grammar): sampling 600 -> 83 ms with the grammar, 103 -> 42 ms without, same text. |
| `LLAMA_GRAMMAR_PREFILTER=0` | Default on: the grammar rejects a token whose first code point matches no grammar stack, or whose second code point matches no stack after the first, without decoding the whole token (first two code points of every token computed once per grammar; json_arr states with few valid tokens: 17-42 -> 1-7 ms for all 262144 tokens). Same valid tokens (8 grammars, 320 steps). Only when the samplers in front of top-k just lower logits (penalties with non-negative parameters, DRY and top-n-sigma off). Compare with `scripts/gfx906/sampling-fast.sh`. |
| `GGML_CUDA_FA_Q8_WAVE=0`   | Default on for GCN/CDNA (64 wide waves): token generation with a q8_0 K/V cache, head size 512, GQA 8 (Gemma 4 global layers) runs on a kernel where one wave computes all 8 Q heads for 64 KV rows: K rows staged in LDS by coalesced loads and dotted with Q quantized to int8 (v_dot4, as the CPU does for a q8_0 K), V rows read directly, 8 values per lane. Every K/V byte is loaded once instead of being converted and read by 8 warps. `0` uses the tile kernel. Compare with `scripts/gfx906/fa-q8-wave.sh`. |
| `GGML_CUDA_FA_Q8_PIPE=0`   | Default on: the FA tile kernel that reads q8_0 loads the next K/V chunk into registers while it computes the current one (head 512 variants). `0` loads and computes one after the other. |
| `LLAMA_KV_SPLIT_HEADS=1`   | Experimental, 2+ GPUs with flash attention: the KV heads of the non-SWA layers are split over two GPUs (half of the KV cache of these layers on the other GPU) and the attention of each half runs on its GPU in parallel. Saved KV cache states (llama-server prompt cache, slot save) are marked and load only with the same setting. Not supported: models with K-only caches, attention sinks. Test with `scripts/gfx906/split-heads.sh`. |
| `GGML_SCHED_GPU_SPLIT_SYNC=1` | The scheduler no longer waits on the host between two graph splits on different GPUs when the second split has no inputs (it serialized independent work of two GPUs, e.g. `LLAMA_KV_SPLIT_HEADS`). `1` restores the wait. |
| `GGML_CUDA_FA_PREFER_VEC=1` | Use the FA vector kernel instead of the tile kernel at batch size 1 with F16 KV cache (head size <= 256). |

## Benchmark and correctness

```bash
# token generation, compare several builds
scripts/gfx906/bench-tg.sh model.gguf build-upstream build-gfx906

# compare the ROCm backend against the CPU backend for the affected ops
scripts/gfx906/test-ops.sh build-gfx906
```

## Changes

### DPP warp reductions (GCN5)

On AMD, `__shfl_xor` is compiled to `ds_bpermute_b32`, which goes through the LDS unit and waits on `lgkmcnt` at every step.
A full wave64 reduction was 6 such round trips. On GCN5 it is now:

- 4 steps with DPP (`quad_perm`, `row_half_mirror`, `row_mirror`), fused by the compiler into `v_add_f32_dpp`,
- 1 `ds_swizzle_b32` (swap 16),
- 2 `v_readlane_b32` for the last step (wave64 only).

This affects every kernel that uses `warp_reduce_sum`/`warp_reduce_max`: MMVQ (the final reduction of each row),
Q8_1 quantization of activations, RMS norm, softmax, top-k MoE routing, and flash attention.
At batch size 1 with MoE models the matrices are small (e.g. K = 2816), so the MMVQ K loop runs only a couple of
iterations and the final reduction is a noticeable fraction of the kernel time.
g gfx906 (1).md…]()



# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
