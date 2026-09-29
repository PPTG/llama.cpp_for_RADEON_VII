# llama.cpp for Radeon VII / Radeon Pro VII / MI50 (gfx906)

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for AMD gfx906 GPUs (Radeon VII, Radeon Pro VII,
Instinct MI50 / MI60) with ROCm / HIP: token generation, two GPUs without peer access, long context with a q8_0 KV
cache, and llama-server with tool calls (grammars). Most measurements are on Gemma 4 26B A4B, but most changes apply to
any model on gfx906 and fall back to the upstream code paths where they do not apply.

## Results

2x Radeon Pro VII (PCIe, no bridge, GPUs on different CPU sockets, no peer access), ROCm 7.1, Gemma 4 26B A4B.

Upstream vs fork, same settings (`llama-bench -fa 1`, f16 KV, `scripts/gfx906/final-bench.sh`, Q4_0):

| test           | 1 GPU upstream | 1 GPU fork | gain | 2 GPU upstream | 2 GPU fork | gain |
|----------------|---------------:|-----------:|-----:|---------------:|-----------:|-----:|
| tg128          |  96.6 | 115.9 | +20% |  82.8 | 107.9 | +30% |
| tg128 @ d4096  |  88.5 | 106.9 | +21% |  75.2 |  98.1 | +31% |
| tg128 @ d16384 |  86.2 | 101.8 | +18% |  75.9 |  92.7 | +22% |

Later changes on top of that (fusions, attention kernels for a q8_0 cache, scheduler and sampling work), 2 GPUs with the
[recommended settings](#run):

| | Q4_0 | Unsloth UD-Q4_K_M |
|---|---:|---:|
| token generation, llama-bench tg32 (d0) | ~110 t/s | ~90 t/s |
| token generation, llama-server | ~128 t/s | |
| prompt processing pp4096 at d0 / d30000 | 1985 / 1168 t/s | |
| prompt processing pp1024 at d100000 | 567 t/s | |

The Q4_K_M file keeps many tensors in q8_0 (15.8 vs 13.3 GiB); token generation is bound by memory bandwidth.
Details, per-change measurements and the history are in [docs/gfx906.md](docs/gfx906.md).

## Build

Requirements: ROCm (tested with 7.1), a gfx906 GPU.

```bash
scripts/gfx906/build.sh                      # -> build-gfx906/
```

The scripts in `scripts/gfx906/` use `build-gfx906` and `models/gemma-4-26B-A4B-it-Q4_0.gguf` by default; pass
another build directory / model as argument or with `MODEL=...`.

## Run

```bash
sudo rocm-smi --setperflevel high               # GPU clocks (+5-10% token generation with 2 GPUs)
sudo cpupower frequency-set -g performance      # CPU governor (kernel launches, sampling)

LLAMA_KV_SPLIT_HEADS=1 ./build-gfx906/bin/llama-server -m models/<model>.gguf \
    -ngl 99 -sm layer -fa on -ctk q8_0 -ctv q8_0 -c 120000 -np 1 -ub 1024 -b 2048
```

- `LLAMA_KV_SPLIT_HEADS=1` (2 GPUs): half of the KV heads of the global attention layers run on the other GPU. Faster
  prompt processing at long context (+57% at 100k), a bit slower at short context (-13% at d0); leave it off for short
  contexts. Experimental: saved KV cache states only load with the same setting.
- `-ub 1024`: +14% prompt processing over the default 512.
- `-sm layer` is faster than `-sm tensor` on two GPUs without peer access.
- Everything else is on by default. [docs/gfx906.md](docs/gfx906.md) lists the environment variables that turn single
  changes off or select tuning variants.

## What is different from upstream

Token generation
- DPP based warp reductions on GCN5.
- Quantized mat-vec (MMVQ) geometry tuned for 64 wide waves (1 warp x 8 rows; 2 x 4 for q8_0 experts).
- Fewer kernels per token: Q/K/V (and gate/up) as one MMVQ launch, several rms_norm chains in one kernel, glu +
  q8_1 quantization, rms_norm + mul + rope + Hadamard rotation of Q, activations quantized once for several matmuls.
- Top-k MoE routing by ranking (one pass instead of k argmax rounds).

Attention (flash attention with a q8_0 KV cache)
- Token generation reads the q8_0 cache directly instead of converting it to f16 on every call; for head size 512
  with GQA 8 (Gemma 4 global layers) a kernel where one wave computes all Q heads with int8 dot products.
- The Hadamard rotation of the quantized KV cache runs inside the kernels that store K/V and quantize the output.
- Tuned tile configurations for the Gemma 4 shapes, more blocks in the combine kernel.

Two GPUs
- Copies between GPUs through pinned host memory with async D2H + H2D (no host stalls without peer access).
- `LLAMA_KV_SPLIT_HEADS`: the attention of a layer on both GPUs at once, with a second stream per GPU.
- Scheduler fixes that keep the two GPUs overlapping on consecutive ubatches (pipeline parallelism).

llama-server
- CPU sampling on the top logits instead of the whole vocabulary (same tokens), and a grammar prefilter on the first
  code points of each token: tool calls with a JSON grammar sample several times faster.

## Correctness

- `test-backend-ops` cases for the new kernels and fusions (compare against the CPU backend).
- `scripts/gfx906/check.sh`: the fused and unfused paths give the same text.
- `scripts/gfx906/quality.sh`: perplexity and KL divergence against an f16 reference, with `SANITY=1` CPU vs GPU.

## Scripts

| script | |
|---|---|
| `build.sh` | build for gfx906 |
| `run-all.sh` | build variants, `test-backend-ops`, `llama-bench`, kernel profile, summary |
| `quality.sh` | KL divergence / perplexity vs an f16 reference |
| `profile-tg.sh`, `profile-pp.sh` | `rocprofv3` kernel profile of token generation / prompt processing |
| `long-ctx.sh`, `split-heads.sh`, `server-bench.sh`, `server-stress.py` | long context, two GPUs, llama-server |
| `tune-*.sh`, `fa-*.sh` | sweeps of the tuning variants |

## Status

Tested on 2x Radeon Pro VII with ROCm 7.1, mainly with Gemma 4 26B A4B (Q4_0, Unsloth UD-Q4_K_M). Other models and
single GPU setups use the same kernels but are less measured. Reports and measurements from other gfx906 setups are
welcome. Same license as llama.cpp (MIT).

---

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
