#!/usr/bin/env python3
# How often generations end in a loop, per llama-server configuration: the fork defaults, the fork with its changes
# turned off by their environment variables, and an f16 KV cache. Same prompts and seeds for every configuration, no
# DRY, Gemma sampling (temp 1.0, top-k 64, top-p 0.95, min-p 0). A loop: the end of the text is one chunk of 4+ words
# repeated 3+ times, or one line of 16+ characters repeated 4+ times in a row.
#
# usage: scripts/gfx906/loop-test.py <model.gguf> [--build build-gfx906] [--seeds 5] [--tokens 1500] [--configs a,b]
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

PROMPTS = [
    ("game", "Write a complete, runnable Python program for a simple Minecraft-like voxel game with the ursina engine: "
             "terrain generation, first person controls, placing and breaking blocks. Explain it briefly."),
    ("tools", "You are an agent that manages Proxmox. Plan step by step how to create an LXC container with Docker, "
              "including the exact pct and shell commands for every step and how to verify each one."),
    ("essay", "Write a detailed essay about the history of the HP ProLiant DL580 server line and why old datacenter "
              "GPUs are popular for running language models at home."),
    ("list", "List 60 practical tips for writing maintainable Python code, each with a one sentence explanation."),
]

# changes of the fork that can be turned off at runtime
FORK_OFF = {
    "LLAMA_SAMPLING_FAST_TOP_K": "0", "LLAMA_GRAMMAR_PREFILTER": "0",
    "GGML_CUDA_FA_Q8_WAVE": "0", "GGML_CUDA_FA_TILE_Q8": "0", "GGML_CUDA_FUSE_FWHT": "0",
    "GGML_CUDA_DISABLE_FUSION": "1", "GGML_CUDA_TOPK_MOE_RANK": "0", "GGML_HIP_MMVQ_VARIANT": "0",
    "GGML_CUDA_MMVQ_Q8_CACHE": "0", "GGML_CUDA_FUSE_QKV": "0", "GGML_CUDA_FUSE_GLU_Q8": "0",
    "GGML_CUDA_FUSE_NORM_MULTI": "0", "GGML_CUDA_FA_PP_CFG_256": "0", "LLAMA_KV_SPLIT_HEADS": "0",
}

CONFIGS = {
    "fork":     ({"LLAMA_KV_SPLIT_HEADS": "1"}, ["-ctk", "q8_0", "-ctv", "q8_0"]),
    "fork-off": (FORK_OFF,                     ["-ctk", "q8_0", "-ctv", "q8_0"]),
    "f16-kv":   ({"LLAMA_KV_SPLIT_HEADS": "1"}, []),
}


def is_loop(text):
    words = text.split()
    tail = words[-400:]
    n = len(tail)
    for p in range(4, n // 3 + 1):
        chunk = tail[n - p:]
        if tail[n - 2 * p:n - p] == chunk and tail[n - 3 * p:n - 2 * p] == chunk:
            return True
    run, prev = 1, None
    for line in (l.strip() for l in text.splitlines()):
        if len(line) >= 16 and line == prev:
            run += 1
            if run >= 4:
                return True
        else:
            run = 1
        prev = line
    return False


def post(port, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_health(port, proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--build", default="build-gfx906")
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()

    out_dir = "results-gfx906/loop-test"
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for name in args.configs.split(","):
        env_add, srv_args = CONFIGS[name]
        env = dict(os.environ, **env_add)
        cmd = [f"{args.build}/bin/llama-server", "-m", args.model, "-ngl", "99", "-sm", "layer", "-fa", "on",
               "-c", str(args.ctx), "-np", "1", "--port", str(args.port)] + srv_args
        log = open(f"{out_dir}/server-{name}.log", "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            if not wait_health(args.port, proc):
                print(f"{name}: server did not start, see {out_dir}/server-{name}.log")
                continue
            loops, total, tps = 0, 0, []
            per_prompt = {}
            for kind, prompt in PROMPTS:
                for seed in range(args.seeds):
                    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": args.tokens,
                            "temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0, "seed": 1000 + seed,
                            "cache_prompt": False}
                    try:
                        res = post(args.port, body)
                    except Exception as e:
                        print(f"{name} {kind} seed {seed}: request failed: {e}")
                        continue
                    text = res["choices"][0]["message"].get("content") or ""
                    loop = is_loop(text)
                    loops += loop
                    total += 1
                    per_prompt.setdefault(kind, [0, 0])
                    per_prompt[kind][0] += loop
                    per_prompt[kind][1] += 1
                    tps.append(res.get("timings", {}).get("predicted_per_second", 0.0))
                    with open(f"{out_dir}/{name}-{kind}-{seed}{'-LOOP' if loop else ''}.txt", "w") as f:
                        f.write(text)
            detail = " ".join(f"{k} {v[0]}/{v[1]}" for k, v in per_prompt.items())
            line = f"{name:9s} loops {loops}/{total}   ({detail})   {sum(tps) / max(len(tps), 1):.1f} t/s"
            print(line, flush=True)
            summary.append(line)
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
    print(f"\ntexts: {out_dir}/ (files ending in -LOOP are the loops)")


if __name__ == "__main__":
    sys.exit(main())
