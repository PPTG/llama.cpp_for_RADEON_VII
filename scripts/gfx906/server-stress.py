#!/usr/bin/env python3
# Stress test for a running llama-server, like an agent stack: several clients at the same time, long shared prompt
# prefixes (the server saves idle slots to its prompt cache and loads them into other slots), requests with tools
# (grammar, CPU sampling) and without. Stops at the first failed request or when the server does not answer.
#
# usage: scripts/gfx906/server-stress.py [--url http://127.0.0.1:8080] [--clients 4] [--requests 60]
# To find the kernel or copy that fails, run the server with
#   AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3 HIP_LAUNCH_BLOCKING=1
# (slower, but the error is reported at the operation that caused it).

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

WORDS = ("server cache kernel memory token prompt attention layer tensor device vector matrix stream graph "
         "buffer thread socket driver compute shader sample logits context window batch split head value key "
         "query model weight bias norm scale route expert block row column queue event fence wave lane").split()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the project",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}},
            "required": ["path"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run a shell command and return its output",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}]


def make_doc(seed, n_words):
    rng = random.Random(seed)
    lines = []
    for i in range(n_words // 12):
        lines.append(f"{i}. " + " ".join(rng.choice(WORDS) for _ in range(12)) + ".")
    return "\n".join(lines)


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--requests", type=int, default=60, help="requests per client")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    # shared "system prompts" of different lengths (~1.3 tokens per word): conversations continue them
    docs = [make_doc(100 + i, n) for i, n in enumerate([600, 2500, 6000, 9000])]
    stop = threading.Event()
    lock = threading.Lock()
    stats = {"ok": 0, "tools": 0, "tokens": 0, "t_gen": 0.0, "errors": []}

    def client(cid):
        rng = random.Random(args.seed * 1000 + cid)
        history = {}  # doc index -> list of messages (the conversation grows, the prefix stays)
        for n in range(args.requests):
            if stop.is_set():
                return
            d = rng.randrange(len(docs))
            msgs = history.setdefault(d, [{"role": "system", "content": "You are a coding agent. Notes:\n" + docs[d]}])
            if rng.random() < 0.2:
                del msgs[1:]  # start the conversation again: prompt cache hit on the system prompt only
            msgs.append({"role": "user", "content": f"Step {n} of client {cid}: summarize notes {rng.randrange(100)} to "
                                                    f"{rng.randrange(100, 400)} in two sentences, or read a file."})
            body = {"messages": msgs, "max_tokens": rng.choice([32, 96, 200]), "temperature": 0.7, "top_k": 64,
                    "top_p": 0.95, "min_p": 0.05, "cache_prompt": True}
            use_tools = rng.random() < 0.5
            if use_tools:
                body["tools"] = TOOLS
            try:
                t0 = time.time()
                r = post(args.url + "/v1/chat/completions", body, args.timeout)
                msg = r["choices"][0]["message"]
                msgs.append({"role": "assistant", "content": msg.get("content") or json.dumps(msg.get("tool_calls", []))})
                tim = r.get("timings", {})
                with lock:
                    stats["ok"] += 1
                    stats["tools"] += use_tools
                    stats["tokens"] += tim.get("predicted_n", 0)
                    stats["t_gen"] += tim.get("predicted_ms", 0.0) / 1000.0
                    if stats["ok"] % 10 == 0:
                        tps = stats["tokens"] / stats["t_gen"] if stats["t_gen"] > 0 else 0
                        print(f"{stats['ok']} requests ok ({stats['tools']} with tools), generation {tps:.1f} t/s, "
                              f"last {time.time() - t0:.1f} s, prompt {tim.get('prompt_n', '?')} new tokens", flush=True)
            except (urllib.error.URLError, ConnectionError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                with lock:
                    stats["errors"].append(f"client {cid} request {n}: {e!r}")
                stop.set()
                return

    threads = [threading.Thread(target=client, args=(c,)) for c in range(args.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    alive = True
    try:
        urllib.request.urlopen(args.url + "/health", timeout=10).read()
    except Exception:
        alive = False

    tps = stats["tokens"] / stats["t_gen"] if stats["t_gen"] > 0 else 0
    print(f"=== {stats['ok']} requests ok ({stats['tools']} with tools), generation {tps:.1f} t/s, "
          f"server {'alive' if alive else 'NOT RESPONDING'}")
    for e in stats["errors"]:
        print("ERROR", e)
    sys.exit(0 if alive and not stats["errors"] else 1)


if __name__ == "__main__":
    main()
