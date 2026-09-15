# π0.5 + NVFP4 on all four LIBERO suites (RTX 5090, `pi05-bs1-latency`)

Question: does serving `pi05_libero` through this branch's fast PyTorch path with **NVFP4** quantization
(`--pytorch-quantization nvfp4`) cost task success? The earlier check ([`pi05_rtx5090_latency.md`](pi05_rtx5090_latency.md))
covered only LIBERO-Spatial with 10 episodes per task, and ran before the masked-camera-slot change (`813e805`). This run
uses the **latest branch code** on **all four suites** with openpi's standard protocol (**10 tasks × 50 episodes per
suite, 2,000 episodes**).

## Result

**No measurable accuracy loss.** NVFP4 averages **96.75 %** across the four suites, against **96.85 %** reported by openpi
for the same checkpoint (JAX, [`examples/libero/README.md`](../examples/libero/README.md)). Every suite is within
1.0 pp of the reported number, well inside the sampling noise of 500 episodes (95 % Wilson intervals below).

| Suite | NVFP4 (this run) | 95 % CI | openpi reported (JAX) | Per task (successes / 50, tasks 0–9) |
| --- | ---: | ---: | ---: | --- |
| LIBERO-Spatial | **489 / 500 = 97.8 %** | 96.1–98.8 % | 98.8 % | 50 50 50 47 47 48 49 50 50 48 |
| LIBERO-Object | **494 / 500 = 98.8 %** | 97.4–99.4 % | 98.2 % | 48 50 50 48 50 49 50 50 49 50 |
| LIBERO-Goal | **489 / 500 = 97.8 %** | 96.1–98.8 % | 98.0 % | 49 50 47 48 49 50 49 50 50 47 |
| LIBERO-10 | **463 / 500 = 92.6 %** | 90.0–94.6 % | 92.4 % | 47 50 49 50 49 50 47 49 **28** 44 |
| **Average** | **1935 / 2000 = 96.75 %** | 95.9–97.4 % | **96.85 %** | |

The weakest task is LIBERO-10 task 8, *"put both moka pots on the stove"* (28/50). openpi publishes only suite averages, so
whether this task is also the weakest for the unquantized model is answered by the control below. A same-code bf16 control (quantization off, identical client and episodes) is
running at the time of writing and will be added below, to separate quantization effects from simulator/stack
differences with openpi's JAX number.

### Latency (clean single-client runs)

Measured separately from the accuracy run: one client, nothing else on the GPU or CPU, all four suites × 10 tasks ×
3 episodes, every chunk served (first 5 warm-up requests excluded). Server time is `Policy.infer` inside the policy
server; round trip is what the robot sees (`client.infer()` over the loopback websocket).

| Model | Chunks | Server p50 | Server p99 | Server mean ± std | Round trip p50 | Round trip p99 | Round trip mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **NVFP4** | 3,749 | **22.07 ms** | **23.42 ms** | 22.07 ± 0.45 ms | **23.25 ms** | **25.55 ms** | 23.30 ± 1.38 ms |
| bf16 (same code, no quantization) | 3,844 | 28.78 ms | 30.19 ms | 28.83 ± 0.49 ms | 30.15 ms | 32.30 ms | 30.22 ± 0.73 ms |

NVFP4 saves **6.7 ms (−23 %)** per chunk at p50. Per suite (NVFP4, p50 / p99 ms):

| Suite | Server | Round trip |
| --- | ---: | ---: |
| LIBERO-Spatial | 22.21 / 24.05 | 23.56 / 26.49 |
| LIBERO-Object | 22.10 / 23.19 | 23.32 / 25.39 |
| LIBERO-Goal | 21.98 / 23.28 | 23.19 / 25.16 |
| LIBERO-10 | 21.98 / 23.30 | 23.07 / 25.34 |

All 40 LIBERO prompts are at most 21 tokens, so every request uses the 48-token bucket and the same captured graph;
latency does not depend on the suite. Compared with the earlier LIBERO-Spatial run (25.1 / 26.1 ms round trip p50 / p99,
3 camera slots), skipping the empty third camera slot saves about 2 ms.

The round-trip numbers inside the 2,000-episode accuracy run are **not** latency measurements: that run used 5
concurrent clients on one server (requests queue), and a second server was compiling on the same GPU for part of it.

## Setup

| | |
| --- | --- |
| Code | `pi05-bs1-latency` @ `70191ce` (+ a websocket compatibility fix in `sim_client_openpi.py`, see below) |
| GPU / host | 1× RTX 5090 32 GB (driver 570.195.03), container limited to 12.75 CPU cores, RunPod |
| Stack | torch 2.9.1+cu128, torchao 0.14.1, Python 3.11 server; LIBERO client in Python 3.8 (`examples/libero/requirements.txt`) |
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` → `examples/convert_jax_model_to_pytorch.py` (bfloat16) + `assets/` |
| Server | `scripts/serve_policy.py --pytorch-quantization nvfp4 policy:checkpoint --policy.config pi05_libero --policy.dir <ckpt>` (defaults: `--token-len-buckets 48`, whole-inference CUDA graph, `torch.compile(max-autotune-no-cudagraphs)`, masked camera slots dropped); 126 PaliGemma-LM linears quantized |
| First request | 19.5 min of compilation + autotuning + graph capture for both prompt buckets on this host (cached in `TORCHINDUCTOR_CACHE_DIR`; the bf16 server's first request took ~14 min) |
| Client | `docs/pi05_rtx5090_eval/scripts/sim_client_openpi.py`, mirrors `examples/libero/main.py`: official init states (episode *i* uses init state *i*), seed 7, 256 px render, 180° flip, `resize_with_pad` 224, 10 no-op settle steps, replan every 5 actions, max steps 220 / 280 / 300 / 520 (Spatial / Object / Goal / 10), MuJoCo EGL |
| Accuracy run | `scripts/run_libero_all_suites.sh nvfp4 8000 5`: 40 (suite, task) jobs over 5 parallel clients, 50 episodes each, ~53 min |

Each task episode set is deterministic in its initial state but not in its actions: the policy samples fresh Gaussian
noise per request, as in openpi's own evaluation.

## Reproduce

```bash
# Server (Python 3.11 venv from `uv sync`, transformers_replace copied into it)
TORCHINDUCTOR_CACHE_DIR=/persistent/inductor uv run scripts/serve_policy.py --port 8000 --pytorch-quantization nvfp4 \
    policy:checkpoint --policy.config pi05_libero --policy.dir /path/to/pi05_libero_pytorch

# Client env (Python 3.8), see examples/libero/README.md; `apt install libegl1` for headless MuJoCo EGL
LIBERO_PYTHON=/path/to/libero_venv/bin/python RUNS_DIR=runs \
    docs/pi05_rtx5090_eval/scripts/run_libero_all_suites.sh nvfp4 8000 5
python docs/pi05_rtx5090_eval/scripts/aggregate_libero_suites.py runs/nvfp4 runs/nvfp4_summary.json
```

Notes from setting this up:
- RunPod's nginx listens on port 8001 (a server started there fails to bind, and a client then reaches whatever nginx
  proxies to); use another port.
- websockets 13.1 (the newest for Python 3.8) has no `ping_interval` in its sync client. `sim_client_openpi.py` now falls back
  to connecting without it; that client sends no keepalive pings, so the long first request cannot time out.

## Raw data

[`pi05_rtx5090_eval/libero_all_suites/`](pi05_rtx5090_eval/libero_all_suites/):
- `nvfp4/<suite>/task<i>.json`: per-episode success, steps, chunks and wall time, plus every chunk's round trip and
  server time (5 concurrent clients, so the timings are queue-affected).
- `nvfp4_summary.json`: per-suite / per-task totals and the failed `(task, episode)` pairs.
- `nvfp4_latency/<suite>.json`, `bf16_latency/<suite>.json`: the clean single-client latency runs.

Why the path is fast, and what openpi vs this branch contribute: [`pi05_rtx5090_speedup_explained.md`](pi05_rtx5090_speedup_explained.md).
