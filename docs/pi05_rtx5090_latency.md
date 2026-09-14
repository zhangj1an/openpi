# π0.5 batch-size-1 latency on RTX 5090 (`pi05-bs1-latency` branch)

Goal: make `pi05_libero` inference as fast as possible at batch size 1 **without losing task success**, for
single-robot deployment. Accuracy is checked by closed-loop LIBERO-Spatial rollouts, not only tensor deltas.

## Headline

| Stack | LIBERO-Spatial success | Client round trip p50 / p90 / p99 (ms) | Server `Policy.infer` p50 (ms) |
| --- | ---: | ---: | ---: |
| Upstream openpi `215abfb` (JAX) | **100 / 100** | 73.1 / 75.4 / 77.4 | 72.2 |
| **This branch, PyTorch + NVFP4** | **100 / 100** | **25.1 / 25.5 / 26.1** | **24.2** |

**2.9× lower end-to-end latency per action chunk, same success rate.** Same checkpoint
(`gs://openpi-assets/checkpoints/pi05_libero`, converted to PyTorch with `examples/convert_jax_model_to_pytorch.py`),
same client, same episodes. The NVFP4 rollout ran before the empty-camera-slot change (`813e805`), i.e. with 3 camera
slots; the 2-slot configuration is expected to be faster still and has not been re-measured end to end yet.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 32 GB (Blackwell, sm_120), driver 580.105.08 |
| CPU | 2× AMD EPYC 7302 (Zen 2), container limited to ~15 cores — slow host, so launch overhead is expensive |
| Model | `pi05_libero`: bf16, 10 flow-matching steps, action horizon 10, 2 real cameras (agentview + wrist) |
| JAX | 0.5.3; cuBLAS 12.8.4 / cuDNN 9.10.2 after `5d74cd8` |
| PyTorch | 2.9.1 (CUDA 12.8 build, Triton 3.5), torchao 0.14.1 |

**LIBERO client** (mirrors `examples/libero/main.py`): LIBERO-Spatial, tasks 0–9 × 10 episodes, official init states,
seed 7, 256px render, 180° flip, `resize_with_pad` to 224, replan every 5 steps, loopback websocket; round trip is
`client.infer()` over every chunk served during the 100 rollouts (2,103 chunks), after 5 warmup requests.

**Latency harness**: one real LIBERO frame, fixed noise, 5 warmup + 60–100 timed `Policy.infer` calls. "Exact" checks
compare actions with XLA autotuning disabled (`--xla_gpu_autotune_level=0`, deterministic across processes); with
autotuning on, cross-process noise alone is ~1.6e-3.

## JAX path

| Commit | Change | `Policy.infer` p50 (ms) | Actions vs `main` |
| --- | --- | ---: | --- |
| `main` | upstream | 63.3 | – |
| `b037fa8` | all camera slots through SigLIP in one call | 59.7 | bit-identical |
| `6240e72` | whole `Policy.infer` as one jitted call (batching, uint8→float, RNG split inside) | 51.4 | 1.6e-3 (kernel selection) |
| `c4560f7` | XLA command buffers for all op types in `serve_policy` | 50.3 | kernels unchanged |
| `5d74cd8` | torch 2.8 pin → shared cuBLAS 12.8 / cuDNN 9.10 | 49.5 | unchanged |
| `aedbab5` | prompt padded to the smallest length bucket (48 instead of 200) | **45.2** | 2.0e-3 |

Tried and dropped: `while_loop` → `scan` (63.8 ms, no gain); XLA command buffers off (56.6 ms); Triton GEMM /
cuDNN fMHA flags (no gain); cuDNN fused attention via `jax.nn.dot_product_attention` (54.6 ms, slower: the padding
mask forces the non-flash path). JAX has no FP4 matmul kernels, so NVFP4 is PyTorch-only.

## PyTorch path

| Commit | Change | `Policy.infer` p50 (ms) | Actions vs openpi JAX |
| --- | --- | ---: | --- |
| upstream | eager | 277.6 | 4.7e-3 |
| `9137ff5` | fixed-trip denoise loop, on-device masks/scalars, batched SigLIP | 240.2 | 5.0e-3 |
| `9137ff5` | + whole inference captured into one CUDA graph | 59.3 | identical to eager |
| `aedbab5` | + prompt buckets (48) | 54.5 | 4.8e-3 |
| `13849fb` | + `torch.compile(max-autotune-no-cudagraphs)` under the CUDA graph | 34.4 | 4.0e-3 |
| `3c2f4ee` | + **NVFP4** PaliGemma LM linears (`--pytorch-quantization nvfp4`, 126 linears) | **22.9** | 1.0e-2 |
| `813e805` | + skip masked camera slots (prefix 816 → 560 tokens, 2 SigLIP images) | not re-measured | 2.5e-3 vs 3 slots |

Notes:
- `torch.compile` with Triton 3.4 (torch 2.8) segfaults in `make_llir` compiling SigLIP on this GPU; Triton 3.5 works.
- Inductor's own CUDA graphs (`max-autotune`, `reduce-overhead`) are not used: the Policy captures the whole inference
  in one graph instead.
- First request pays compilation + capture: ~7 min per prompt-length shape on this CPU (~11 min for both shapes with
  NVFP4). Results land in the Inductor cache (`TORCHINDUCTOR_CACHE_DIR`), so later server starts are faster; warm the
  cache once per deployment image.
- FP8 (torchao dynamic per-row) was started but not finished: its autotuning (92 kernel choices per shape) was too slow
  on this host to complete; NVFP4 was prioritized.

### Serving

```bash
# JAX (defaults now include command buffers and --token-len-buckets 48)
uv run scripts/serve_policy.py --env LIBERO

# PyTorch + NVFP4 (checkpoint converted with examples/convert_jax_model_to_pytorch.py, assets/ copied next to it)
TORCHINDUCTOR_CACHE_DIR=/persistent/inductor_cache \
uv run scripts/serve_policy.py --pytorch-quantization nvfp4 \
    policy:checkpoint --policy.config pi05_libero --policy.dir /path/to/pi05_libero_pytorch
```

## Other stacks measured on the same machine (for reference)

| Stack | Checkpoint | LIBERO-Spatial | Round trip p50 (ms) |
| --- | --- | ---: | ---: |
| LeRobot `PI05Policy`, eager bf16 | `lerobot/pi05_libero_finetuned_v044` | 8/10 (latency run) | ~279 per chunk (in-process) |
| vLLM-Omni `pi05-cudagraph` | openpi `pi05_libero` converted | 99 / 100 | 93.3 |
| vLLM-Omni `pi05-cudagraph` | `lerobot/pi05_libero_finetuned_v044` | 97 / 100 | 94.6 |

## In progress: FLASH speculative inference

[Realtime-VLA FLASH](https://arxiv.org/abs/2605.13778) (code: `dexmal/realtime-vla-flash`, π0 only) skips the
PaliGemma prefill on most replanning rounds: a ~110M-parameter draft (one Gemma block initialized from VLM layer 0,
learned action queries) proposes the chunk from the current prefix embeddings, the Action Expert reconstructs the
endpoint at t ∈ {0.10, 0.05} using the last full round's KV cache, and the longest prefix within δ = 0.15 is executed;
rejection or a predicted gripper switch falls back to a full round. Gated on NVFP4 keeping 99–100 % success (it does).

Status:
1. Teacher targets: NVFP4 policy with zero noise over all 52,970 LIBERO-Spatial training frames (~25 min at 35 frames/s).
2. Draft training: 5 epochs, prefix embeddings recomputed on the fly (a cached prefix would be hundreds of GB).
3. Serving: `FlashPolicy` with separate CUDA graphs for full rounds (exposing the prefix KV cache) and flash rounds
   (SigLIP + draft + K=2 batched verification reading that cache without copies); then the same 100-episode eval.

Not committed yet: `src/openpi/models_pytorch/flash.py`, `src/openpi/policies/flash_policy.py`, `scripts/flash/`.
