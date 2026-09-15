# π0.5 batch-size-1 latency on RTX 5090 (`pi05-bs1-latency` branch)

Goal: make `pi05_libero` inference as fast as possible at batch size 1 **without losing task success**, for
single-robot deployment. Accuracy is checked by closed-loop LIBERO-Spatial rollouts, not only tensor deltas.

## Headline

| Stack | LIBERO-Spatial success | Client round trip p50 / p90 / p99 (ms) | Server `Policy.infer` p50 (ms) |
| --- | ---: | ---: | ---: |
| Upstream openpi `215abfb` (JAX) | **100 / 100** | 73.1 / 75.4 / 77.4 | 72.2 |
| **This branch, PyTorch + NVFP4** | **100 / 100** | **25.1 / 25.5 / 26.1** | **24.2** |

**All four LIBERO suites with the latest code** (10 tasks × 50 episodes each, 2,000 episodes): NVFP4 reaches
**96.75 %** (openpi reports 96.85 % for this checkpoint), at 22.07 ms server / 23.25 ms round trip p50. See
[`pi05_rtx5090_libero_all_suites.md`](pi05_rtx5090_libero_all_suites.md); how the speedup comes about:
[`pi05_rtx5090_speedup_explained.md`](pi05_rtx5090_speedup_explained.md).

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

## FLASH speculative inference: faster, but accepted actions deviate from the policy

[Realtime-VLA FLASH](https://arxiv.org/abs/2605.13778) (code: `dexmal/realtime-vla-flash`, π0 only) skips the
PaliGemma prefill on most replanning rounds: a ~110M-parameter draft (one Gemma block initialized from VLM layer 0,
learned action queries) proposes the chunk from the current prefix embeddings, the Action Expert reconstructs the
endpoint at t ∈ {0.10, 0.05} using the last full round's KV cache, and the longest prefix within δ = 0.15 is executed;
rejection or a predicted gripper switch falls back to a full round. A [DSpark](https://arxiv.org/abs/2607.05147)-style
confidence head (per-action acceptance probability) was trained jointly with the draft. The verifier settings match
the reference implementation (`t_list = (0.10, 0.05)`, prefix rule, gripper fallback) with the paper's δ = 0.15.

**Status: not adopted; closed-loop success untested.** Under the real verifier the draft reaches the paper's
flash-path rate and a 1.8× per-action speedup on an H100 (bf16), but the actions it gets accepted on unseen episodes are
far from what the policy would output, and the verifier hardly distinguishes accurate from inaccurate drafts. Whether
that costs task success needs closed-loop LIBERO rollouts, which were not run.

### What was run (1× H100, bfloat16, `scripts/flash/`)

1. Teacher targets: bfloat16 `pi05_libero` (no quantization; H100 has no NVFP4 kernels), zero noise, all 52,970
   LIBERO-Spatial frames, 32 min.
2. Draft training: 100 epochs, batch 64, 411 train / 21 validation episodes, `--cache-prefixes --confidence`
   (~5 min to cache prefixes in host RAM, then ~50 min at 0.038 s/step). Best checkpoint by validation RMS: step 37,000.
3. Offline verifier check (`eval_verifier_offline.py`): recorded episodes replayed through `FlashPolicy` as a client
   would serve them (full round at the start, then one request per replan at the frame after the executed actions, so
   flash rounds verify against a KV cache several frames old). Observations follow the demonstration, not the policy's
   own actions. Eager model with CUDA graphs, no `torch.compile`.

### Verifier acceptance and latency (H100, bf16)

| Episodes | Rounds | Flash-path rounds | Attempts accepted | Accepted prefix (flash rounds) | Full / flash / fallback round (ms, median) | ms per action | Speedup per action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 validation | 592 | 67.2 % | 77.6 % | 88.5 % of 5 | 45.4 / 8.4 / 53.2 | 4.96 | 1.83× |
| 21 training | 514 | 71.2 % | 82.8 % | 93.8 % of 5 | 45.4 / 8.4 / 53.1 | 4.40 | 2.07× |

Paper, FLASH-π0 on LIBERO-Spatial: 71.6 % flash-path rounds, 75.8 % accepted prefix. About 10 % of flash attempts
fall back because a gripper switch is predicted.

### How close are executed actions to the policy?

RMS distance (first 6 dims, normalized action space) of executed actions to the teacher's zero-noise chunk on the same
frame. For full rounds the actions are the policy's own output with random noise, which measures its sampling spread.

| Episodes | Full-round actions: mean / within 0.15 | Accepted draft actions: mean / within 0.15 |
| --- | ---: | ---: |
| 21 validation | 0.036 / 99.3 % | **0.196 / 32.3 %** |
| 21 training | 0.035 / 99.2 % | 0.107 / 85.8 % |

- **Accepted actions deviate.** pi05_libero is nearly deterministic here (samples within 0.036 of the zero-noise chunk),
  but on unseen episodes the accepted draft actions are 0.196 away, about 5× the policy's own spread.
- **The verifier barely tracks draft quality.** Drafts are twice as accurate on training episodes (0.107 vs 0.196), yet
  acceptance rises only from 77.6 % to 82.8 %. Its per-step distances cluster just under δ: raising δ from 0.10 to 0.15
  lifts the mean accepted prefix from 0.8 to 3.5 steps. The endpoint reconstructed from x_t = t·noise + (1 − t)·draft
  moves only about t times the policy's correction away from the draft, so small t makes the check lenient. The paper
  reports the same risk (Table 7: the verifier alone at K = 2, δ = 0.15 drops LIBERO-10 success to 58.4 %) and recovers
  it with phase-aware fallback and periodic full rounds.
- **The draft overfits.** Validation RMS plateaus near 0.195-0.21 from ~16k of 78.5k steps while training loss keeps
  falling, and accepted actions are much more accurate on training than on validation episodes.
- **The confidence head is miscalibrated for its training labels** (distance to the teacher, not verifier acceptance):
  on held-out episodes mean p = 0.81 vs 35 % of steps within 0.15.

An earlier revision of this section first called FLASH ineffective from the draft-vs-teacher distance alone, then
argued that distance underestimates acceptance because a flow policy has several valid chunks. Acceptance is indeed
much higher than that distance suggested, but the policy's samples are tightly clustered, so the distance is a fair
measure of how far executed actions are from the policy.

### Expected speedup on the RTX 5090 (not measured)

NVFP4 speeds up the PaliGemma prefill that flash rounds skip, so the gain shrinks. With the validation round mix above
(67.2 % flash, 29.2 % fallback, 3.5 % full, 4.61 executed actions per round), a 24.2 ms full round and a flash round
costing S ms, latency per action is (0.964·S + 7.9) / 4.61 ms vs 4.84 ms for full rounds only: 1.63× at S = 6,
1.43× at S = 8, 1.15× at S = 12. Periodic full rounds, which the paper needs for reliability, reduce it further.

### Numbers used here

Training metrics: [`docs/pi05_rtx5090_eval/flash/metrics.jsonl`](pi05_rtx5090_eval/flash/metrics.jsonl); verifier
summaries: [`verifier_val.json`](pi05_rtx5090_eval/flash/verifier_val.json),
[`verifier_train21.json`](pi05_rtx5090_eval/flash/verifier_train21.json). Draft checkpoint (private):
`zhangj1an/pi05-libero-spatial-flash-draft` on Hugging Face. Code: `src/openpi/models_pytorch/flash.py` (draft head
with an action-slot forward bitwise equal to the full block, `triton_ops.py` kernels matching PyTorch's rounding,
confidence head), `src/openpi/policies/flash_policy.py`, `scripts/flash/` (see `scripts/flash/README.md`).

## Raw results

Per-episode LIBERO results, per-chunk latencies, the latency/exactness harness outputs and the scripts that produced
them are in [`docs/pi05_rtx5090_eval/`](pi05_rtx5090_eval/README.md).
