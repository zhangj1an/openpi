# Why `pi05-bs1-latency` is fast: openpi's design + what this branch adds

This document explains where the latency of one π0.5 inference comes from and why the
`pi05-bs1-latency` branch serves `pi05_libero` in **~22–25 ms per action chunk on an RTX 5090** (upstream openpi:
~63 ms JAX, ~278 ms PyTorch eager, same GPU), without losing LIBERO task success
(accuracy check across all four suites: [`pi05_rtx5090_libero_all_suites.md`](pi05_rtx5090_libero_all_suites.md)).

It covers two layers:

1. **What openpi / π0.5 already does to keep inference cheap.** None of this is new on the branch, but the
   branch's optimizations rely on it.
2. **What the branch adds**, commit by commit, with the mechanism, whether outputs change, and the measured effect.

Branch measurements come from [`pi05_rtx5090_latency.md`](pi05_rtx5090_latency.md) (in-process `Policy.infer` harness,
one real LIBERO frame, fixed noise) unless marked otherwise. Numbers marked *estimate* are back-of-the-envelope.

---

## 1. One π0.5 inference, as designed in openpi

```
 robot / LIBERO sim process                          policy server process (scripts/serve_policy.py)
 ───────────────────────────                         ───────────────────────────────────────────────
 2 cameras (agentview, wrist) 224×224                websocket + msgpack-numpy
 8-D proprio state, task string    ── request ──►    input transforms (CPU): LiberoInputs → Normalize →
                                                     ResizeImages → TokenizePrompt → PadStatesAndActions
                                                                        │
                                                      ┌─────────────────▼──────────────────────────┐
                                                      │ PREFIX (once per request)                  │
                                                      │  SigLIP So400m/14: 256 tokens per image    │
                                                      │  + prompt tokens                           │
                                                      │  → Gemma-2B (18 layers, width 2048)        │
                                                      │    bidirectional, writes KV cache          │
                                                      ├────────────────────────────────────────────┤
                                                      │ SUFFIX (10 flow-matching Euler steps)      │
                                                      │  10 noisy action tokens + time (adaRMS)    │
                                                      │  → action expert Gemma-300M (width 1024)   │
                                                      │    attends to cached prefix KV             │
                                                      │  x ← x + dt·v(x, t)                        │
                                                      └─────────────────┬──────────────────────────┘
 execute first 5 of 10 actions    ◄── 10×7 actions ── output transforms: Unnormalize → LiberoOutputs
 then request again
```

### openpi design decisions that make this cheap

| Design (upstream openpi / π0.5) | Why it matters for latency |
| --- | --- |
| **Prefix/suffix split with block-causal attention.** Images + prompt form a bidirectional prefix; action tokens attend to the prefix but the prefix never attends to them. | The prefix does not depend on the noisy actions, so the 2B VLM runs **once per request**. Its KV cache is reused by all 10 denoising steps (`sample_actions` in `pi0.py` / `pi0_pytorch.py`). Without the cache every step would re-run the VLM: roughly 10× the compute. |
| **Separate, small action expert (Gemma-300M) in a mixture-of-transformers.** Both experts share attention per layer, but have their own weights. | The per-step work is a 300M model over only `action_horizon` = 10 tokens, not the 2B model. |
| **Flow matching with 10 Euler steps**, the whole chunk denoised in parallel. | 10 network evaluations per chunk, versus 50–100 for typical diffusion samplers, or one decode step per action token for autoregressive π0-FAST. |
| **Action chunking**: the model predicts 10 actions, and the LIBERO client executes 5 before replanning (`replan_steps=5`). | One inference per 5 control steps. At LIBERO's 20 Hz control that is a 250 ms budget per inference, so latency mostly matters for real robots and throughput. |
| **π0.5 conditioning**: the flow time enters the expert through adaRMSNorm, not an extra token + MLP. With `pi05_libero`, `discrete_state_input=False`, so the prompt is just the task. | The suffix is exactly the 10 action tokens, and the prefix prompt stays short (at most 21 tokens for the 40 LIBERO tasks). |
| **Multi-query attention** (`num_kv_heads=1`, head_dim 256) in both Gemmas. | The KV cache is small: 18 layers × 1 head × 256 dims × ~560 tokens. Suffix attention over the cache is cheap and memory-light. |
| **bfloat16 weights/activations**; JAX `jit` (XLA) or PyTorch `torch.compile` support. | Standard mixed-precision inference and a compiled graph instead of Python-driven eager ops (in principle; see §2 for what upstream PyTorch actually loses). |
| **Client/server split over a websocket** (`openpi_client`), msgpack-numpy payloads. | The robot/sim loop and the GPU model live in separate processes. Serialization of two 224×224 images is ~1 ms on loopback. |

### Where the compute goes (*estimate*)

Linear-layer FLOPs for one `pi05_libero` request with two real cameras and a 48-token prompt:

| Part | Tokens × passes | Params touched | ≈ FLOPs | Share |
| --- | --- | --- | ---: | ---: |
| SigLIP So400m | 2 images × 256 tokens | ~0.4 B | ~0.4 T | ~15 % |
| Gemma-2B prefix | 560 tokens, once | ~2.0 B | ~2.2 T | ~80 % |
| Gemma-300M suffix | 10 tokens × 10 steps | ~0.3 B | ~0.06 T | ~2 % |

So the **prefix LM dominates compute**. The action expert is tiny in FLOPs, but it is 10 sequential passes of ~18
layers of small kernels. At batch size 1 that part is dominated by **kernel-launch and host overhead**, not math. The
branch's optimizations fall into these two buckets: cut launch/host overhead everywhere, then cut the prefix LM's compute.

---

## 2. What the branch adds (PyTorch path, the one that is served)

Cumulative `Policy.infer` p50 on the RTX 5090:

| Step | Commit | Change | p50 (ms) | Outputs |
| --- | --- | --- | ---: | --- |
| 0 | upstream | PyTorch eager | 277.6 | – |
| 1 | `9137ff5` | Fixed-trip denoise loop, on-device masks/scalars, batched SigLIP | 240.2 | same math |
| 2 | `9137ff5` | **Whole inference captured in one CUDA graph** | 59.3 | bit-identical to eager |
| 3 | `aedbab5` | **Prompt length buckets** (pad to 48, not 200) | 54.5 | exact (padding is masked) |
| 4 | `13849fb` | **`torch.compile(max-autotune-no-cudagraphs)` under the CUDA graph** | 34.4 | kernel-level fp differences |
| 5 | `3c2f4ee` | **NVFP4 W4A4 for the 126 PaliGemma-LM linears** | **22.9** | numerics change (max-abs 1e-2 vs JAX) |
| 6 | `813e805` | **Skip masked camera slots** (3 → 2 SigLIP images, prefix 816 → 560 tokens) | **22.07** (server p50 in the all-suite latency run; bf16: 28.78) | exact |

For reference, upstream's `torch.compile(mode="default")` without the branch changes measured 117.9 ms. End to end
through the websocket with the final code: **23.25 ms p50 / 25.55 ms p99** round trip (NVFP4), 30.15 / 32.30 ms (bf16),
from single-client LIBERO runs over all four suites ([`pi05_rtx5090_libero_all_suites.md`](pi05_rtx5090_libero_all_suites.md)).

### 2.1 Make the whole inference capturable, then capture it in one CUDA graph (`9137ff5`)

**Problem.** Upstream `PI0Pytorch.sample_actions` is Python-driven:

- `while time >= -dt / 2:` compares a CUDA tensor with a Python float. That forces a **device→host sync on every
  denoising step**, and makes the loop data-dependent.
- `torch.tensor(dt)`, `torch.tensor(1.0)` and `torch.tensor(att_masks_list)` do a **host→device copy** on every call
  (inside `embed_prefix` / `embed_suffix`, so once per denoising step).
- Everything else is ~thousands of small CUDA kernel launches per request: 18 LM layers + 10 × 18 expert layers, each
  with norms, rotary, attention, MLP, residual gates. At batch size 1 the GPU finishes each kernel faster than Python
  can launch the next one, so the GPU is idle most of the time.

**Change.**

- `for _ in range(num_steps)` replaces the `while` loop. The Euler updates are identical, with no sync and no branch.
- Scalars and masks are created on-device (`torch.full`, `torch.zeros(...).fill_`), with no host copies.
- `_TorchChunkGraph` (`src/openpi/policies/policy.py`) wraps *everything after the CPU transforms* in one
  `torch.cuda.CUDAGraph`: uint8→float image conversion, preprocessing, SigLIP, the prefix pass with KV cache, all 10
  denoising steps. It warms up on a side stream (so cuBLAS handles/autotuning happen outside the capture), captures once
  per input shape, and serves each request by copying the inputs into static buffers, drawing noise into a static
  buffer, calling `graph.replay()`, and copying the 10×32 action tensor back.

**Effect.** One launch per request instead of thousands: **240 → 59 ms** (4×). Replay is bit-identical to eager.

### 2.2 Batch the camera slots through SigLIP (`b037fa8`, JAX and PyTorch)

Upstream embeds each camera slot with its own SigLIP call. The branch concatenates the slots on the batch axis and
runs SigLIP once, then reshapes back to `(batch, views × 256, d)`. It is bit-identical and uses one larger,
GEMM-friendlier batch instead of 2–3 small ones. JAX: 63.3 → 59.7 ms.

### 2.3 Prompt length buckets (`aedbab5`)

`pi05` pads every prompt to `max_token_len = 200`, but the 40 LIBERO task prompts are 21 tokens at most. `PaligemmaTokenizer` now pads
to the smallest configured bucket that fits (`--token-len-buckets`, default `(48,)`; 200 is always kept as a fallback).
Padding tokens are masked out of attention and do not advance positions, so outputs are unchanged. The prefix
shrinks by 152 tokens. Each bucket is a separate graph shape, so `Policy` compiles/captures **all buckets on the first
request** (`_warmup_prompt_lengths`). A new prompt length can then never stall a running episode.
PyTorch: 59.3 → 54.5 ms; JAX: 49.5 → 45.2 ms.

### 2.4 `torch.compile` *under* the CUDA graph (`13849fb`)

The CUDA graph removes launch overhead but still replays eager kernels: separate kernels for RMSNorm, rotary,
GELU-gated MLP, adaRMS scale/shift, residual gates, dtype casts. The branch sets the default
`pytorch_compile_mode` to `max-autotune-no-cudagraphs`:

- Inductor fuses those pointwise chains into Triton kernels (fewer, bigger kernels, less memory traffic).
- `max-autotune` benchmarks GEMM/Triton choices for each shape on this GPU.
- **`-no-cudagraphs`** because the Policy already captures the whole inference. Inductor's own CUDA graphs
  (`max-autotune`, `reduce-overhead`) would nest/conflict with it and only cover compiled regions. `Policy` warns and
  disables its own graph if one of those modes is configured.

**Effect.** 54.5 → **34.4 ms**. Cost: first-request compilation + autotuning of ~7–11 min per deployment on the
benchmark host (19.5 min on the 12-core container used for the all-suite eval). It is cached in
`TORCHINDUCTOR_CACHE_DIR`. Needs Triton 3.5 (torch 2.9.1); Triton 3.4 segfaults compiling SigLIP on sm_120.

### 2.5 NVFP4 for the PaliGemma language model (`3c2f4ee`)

`--pytorch-quantization nvfp4` runs `PI0Pytorch.quantize_language_model("nvfp4")` after loading, which applies
torchao's `NVFP4InferenceConfig` to the **126 bf16 `nn.Linear`s of the Gemma-2B LM** (18 layers × q/k/v/o/gate/up/down):

- **Format:** 4-bit E2M1 values, one FP8 (E4M3) scale per block of 16 values, plus a per-tensor scale.
- **W4A4, dynamic:** weights are quantized once, and activations are quantized per call (Triton kernel for the
  scale computation). The matmul runs as a native FP4 GEMM (`torch._scaled_mm` on `float4_e2m1fn_x2` → cuBLASLt) on
  Blackwell (sm_100+), so this is real FP4 compute, not weight-only dequantization.
- **Scope is deliberate:** only the prefix LM, which is ~80 % of the FLOPs (§1). The action expert is left in bf16:
  at batch size 1 its GEMMs see ~10 tokens, and quantizing the activation costs more than the smaller GEMM saves.
  SigLIP stays bf16, which also keeps visual features at full precision.
- Quantization happens before compilation and capture, so the FP4 kernels are compiled and captured like everything
  else.

**Effect.** 34.4 → **22.9 ms**. Actions move by up to 1.0e-2 (max-abs vs JAX, normalized space), versus 4.0e-3 for bf16.
That is why closed-loop success was checked: 100/100 on LIBERO-Spatial originally, and 96.75 % over all
2,000 episodes of the four suites with the final code (openpi reports 96.85 % for this checkpoint).
JAX has no FP4 matmul kernels, which is why the served fast path is PyTorch.

FP8 (torchao dynamic per-row) is wired up (`--pytorch-quantization fp8`) but was not benchmarked to completion.

### 2.6 Skip camera slots that are masked out (`813e805`)

π0.5's input format has three image slots (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`). LIBERO has two
cameras, so `LiberoInputs` fills the third with zeros and sets its mask to False. Upstream still runs SigLIP on that
black image and carries its 256 tokens through all 18 prefix layers, where they are masked out of attention.
`drop_masked_image_slots` removes a slot whose mask is False for the whole request before the graph is built
(`preprocess_observation_pytorch` accepts the missing key at inference). Masked tokens never influenced attention or
positions, so this is **exact** (2.5e-3 vs 3 slots, the fp noise between two differently shaped compiled graphs).
It skips one SigLIP pass, and the prefix goes from 816 to 560 tokens, which also shortens every suffix attention.

### 2.7 JAX path (served with `--env LIBERO` defaults; not the fastest path)

The branch also speeds up the JAX policy (63.3 → 45.2 ms) for users who stay on JAX:

| Commit | Change | Effect |
| --- | --- | --- |
| `b037fa8` | batched SigLIP (§2.2) | 63.3 → 59.7 ms, bit-identical |
| `6240e72` | the whole `Policy.infer` (batch dim, uint8→float, RNG split, sampling) is **one `jax.jit` call** instead of eager array ops around a jitted `sample_actions` | → 51.4 ms |
| `c4560f7` | `XLA_FLAGS` records all op types into XLA command buffers (XLA's CUDA graphs) in `serve_policy.py` | → 50.3 ms |
| `5d74cd8` | torch 2.8+ pin so JAX and torch share cuBLAS 12.8 / cuDNN 9.10 | → 49.5 ms |
| `aedbab5` | prompt buckets (§2.3) | → **45.2 ms** |

### 2.8 Tried and not adopted

| Attempt | Result | Why not adopted |
| --- | --- | --- |
| JAX: denoising `while_loop` → `scan` | 63.8 ms (baseline 63.3) | no gain |
| JAX: XLA command buffers off | 56.6 ms (vs 50.3 on) | slower |
| JAX: Triton GEMM / cuDNN fused-MHA XLA flags | no gain | – |
| JAX: cuDNN fused attention via `jax.nn.dot_product_attention` | 54.6 ms | slower: the padding mask forces the non-flash kernel |
| JAX: NVFP4 | not possible | JAX has no FP4 matmul kernels |
| PyTorch: Inductor's own CUDA graphs (`max-autotune`, `reduce-overhead`) | not used | conflicts with the whole-inference graph (§2.1), which also covers the non-compiled parts |
| PyTorch: quantize the action expert as well | not used | ~10 tokens per GEMM at batch size 1; activation quantization costs more than it saves |
| PyTorch: FP8 (torchao dynamic per-row) | not finished | autotuning (92 kernel choices per shape) too slow on the benchmark host; NVFP4 prioritized. Still selectable with `--pytorch-quantization fp8` |
| torch 2.8 / Triton 3.4 | segfault | crashes in `make_llir` compiling SigLIP on sm_120 → torch 2.9.1 / Triton 3.5 |
| FLASH speculative inference (draft head, action-expert verifier, DSpark confidence head; `scripts/flash/`) | flash rounds 8.4 ms, 1.83× per action (H100 bf16, offline) | accepted actions are 0.196 RMS from the policy (its own spread: 0.036), the draft overfits, the verifier barely separates good from bad drafts, no closed-loop rollout; see [`pi05_rtx5090_latency.md`](pi05_rtx5090_latency.md#flash-speculative-inference-faster-but-accepted-actions-deviate-from-the-policy) |

---

## 3. Summary: what makes it fast

| Layer | Contribution |
| --- | --- |
| **openpi / π0.5 architecture** | VLM prefix computed once with a KV cache; 300M action expert over only 10 tokens; 10-step flow matching; action chunks of 10 with replanning every 5; MQA. This makes the problem "one 2B prefill + ten tiny decodes" instead of ten 2B passes or autoregressive decoding. |
| **Branch: remove host overhead** | No syncs or host copies in the loop, then one CUDA graph for the whole request (4× on its own). JAX: one jitted call + command buffers. |
| **Branch: remove wasted tokens** | Prompt buckets (−152 padding tokens) and skipped masked camera slots (−256 tokens, −1 SigLIP pass). Both are exact. |
| **Branch: faster kernels** | Inductor-fused and autotuned kernels inside the graph (−37 %). |
| **Branch: cheaper prefix math** | NVFP4 W4A4 GEMMs on Blackwell for the 2B LM, the FLOP-dominant part (−33 %). Validated in closed loop on all four LIBERO suites. |

## 4. Costs and limits

- **Warm-up:** the first request compiles, autotunes and captures every prompt bucket: ~10–20 min on slow hosts. Warm
  `TORCHINDUCTOR_CACHE_DIR` once per deployment image. Clients must not time out on the first request.
- **Static shapes:** batch size 1, one graph per (prompt bucket, camera-slot set). Different shapes trigger another
  capture. Graph memory is held for the process lifetime.
- **Hardware:** NVFP4 W4A4 needs a Blackwell GPU (sm_100+, e.g. RTX 5090, B200). Without quantization the path still
  gets steps 1–4 and 6 (~34 ms class).
- **Numerics:** NVFP4 changes outputs slightly (1e-2 max-abs). Validate task success per checkpoint/task family before
  deploying, as done here for `pi05_libero`.
- **Version pins:** torch 2.9.1 (CUDA 12.8 build, Triton 3.5) and torchao 0.14.1. torchao logs that its C++ extensions are
  skipped for this torch version; the NVFP4 path used here (Triton + `torch._scaled_mm`) does not need them.
