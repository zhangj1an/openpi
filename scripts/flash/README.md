# FLASH speculative inference for π0.5 (PyTorch)

Implementation of [Realtime-VLA FLASH](https://arxiv.org/abs/2605.13778) for openpi's PyTorch pi0/pi05 models, adapted
from [dexmal/realtime-vla-flash](https://github.com/dexmal/realtime-vla-flash) (Apache-2.0, π0 only).

| Step | Script | Status |
| --- | --- | --- |
| 1. Teacher targets | `make_teacher_targets.py` | tested (RTX 5090: 35 frames/s with NVFP4) |
| 2. Draft training | `train_draft.py` | tested on a 300-frame smoke run; full run in progress |
| 3. Serving | `serve_flash_policy.py` + `openpi.policies.flash_policy.FlashPolicy` | **not yet validated end to end** |

## Setup (any Blackwell GPU for NVFP4, e.g. B200)

```bash
git clone -b pi05-bs1-latency https://github.com/zhangj1an/openpi.git && cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
# openpi's PyTorch model needs its transformers patches:
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

# JAX checkpoint -> PyTorch checkpoint (float32), then copy the norm stats next to it
uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"
uv run examples/convert_jax_model_to_pytorch.py --config_name pi05_libero --precision float32 \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero --output_path /data/pi05_libero_pytorch
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets /data/pi05_libero_pytorch/

# LIBERO training data (LeRobot format, 23 GB for all suites; LIBERO-Spatial is dataset tasks 30-39, ~6 GB)
huggingface-cli download physical-intelligence/libero --repo-type dataset --local-dir /data/libero
```

## 1. Teacher targets

Use the same model you will deploy (here NVFP4). Zero noise, normalized action space, one chunk per training frame.

```bash
export TORCHINDUCTOR_CACHE_DIR=/data/inductor_cache
uv run scripts/flash/make_teacher_targets.py \
    --dataset-dir /data/libero --task-indices 30 31 32 33 34 35 36 37 38 39 \
    --checkpoint-dir /data/pi05_libero_pytorch --pytorch-quantization nvfp4 \
    --output /data/teacher_libero_spatial_nvfp4.npz
```

The first ~1-10 min is `torch.compile` + CUDA graph capture; then it runs one frame per inference (batch size 1).

## 2. Draft training

```bash
uv run scripts/flash/train_draft.py \
    --dataset-dir /data/libero --checkpoint-dir /data/pi05_libero_pytorch \
    --teacher /data/teacher_libero_spatial_nvfp4.npz --output /data/draft_libero_spatial \
    --batch-size 128 --epochs 20 --decode-threads 32
```

- The draft is one Gemma-2B decoder block (initialized from VLM layer 0) + state token + learned action queries +
  linear head, ~110M parameters. It regresses the teacher chunks with a step-weighted Huber loss.
- Prefix embeddings (frozen SigLIP + prompt embeddings) are recomputed every step instead of cached, so the CPU
  (PNG decode + transforms, `--decode-threads`) and the frozen SigLIP forward are part of the step time.
- Paper setting: 100 epochs, batch 64, AdamW 2e-3, 4× RTX 4090D ~6 h with cached prefixes. Watch
  `metrics.jsonl`: `val_frac_chunks_all_within_0.15` approximates how often a draft chunk would be accepted over the
  replan window. The best checkpoint by `val_rms_dist_exec` is written to `draft.safetensors`.
- Masked camera slots are dropped (same as serving), so a single-camera robot trains and serves on 256 image tokens.

## 3. Serving (not yet validated)

```bash
uv run scripts/flash/serve_flash_policy.py --checkpoint-dir /data/pi05_libero_pytorch \
    --draft-dir /data/draft_libero_spatial --pytorch-quantization nvfp4 --port 8000
```

Clients send `flash_reset: True` with the first observation of each episode and execute at most `--max-exec-steps`
actions per response (`accepted_prefix_len` tells how many were accepted). A rejected draft or a predicted gripper switch
falls back to a full round within the same request.
