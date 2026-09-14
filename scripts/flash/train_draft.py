"""Train a FLASH draft head for a pi0 / pi05 PyTorch policy (Realtime-VLA FLASH, arXiv:2605.13778).

The draft regresses teacher action chunks (scripts/flash/make_teacher_targets.py) from the frozen policy's prefix
embeddings (SigLIP image tokens + prompt embeddings) and the normalized robot state. Prefix embeddings are recomputed
on the fly instead of cached, since a cache of 800 x 2048 tokens per frame does not fit on small disks.

    uv run scripts/flash/train_draft.py --dataset-dir /data/libero --checkpoint-dir /ckpt/pi05_libero_pytorch \
        --teacher teacher_libero_spatial.npz --output draft_libero_spatial
"""

import concurrent.futures
import dataclasses
import io
import json
import logging
import math
import pathlib
import time

import jax
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import safetensors.torch
import torch
import torch.nn.functional as F  # noqa: N812
import tyro

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.models_pytorch import flash
from openpi.policies.policy import drop_masked_image_slots
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as _transforms


@dataclasses.dataclass
class Args:
    dataset_dir: str
    checkpoint_dir: str
    teacher: str
    output: str
    config: str = "pi05_libero"
    token_len_buckets: tuple[int, ...] = (48,)
    batch_size: int = 32
    epochs: float = 5.0
    lr_new: float = 2e-3  # state token, action queries, action head
    lr_block: float = 2e-4  # Gemma block initialized from the VLM's first layer
    weight_decay: float = 0.01
    warmup_steps: int = 300
    # Step weights: gamma**h over the first `exec_steps` actions, `tail_weight` for the rest (normalized to sum 1).
    exec_steps: int = 5
    gamma: float = 0.9
    tail_weight: float = 0.1
    huber_beta: float = 1.0
    val_episode_frac: float = 0.05
    eval_every: int = 500
    decode_threads: int = 12
    seed: int = 0


def _input_transforms(train_config, checkpoint_dir: str, buckets):
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id)
    model_transforms = [
        dataclasses.replace(t, tokenizer=_tokenizer.PaligemmaTokenizer(train_config.model.max_token_len, buckets))
        if isinstance(t, _transforms.TokenizePrompt)
        else t
        for t in data_config.model_transforms.inputs
    ]
    return _transforms.compose(
        [
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *model_transforms,
        ]
    )


def _load_frames(root: pathlib.Path, teacher) -> list[dict]:
    info = json.loads((root / "meta/info.json").read_text())
    tasks = {t["task_index"]: t["task"] for t in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
    chunk_size = info.get("chunks_size", 1000)
    frames: list[dict] = []
    for ep in dict.fromkeys(teacher["episode_index"].tolist()):
        path = root / info["data_path"].format(episode_chunk=ep // chunk_size, episode_index=ep)
        table = pq.read_table(path, columns=["image", "wrist_image", "state", "frame_index", "task_index"])
        frames.extend(
            {
                "image": r["image"]["bytes"],
                "wrist": r["wrist_image"]["bytes"],
                "state": r["state"],
                "frame": r["frame_index"],
                "episode": ep,
                "prompt": tasks[r["task_index"]],
            }
            for r in table.to_pylist()
        )
    return frames[: len(teacher["targets"])]  # a teacher file may stop mid-episode (--max-frames)


def _step_weights(h: int, args: Args) -> torch.Tensor:
    w = torch.tensor([args.gamma**i if i < args.exec_steps else args.tail_weight for i in range(h)])
    return w / w.sum()


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    train_config = _config.get_config(args.config)
    transform = _input_transforms(train_config, args.checkpoint_dir, args.token_len_buckets)

    teacher = np.load(args.teacher)
    frames = _load_frames(pathlib.Path(args.dataset_dir), teacher)
    assert len(frames) == len(teacher["targets"]), (len(frames), len(teacher["targets"]))
    assert all(f["frame"] == fi for f, fi in zip(frames, teacher["frame_index"], strict=True))
    targets = torch.from_numpy(teacher["targets"])  # (N, H, 7), normalized

    episodes = np.unique(teacher["episode_index"])
    val_eps = set(rng.choice(episodes, max(1, int(len(episodes) * args.val_episode_frac)), replace=False).tolist())
    val_idx = np.array([i for i, f in enumerate(frames) if f["episode"] in val_eps])
    train_idx = np.array([i for i, f in enumerate(frames) if f["episode"] not in val_eps])
    logging.info("frames: %d train / %d val (%d val episodes)", len(train_idx), len(val_idx), len(val_eps))

    model = train_config.model.load_pytorch(train_config, str(pathlib.Path(args.checkpoint_dir) / "model.safetensors"))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to(device).eval()
    model.requires_grad_(requires_grad=False)
    text_config = model.paligemma_with_expert.paligemma.language_model.config
    horizon = train_config.model.action_horizon

    draft = flash.DraftChunkHead(text_config, chunk_len=horizon, action_dim=7).to(device)
    draft.init_from_vlm_layer(model.paligemma_with_expert.paligemma.language_model.layers[0])
    draft.float().train()
    block_params = list(draft.block.parameters())
    new_params = [p for n, p in draft.named_parameters() if not n.startswith("block.")]
    optim = torch.optim.AdamW(
        [{"params": new_params, "lr": args.lr_new}, {"params": block_params, "lr": args.lr_block}],
        weight_decay=args.weight_decay,
    )
    base_lrs = [args.lr_new, args.lr_block]
    total_steps = int(args.epochs * len(train_idx) / args.batch_size)
    weights = _step_weights(horizon, args).to(device)
    pool = concurrent.futures.ThreadPoolExecutor(args.decode_threads)
    logging.info("draft params: %.1fM, steps: %d", sum(p.numel() for p in draft.parameters()) / 1e6, total_steps)

    def make_inputs(i: int) -> dict:
        f = frames[i]
        obs = {
            "observation/image": np.asarray(Image.open(io.BytesIO(f["image"])).convert("RGB")),
            "observation/wrist_image": np.asarray(Image.open(io.BytesIO(f["wrist"])).convert("RGB")),
            "observation/state": np.asarray(f["state"], dtype=np.float32),
            "prompt": f["prompt"],
        }
        return drop_masked_image_slots(transform(obs))  # same inputs as serving: no empty camera slot

    def prefix_batch(idx: np.ndarray):
        items = list(pool.map(make_inputs, idx.tolist()))
        batch = jax.tree.map(lambda *xs: torch.from_numpy(np.stack(xs)).to(device), *items)
        observation = _model.Observation.from_dict(batch)
        with torch.no_grad():
            images, img_masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
            prefix_embs, pad, att = model.embed_prefix(images, img_masks, tokens, token_masks)
        return prefix_embs, pad, att, state.float()

    def loss_fn(pred, tgt):
        per = F.smooth_l1_loss(pred, tgt, reduction="none", beta=args.huber_beta).mean(dim=-1)  # (B, H)
        return (per * weights).sum(dim=-1).mean()

    @torch.no_grad()
    def evaluate(max_batches: int = 20) -> dict:
        draft.eval()
        dists, grip_ok, loss = [], [], []
        for b in range(min(max_batches, math.ceil(len(val_idx) / args.batch_size))):
            idx = val_idx[b * args.batch_size : (b + 1) * args.batch_size]
            prefix_embs, pad, att, state = prefix_batch(idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = draft(prefix_embs, pad, att, state)
            tgt = targets[idx].to(device)
            loss.append(loss_fn(pred, tgt).item())
            e = args.exec_steps
            dists.append((torch.linalg.vector_norm(pred[:, :e, :6] - tgt[:, :e, :6], dim=-1) / math.sqrt(6)).cpu())
            grip_ok.append(((pred[:, :e, 6] < 0) == (tgt[:, :e, 6] < 0)).float().mean().item())
        draft.train()
        d = torch.cat(dists)
        return {
            "val_loss": float(np.mean(loss)),
            "val_rms_dist_exec": float(d.mean()),
            "val_frac_steps_within_0.15": float((d <= 0.15).float().mean()),
            "val_frac_chunks_all_within_0.15": float((d <= 0.15).all(dim=1).float().mean()),
            "val_gripper_sign_acc": float(np.mean(grip_ok)),
        }

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    step, epoch_perm, cursor = 0, rng.permutation(train_idx), 0
    start = time.monotonic()
    while step < total_steps:
        if cursor + args.batch_size > len(epoch_perm):
            epoch_perm, cursor = rng.permutation(train_idx), 0
        idx = epoch_perm[cursor : cursor + args.batch_size]
        cursor += args.batch_size

        lr_scale = min(1.0, (step + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * step / total_steps))
        for group, base in zip(optim.param_groups, base_lrs, strict=True):
            group["lr"] = base * lr_scale
        prefix_embs, pad, att, state = prefix_batch(idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = draft(prefix_embs, pad, att, state)
        loss = loss_fn(pred, targets[idx].to(device))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optim.step()
        step += 1

        if step % 50 == 0:
            logging.info(
                "step %d/%d loss %.4f  %.2f s/step", step, total_steps, loss.item(), (time.monotonic() - start) / step
            )
        if step % args.eval_every == 0 or step == total_steps:
            metrics = {"step": step, **evaluate()}
            logging.info("eval %s", json.dumps(metrics))
            with (out / "metrics.jsonl").open("a") as fh:
                fh.write(json.dumps(metrics) + "\n")
            if metrics["val_rms_dist_exec"] < best:
                best = metrics["val_rms_dist_exec"]
                safetensors.torch.save_model(draft, str(out / "draft.safetensors"))
                (out / "draft_meta.json").write_text(
                    json.dumps({"config": args.config, "chunk_len": horizon, "action_dim": 7, "step": step, **metrics})
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
