"""Train a FLASH draft head for a pi0 / pi05 PyTorch policy (Realtime-VLA FLASH, arXiv:2605.13778).

The draft regresses teacher action chunks (scripts/flash/make_teacher_targets.py) from the frozen policy's prefix
embeddings (SigLIP image tokens + prompt embeddings) and the normalized robot state. By default prefix embeddings are
recomputed on the fly. With --cache-prefixes they are computed once and kept in host RAM (bfloat16, ~2.3 MB per frame
for 560 tokens, e.g. ~120 GB for LIBERO-Spatial), which removes PNG decoding and the SigLIP forward from every step.

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
    # Keep every frame's prefix embeddings in host RAM instead of recomputing them each step (see module docstring).
    cache_prefixes: bool = False
    cache_batch_size: int = 128
    # Write a resumable checkpoint (draft, optimizer, data order) every this many steps; resumed on restart.
    checkpoint_every: int = 2000
    # DSpark-style confidence head (arXiv:2607.05147), trained jointly: per-action probability that the verifier
    # accepts the action, supervised with BCE against "within `accept_threshold` of the teacher" (the verifier's
    # distance, with the teacher standing in for the Action Expert reconstructions).
    confidence: bool = False
    conf_weight: float = 1.0
    accept_threshold: float = 0.15


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


class _PrefixCache:
    """Prefix embeddings, masks, and states of every frame in host RAM, gathered into pinned buffers per batch."""

    def __init__(self, embs: torch.Tensor, pad: torch.Tensor, att: torch.Tensor, state: torch.Tensor):
        self.embs, self.pad, self.att, self.state = embs, pad, att, state
        self._device = torch.device("cuda")
        # Two pinned buffer sets per batch size, used alternately; each is reused only once its copy has finished.
        self._buffers: dict[int, list] = {}
        self._turn = 0

    def batch(self, idx: np.ndarray):
        sources = (self.embs, self.pad, self.att, self.state)
        if len(idx) not in self._buffers:
            self._buffers[len(idx)] = [
                (
                    [torch.empty((len(idx), *s.shape[1:]), dtype=s.dtype, pin_memory=True) for s in sources],
                    torch.cuda.Event(),
                )
                for _ in range(2)
            ]
        self._turn ^= 1
        bufs, copied = self._buffers[len(idx)][self._turn]
        copied.synchronize()  # the previous copy out of these buffers
        index = torch.from_numpy(idx)
        out = []
        for src, buf in zip(sources, bufs, strict=True):
            torch.index_select(src, 0, index, out=buf)
            out.append(buf.to(self._device, non_blocking=True))
        copied.record()
        return tuple(out)


def _build_prefix_cache(compute_prefix_batch, n: int, batch_size: int) -> _PrefixCache:
    embs = pad = att = state = None
    start = time.monotonic()
    for lo in range(0, n, batch_size):
        idx = np.arange(lo, min(lo + batch_size, n))
        e, p, a, s = compute_prefix_batch(idx)
        if embs is None:
            logging.info(
                "Caching prefixes: %d frames x %s %s (%.1f GB host RAM)",
                n,
                tuple(e.shape[1:]),
                e.dtype,
                n * e[0].numel() * e.element_size() / 1e9,
            )
            embs = torch.empty((n, *e.shape[1:]), dtype=e.dtype)
            pad = torch.empty((n, *p.shape[1:]), dtype=p.dtype)
            att = torch.empty((n, *a.shape[1:]), dtype=a.dtype)
            state = torch.empty((n, *s.shape[1:]), dtype=s.dtype)
        assert e.shape[1:] == embs.shape[1:], f"prefix length changed: {tuple(e.shape)} vs {tuple(embs.shape)}"
        embs[lo : lo + len(idx)].copy_(e)
        pad[lo : lo + len(idx)].copy_(p)
        att[lo : lo + len(idx)].copy_(a)
        state[lo : lo + len(idx)].copy_(s)
        if (lo // batch_size) % 50 == 0:
            logging.info("cached %d/%d frames (%.0f s)", lo + len(idx), n, time.monotonic() - start)
    logging.info("Prefix cache built in %.1f min", (time.monotonic() - start) / 60)
    return _PrefixCache(embs, pad, att, state)


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

    draft = flash.DraftChunkHead(text_config, chunk_len=horizon, action_dim=7, confidence=args.confidence).to(device)
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

    def compute_prefix_batch(idx: np.ndarray):
        items = list(pool.map(make_inputs, idx.tolist()))
        batch = jax.tree.map(lambda *xs: torch.from_numpy(np.stack(xs)).to(device), *items)
        observation = _model.Observation.from_dict(batch)
        with torch.no_grad():
            images, img_masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
            prefix_embs, pad, att = model.embed_prefix(images, img_masks, tokens, token_masks)
        return prefix_embs, pad, att, state.float()

    prefix_batch = compute_prefix_batch
    if args.cache_prefixes:
        cache = _build_prefix_cache(compute_prefix_batch, len(frames), args.cache_batch_size)
        del model  # only the cached embeddings are needed from here on
        torch.cuda.empty_cache()
        prefix_batch = cache.batch

    accept_config = flash.FlashConfig(threshold=args.accept_threshold, max_exec_steps=args.exec_steps)

    def predict(prefix_embs, pad, att, state):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.confidence:
                return draft.forward_with_confidence(prefix_embs, pad, att, state)
            return draft(prefix_embs, pad, att, state), None

    def loss_fn(pred, tgt):
        per = F.smooth_l1_loss(pred, tgt, reduction="none", beta=args.huber_beta).mean(dim=-1)  # (B, H)
        return (per * weights).sum(dim=-1).mean()

    def conf_loss_fn(pred, conf_logits, tgt):
        labels = flash.step_accepted(pred.detach(), tgt, accept_config).float()  # (B, H)
        return F.binary_cross_entropy_with_logits(conf_logits, labels), labels

    @torch.no_grad()
    def evaluate(max_batches: int = 20) -> dict:
        draft.eval()
        dists, grip_ok, loss, conf_losses, probs, labels = [], [], [], [], [], []
        e = args.exec_steps
        for b in range(min(max_batches, math.ceil(len(val_idx) / args.batch_size))):
            idx = val_idx[b * args.batch_size : (b + 1) * args.batch_size]
            pred, conf_logits = predict(*prefix_batch(idx))
            tgt = targets[idx].to(device)
            loss.append(loss_fn(pred, tgt).item())
            dists.append((torch.linalg.vector_norm(pred[:, :e, :6] - tgt[:, :e, :6], dim=-1) / math.sqrt(6)).cpu())
            grip_ok.append(((pred[:, :e, 6] < 0) == (tgt[:, :e, 6] < 0)).float().mean().item())
            if conf_logits is not None:
                conf_loss, step_labels = conf_loss_fn(pred, conf_logits, tgt)
                conf_losses.append(conf_loss.item())
                probs.append(torch.sigmoid(conf_logits[:, :e]).cpu())
                labels.append(step_labels[:, :e].cpu())
        draft.train()
        d = torch.cat(dists)
        metrics = {
            "val_loss": float(np.mean(loss)),
            "val_rms_dist_exec": float(d.mean()),
            "val_frac_steps_within_0.15": float((d <= 0.15).float().mean()),
            "val_frac_chunks_all_within_0.15": float((d <= 0.15).all(dim=1).float().mean()),
            "val_gripper_sign_acc": float(np.mean(grip_ok)),
        }
        if probs:
            p, y = torch.cat(probs), torch.cat(labels)  # (N, exec_steps)
            survival = torch.cumprod(p, dim=1)
            accepted = torch.cumprod(y, dim=1)
            none_accepted = accepted[:, 0] == 0
            predicted_none = survival[:, 0] < 0.5
            metrics |= {
                "val_conf_bce": float(np.mean(conf_losses)),
                "val_conf_step_acc": float(((p >= 0.5) == (y == 1)).float().mean()),
                "val_conf_mean_prob": float(p.mean()),
                "val_accept_rate": float(y.mean()),
                # Expected accepted prefix (sum of survival probabilities) against the teacher-proxy prefix.
                "val_prefix_expected": float(survival.sum(dim=1).mean()),
                "val_prefix_actual": float(accepted.sum(dim=1).mean()),
                "val_prefix_mae": float((survival.sum(dim=1) - accepted.sum(dim=1)).abs().mean()),
                # Rounds with nothing accepted, which a scheduler would send straight to the full path.
                "val_reject_frac": float(none_accepted.float().mean()),
                "val_reject_recall": float((predicted_none & none_accepted).sum() / none_accepted.sum().clamp(min=1)),
                "val_reject_precision": float(
                    (predicted_none & none_accepted).sum() / predicted_none.sum().clamp(min=1)
                ),
            }
        return metrics

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    step, epoch_perm, cursor = 0, rng.permutation(train_idx), 0
    resume_path = out / "last.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        draft.load_state_dict(ckpt["draft"])
        optim.load_state_dict(ckpt["optim"])
        step, cursor, best, epoch_perm = ckpt["step"], ckpt["cursor"], ckpt["best"], ckpt["epoch_perm"]
        rng.bit_generator.state = ckpt["rng"]
        logging.info("Resumed from %s at step %d", resume_path, step)
    start, start_step = time.monotonic(), step
    while step < total_steps:
        if cursor + args.batch_size > len(epoch_perm):
            epoch_perm, cursor = rng.permutation(train_idx), 0
        idx = epoch_perm[cursor : cursor + args.batch_size]
        cursor += args.batch_size

        lr_scale = min(1.0, (step + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * step / total_steps))
        for group, base in zip(optim.param_groups, base_lrs, strict=True):
            group["lr"] = base * lr_scale
        pred, conf_logits = predict(*prefix_batch(idx))
        tgt = targets[idx].to(device)
        action_loss = loss_fn(pred, tgt)
        loss = action_loss
        if conf_logits is not None:
            conf_loss = conf_loss_fn(pred, conf_logits, tgt)[0]
            loss = action_loss + args.conf_weight * conf_loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optim.step()
        step += 1

        if step % 50 == 0:
            logging.info(
                "step %d/%d (epoch %.2f) action loss %.4f%s  %.3f s/step",
                step,
                total_steps,
                step * args.batch_size / len(train_idx),
                action_loss.item(),
                f"  conf bce {conf_loss.item():.4f}" if conf_logits is not None else "",
                (time.monotonic() - start) / (step - start_step),
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
                    json.dumps(
                        {
                            "config": args.config,
                            "chunk_len": horizon,
                            "action_dim": 7,
                            "confidence": args.confidence,
                            "accept_threshold": args.accept_threshold,
                            "step": step,
                            **metrics,
                        }
                    )
                )
        if step % args.checkpoint_every == 0 and step < total_steps:
            ckpt = {
                "draft": draft.state_dict(),
                "optim": optim.state_dict(),
                "step": step,
                "cursor": cursor,
                "best": best,
                "epoch_perm": epoch_perm,
                "rng": rng.bit_generator.state,
            }
            torch.save(ckpt, out / "last.pt.tmp")
            (out / "last.pt.tmp").replace(resume_path)
    resume_path.unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
