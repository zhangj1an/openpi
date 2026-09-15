"""Measure FLASH acceptance under the real verifier by replaying recorded episodes through `FlashPolicy`.

Each episode is served as a client would: the first request resets (full round), every response is followed by a
request at the frame after the executed actions (a full round executes `max_exec_steps` actions, a flash round its
accepted prefix). Flash rounds therefore verify against the KV cache of the last full round, several frames old, exactly
as in serving. The observations come from the demonstration rather than from executing the policy's own actions, so
this measures acceptance along expert trajectories, not closed-loop success.

    uv run scripts/flash/eval_verifier_offline.py --dataset-dir /data/libero --checkpoint-dir /ckpt/pi05_libero_pytorch \
        --draft-dir draft_libero_spatial --teacher teacher_libero_spatial.npz --output verifier_val.json
"""

import dataclasses
import io
import json
import logging
import pathlib
import time

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import safetensors.torch
import torch
import tyro

from openpi.models_pytorch import flash as _flash
from openpi.policies import flash_policy as _flash_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    dataset_dir: str
    checkpoint_dir: str
    draft_dir: str
    # Teacher file of the draft's training run: its episodes and seed reproduce the train / validation split.
    teacher: str
    output: str
    config: str = "pi05_libero"
    token_len_buckets: tuple[int, ...] = (48,)
    pytorch_quantization: str | None = None
    split: str = "val"  # "val", "train", or "all"
    max_episodes: int | None = None
    val_episode_frac: float = 0.05  # must match train_draft.py
    seed: int = 0  # must match train_draft.py
    noise_seed: int = 0
    verify_timesteps: tuple[float, ...] = (0.10, 0.05)
    threshold: float = 0.15
    max_exec_steps: int = 5
    full_every_n_flash_rounds: int = 0
    # Rounds served (and discarded) first, so CUDA graph capture does not count toward latency.
    warmup_rounds: int = 12


def _episodes(args: Args) -> list[int]:
    teacher = np.load(args.teacher)
    episodes = np.unique(teacher["episode_index"])
    rng = np.random.default_rng(args.seed)  # the same first draw as train_draft.py
    val = set(rng.choice(episodes, max(1, int(len(episodes) * args.val_episode_frac)), replace=False).tolist())
    chosen = {
        "val": [e for e in episodes if e in val],
        "train": [e for e in episodes if e not in val],
        "all": list(episodes),
    }[args.split]
    return [int(e) for e in chosen[: args.max_episodes]]


def _load_episode(root: pathlib.Path, info: dict, tasks: dict, episode: int) -> list[dict]:
    path = root / info["data_path"].format(
        episode_chunk=episode // info.get("chunks_size", 1000), episode_index=episode
    )
    table = pq.read_table(path, columns=["image", "wrist_image", "state", "task_index"])
    return [
        {
            "observation/image": np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")),
            "observation/wrist_image": np.asarray(Image.open(io.BytesIO(r["wrist_image"]["bytes"])).convert("RGB")),
            "observation/state": np.asarray(r["state"], dtype=np.float32),
            "prompt": tasks[r["task_index"]],
        }
        for r in table.to_pylist()
    ]


def _serve_episode(policy: _flash_policy.FlashPolicy, frames: list[dict], max_exec_steps: int) -> list[dict]:
    rounds, f, executed = [], 0, None
    while f < len(frames):
        obs = dict(frames[f])
        if executed is None:
            obs["flash_reset"] = True
        else:
            obs["executed_steps"] = executed
        out = policy.infer(obs)
        returned = len(out["actions"])
        executed = min(returned, max_exec_steps)
        rounds.append({"frame": f, "executed": executed, "infer_ms": out["policy_timing"]["infer_ms"], **out["flash"]})
        f += executed
    return rounds


def _teacher_distances(rounds: list[dict], teacher: dict, args: Args) -> dict:
    """RMS distance (first 6 dims, normalized space) of executed actions to the teacher's zero-noise chunk on that frame.

    For flash rounds these are the accepted draft actions; for full rounds the policy's own actions with random noise,
    which shows how far a correct action can be from the zero-noise sample.
    """
    by_kind: dict[str, list[float]] = {}
    for r in rounds:
        target = teacher.get((r["episode"], r["frame"]))
        if target is None:
            continue
        n = r["executed"]
        executed = np.asarray(r["draft"] if r["round"] == "flash" else r["actions"])[:n, :6]
        d = np.linalg.norm(executed - target[:n, :6], axis=-1) / np.sqrt(6)
        by_kind.setdefault("flash" if r["round"] == "flash" else "full", []).extend(d.tolist())
    return {
        kind: {
            "steps": len(d),
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "p90": float(np.percentile(d, 90)),
            "frac_within_0.15": float(np.mean(np.asarray(d) <= 0.15)),
        }
        for kind, d in by_kind.items()
    }


def _summary(rounds: list[dict], args: Args, teacher: dict) -> dict:
    kinds = [r["round"] for r in rounds]
    attempts = [r for r in rounds if "draft_accepted" in r]
    n = len(rounds)
    ms = {k: [r["infer_ms"] for r in rounds if r["round"] == k] for k in set(kinds)}
    executed = sum(r["executed"] for r in rounds)
    full_ms = float(np.median(ms["full"])) if ms.get("full") else float("nan")
    total_ms = sum(r["infer_ms"] for r in rounds)
    summary = {
        "rounds": n,
        "executed_actions": executed,
        "round_kinds": {k: kinds.count(k) / n for k in sorted(set(kinds))},
        "flash_attempts": len(attempts),
        # Among flash attempts: how often the verifier accepted a prefix, and its average length.
        "attempt_accept_rate": float(np.mean([r["draft_accepted"] > 0 for r in attempts])) if attempts else None,
        "attempt_mean_accepted": float(np.mean([r["draft_accepted"] for r in attempts])) if attempts else None,
        "attempt_accepted_hist": np.bincount(
            [r["draft_accepted"] for r in attempts], minlength=args.max_exec_steps + 1
        ).tolist()
        if attempts
        else None,
        "attempt_gripper_switch_rate": float(np.mean([r["gripper_switch"] for r in attempts])) if attempts else None,
        # Paper-style: accepted prefix of flash rounds normalized by the replan window, and flash-path round share.
        "flash_round_mean_accepted_frac": float(
            np.mean([r["draft_accepted"] / args.max_exec_steps for r in rounds if r["round"] == "flash"])
        )
        if "flash" in kinds
        else 0.0,
        "flash_path_rate": kinds.count("flash") / n,
        "median_infer_ms": {k: float(np.median(v)) for k, v in ms.items()},
        # Latency per executed action on this GPU and precision, against serving every round as a full round.
        "ms_per_action": total_ms / executed,
        "full_only_ms_per_action": full_ms / args.max_exec_steps,
        "speedup_per_action": (full_ms / args.max_exec_steps) / (total_ms / executed),
    }
    summary["executed_vs_teacher"] = _teacher_distances(rounds, teacher, args)
    # Acceptance at other thresholds, from the recorded per-step distances (prefix up to the first failing step).
    if attempts:
        dists = np.array([r["step_dist"] for r in attempts])  # (attempts, exec_steps)
        summary["attempt_mean_accepted_by_threshold"] = {
            str(t): float(np.cumprod(dists <= t, axis=1).sum(axis=1).mean()) for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
        }
    return summary


def main(args: Args) -> None:
    root = pathlib.Path(args.dataset_dir)
    info = json.loads((root / "meta/info.json").read_text())
    tasks = {t["task_index"]: t["task"] for t in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
    episodes = _episodes(args)
    t = np.load(args.teacher)
    teacher = {(int(e), int(f)): x for e, f, x in zip(t["episode_index"], t["frame_index"], t["targets"], strict=True)}
    logging.info("%s split: %d episodes", args.split, len(episodes))

    train_config = _config.get_config(args.config)
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(
            train_config.model, pytorch_quantization=args.pytorch_quantization, pytorch_compile_mode=None
        ),
    )
    policy = _policy_config.create_trained_policy(
        train_config, args.checkpoint_dir, token_len_buckets=args.token_len_buckets
    )
    model = policy._model  # noqa: SLF001
    draft_dir = pathlib.Path(args.draft_dir)
    draft_meta = json.loads((draft_dir / "draft_meta.json").read_text())
    draft = _flash.DraftChunkHead(
        model.paligemma_with_expert.paligemma.language_model.config,
        chunk_len=train_config.model.action_horizon,
        action_dim=7,
        confidence=draft_meta.get("confidence", False),
    )
    safetensors.torch.load_model(draft, str(draft_dir / "draft.safetensors"))
    flash_config = _flash.FlashConfig(
        verify_timesteps=args.verify_timesteps,
        threshold=args.threshold,
        max_exec_steps=args.max_exec_steps,
        full_every_n_flash_rounds=args.full_every_n_flash_rounds,
    )
    flash = _flash_policy.FlashPolicy(policy, draft, flash_config, diagnostics=True)
    torch.manual_seed(args.noise_seed)

    warm = _load_episode(root, info, tasks, episodes[0])
    warm_rounds = _serve_episode(flash, warm[: args.warmup_rounds * args.max_exec_steps], args.max_exec_steps)
    logging.info("warmup: %d rounds (%s)", len(warm_rounds), sorted({r["round"] for r in warm_rounds}))

    results, all_rounds, start = [], [], time.monotonic()
    for i, episode in enumerate(episodes):
        rounds = [
            {"episode": episode, **r}
            for r in _serve_episode(flash, _load_episode(root, info, tasks, episode), args.max_exec_steps)
        ]
        results.append({"episode": episode, "rounds": rounds})
        all_rounds.extend(rounds)
        s = _summary(all_rounds, args, teacher)
        logging.info(
            "episode %d/%d: flash-path %.1f %%, attempt accept %.1f %%, mean accepted %.2f, %.2fx per action (%.0f s)",
            i + 1,
            len(episodes),
            100 * s["flash_path_rate"],
            100 * (s["attempt_accept_rate"] or 0),
            s["attempt_mean_accepted"] or 0,
            s["speedup_per_action"],
            time.monotonic() - start,
        )
    summary = _summary(all_rounds, args, teacher)
    logging.info("summary %s", json.dumps(summary))
    out = {
        "args": dataclasses.asdict(args),
        "draft_meta": draft_meta,
        "gpu": torch.cuda.get_device_name(),
        "summary": summary,
        "episodes": results,
    }
    pathlib.Path(args.output).write_text(json.dumps(out))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
