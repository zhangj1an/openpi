"""Generate FLASH draft-model regression targets from a trained pi0 / pi05 PyTorch policy.

For every frame of a LeRobot-format dataset (e.g. physical-intelligence/libero), runs the policy's full inference with
zero noise and stores the resulting action chunk in the model's normalized action space. The draft model regresses
these teacher chunks (Realtime-VLA FLASH, arXiv:2605.13778, "teacher_zero_noise" targets).

    uv run scripts/flash/make_teacher_targets.py \
        --dataset-dir /data/libero --task-indices 30 31 32 33 34 35 36 37 38 39 \
        --checkpoint-dir /ckpt/pi05_libero_pytorch --pytorch-quantization nvfp4 \
        --output teacher_libero_spatial.npz
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
import tqdm
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    dataset_dir: str
    checkpoint_dir: str
    output: str
    config: str = "pi05_libero"
    # Only keep episodes with these dataset task indices (physical-intelligence/libero: libero_spatial is 30-39).
    task_indices: tuple[int, ...] | None = None
    pytorch_quantization: str | None = None
    token_len_buckets: tuple[int, ...] = (48,)
    max_frames: int | None = None
    save_every: int = 5000


def _png(cell) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def main(args: Args) -> None:
    root = pathlib.Path(args.dataset_dir)
    info = json.loads((root / "meta/info.json").read_text())
    tasks = {t["task_index"]: t["task"] for t in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    if args.task_indices:
        keep = {tasks[i] for i in args.task_indices}
        episodes = [e for e in episodes if e["tasks"][0] in keep]
    chunk_size = info.get("chunks_size", 1000)
    paths = {
        e["episode_index"]: root
        / info["data_path"].format(episode_chunk=e["episode_index"] // chunk_size, episode_index=e["episode_index"])
        for e in episodes
    }
    if missing := [str(p) for p in paths.values() if not p.exists()]:
        raise FileNotFoundError(f"{len(missing)} episode files are missing, e.g. {missing[:3]}")

    train_config = _config.get_config(args.config)
    if args.pytorch_quantization:
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, pytorch_quantization=args.pytorch_quantization)
        )
    policy = _policy_config.create_trained_policy(
        train_config, args.checkpoint_dir, token_len_buckets=args.token_len_buckets
    )
    model_cfg = train_config.model
    zero_noise = np.zeros((model_cfg.action_horizon, model_cfg.action_dim), dtype=np.float32)

    n_total = sum(e["length"] for e in episodes)
    if args.max_frames:
        n_total = min(n_total, args.max_frames)
    targets = np.zeros((n_total, model_cfg.action_horizon, 7), dtype=np.float32)
    episode_index = np.zeros(n_total, dtype=np.int64)
    frame_index = np.zeros(n_total, dtype=np.int64)
    n = 0
    # Progress is saved every `save_every` frames and resumed on restart.
    partial = pathlib.Path(args.output + ".partial.npz")
    if partial.exists():
        saved = np.load(partial)
        n = int(saved["n"])
        targets[:n], episode_index[:n], frame_index[:n] = saved["targets"], saved["episode_index"], saved["frame_index"]
        logging.info("Resuming from %s at frame %d", partial, n)
    done = n
    start = time.monotonic()
    seen = 0
    with tqdm.tqdm(total=n_total, initial=n) as bar:
        for ep in episodes:
            if seen + ep["length"] <= done:  # already generated before a restart
                seen += ep["length"]
                continue
            table = pq.read_table(
                paths[ep["episode_index"]], columns=["image", "wrist_image", "state", "frame_index", "task_index"]
            )
            for row in table.to_pylist():
                if seen < done:
                    seen += 1
                    continue
                seen += 1
                if n >= n_total:
                    break
                obs = {
                    "observation/image": _png(row["image"]),
                    "observation/wrist_image": _png(row["wrist_image"]),
                    "observation/state": np.asarray(row["state"], dtype=np.float32),
                    "prompt": tasks[row["task_index"]],
                }
                inputs = policy._input_transform(obs)  # noqa: SLF001
                raw = policy._torch_graph_for(inputs)(inputs, zero_noise)  # noqa: SLF001  (normalized action space)
                targets[n] = raw[:, :7]
                episode_index[n] = ep["episode_index"]
                frame_index[n] = row["frame_index"]
                n += 1
                bar.update(1)
                if n % args.save_every == 0:
                    np.savez(
                        partial, n=n, targets=targets[:n], episode_index=episode_index[:n], frame_index=frame_index[:n]
                    )
            if n >= n_total:
                break
    logging.info("Generated %d teacher chunks in %.1f min", n - done, (time.monotonic() - start) / 60)
    np.savez(
        args.output,
        targets=targets[:n],
        episode_index=episode_index[:n],
        frame_index=frame_index[:n],
        meta=json.dumps(
            {
                "config": args.config,
                "checkpoint_dir": args.checkpoint_dir,
                "quantization": args.pytorch_quantization,
                "task_indices": list(args.task_indices or []),
                "space": "normalized",
                "noise": "zeros",
            }
        ),
    )
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
