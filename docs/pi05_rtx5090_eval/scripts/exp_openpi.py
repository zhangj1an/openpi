#!/usr/bin/env python
"""Run one openpi pi05_libero optimisation experiment: latency (default XLA autotune) or exactness (autotune off).

  exp_openpi.py --mode latency --patch scan_loop,batched_siglip --max-token-len 48 --label x
  exp_openpi.py --mode exact   --patch scan_loop --label x      # compares to runs/opt/exact_baseline.npy
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "latency"
if _mode == "exact":
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0").strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(HERE))
import openpi_patches  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["latency", "exact"], default="latency")
    p.add_argument("--patch", default="")
    p.add_argument("--max-token-len", type=int, default=0)
    p.add_argument("--token-len-buckets", default="", help="comma list, e.g. 48")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--label", required=True)
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--ckpt", default="/dev/shm/openpi_data/openpi-assets/checkpoints/pi05_libero")
    args = p.parse_args()
    patches = [x for x in args.patch.split(",") if x]
    openpi_patches.apply(patches)

    cfg = _config.get_config("pi05_libero")
    if args.max_token_len:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, max_token_len=args.max_token_len))
    buckets = tuple(int(x) for x in args.token_len_buckets.split(",") if x)
    policy = policy_config.create_trained_policy(cfg, args.ckpt, token_len_buckets=buckets or None)
    d = np.load(HERE / "libero_obs_sample.npz")
    obs = {"observation/image": d["image"], "observation/wrist_image": d["wrist_image"],
           "observation/state": d["state"], "prompt": str(d["prompt"])}
    noise = np.load(HERE / "runs/parity_openpi_ref.npz")["noise"]

    t = time.perf_counter()
    actions = policy.infer(dict(obs), noise=noise)["actions"]
    first_ms = (time.perf_counter() - t) * 1e3
    res = {"label": args.label, "mode": args.mode, "patches": patches, "max_token_len": cfg.model.max_token_len, "token_len_buckets": list(buckets),
           "first_call_ms": first_ms}

    base = HERE / "runs" / "opt" / f"{args.mode}_baseline.npy"
    base.parent.mkdir(parents=True, exist_ok=True)
    if args.save_baseline:
        np.save(base, actions)
    if base.exists():
        res["maxabs_vs_baseline"] = float(np.abs(actions - np.load(base)).max())

    if args.mode == "latency":
        for _ in range(5):
            policy.infer(dict(obs), noise=noise)
        e2e = []
        for _ in range(args.iters):
            t = time.perf_counter()
            policy.infer(dict(obs), noise=noise)
            e2e.append((time.perf_counter() - t) * 1e3)
        inputs = policy._input_transform(dict(obs))  # noqa: SLF001
        observation = _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))
        jn, rng = jnp.asarray(noise)[None], jax.random.key(0)
        for _ in range(3):
            np.asarray(policy._sample_actions(rng, observation, noise=jn))  # noqa: SLF001
        model = []
        for _ in range(args.iters):
            t = time.perf_counter()
            np.asarray(policy._sample_actions(rng, observation, noise=jn))  # noqa: SLF001
            model.append((time.perf_counter() - t) * 1e3)
        res.update(infer_p50_ms=float(np.median(e2e)), infer_p90_ms=float(np.percentile(e2e, 90)),
                   model_p50_ms=float(np.median(model)))
    print("RESULT " + json.dumps(res))
    (HERE / "runs" / "opt" / f"{args.label}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
