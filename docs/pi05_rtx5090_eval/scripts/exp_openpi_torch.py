#!/usr/bin/env python
"""openpi PyTorch pi05_libero on RTX 5090: latency (Policy.infer + model-only) and parity vs openpi JAX.

  exp_openpi_torch.py --compile none|default|reduce-overhead|max-autotune --quant none|fp8|nvfp4 --label x
"""

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent

import jax  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402


def quantize_lm(model, quant):
    if quant == "none":
        return 0
    from torchao.quantization import quantize_

    if quant == "fp8":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    elif quant == "nvfp4":
        from torchao.prototype.mx_formats import NVFP4InferenceConfig

        cfg = NVFP4InferenceConfig()
    else:
        raise ValueError(quant)
    lm = model.paligemma_with_expert.paligemma.language_model
    n = sum(isinstance(m, torch.nn.Linear) for m in lm.modules())
    quantize_(lm, cfg, filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.bfloat16)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/dev/shm/pi05_libero_openpi_pt")
    p.add_argument("--compile", default="none")
    p.add_argument("--quant", default="none", choices=["none", "fp8", "nvfp4"])
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--token-len-buckets", default="")
    p.add_argument("--label", required=True)
    args = p.parse_args()

    cfg = _config.get_config("pi05_libero")
    mode = None if args.compile == "none" else args.compile
    # compile is applied after quantization below, so load uncompiled
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    t = time.perf_counter()
    buckets = tuple(int(x) for x in args.token_len_buckets.split(",") if x)
    policy = policy_config.create_trained_policy(cfg, args.ckpt, token_len_buckets=buckets or None)
    load_s = time.perf_counter() - t
    model = policy._model  # noqa: SLF001
    policy._use_cuda_graph = not args.no_cuda_graph  # noqa: SLF001
    n_q = quantize_lm(model, args.quant)
    if mode is not None:
        model.sample_actions = torch.compile(model.sample_actions, mode=mode)
        policy._sample_actions = model.sample_actions  # noqa: SLF001

    d = np.load(HERE / "libero_obs_sample.npz")
    obs = {"observation/image": d["image"], "observation/wrist_image": d["wrist_image"],
           "observation/state": d["state"], "prompt": str(d["prompt"])}
    ref = np.load(HERE / "runs/parity_openpi_ref.npz")
    noise = ref["noise"]

    with torch.inference_mode():
        t = time.perf_counter()
        actions = policy.infer(dict(obs), noise=noise)["actions"]
        first_ms = (time.perf_counter() - t) * 1e3
        for _ in range(4):
            policy.infer(dict(obs), noise=noise)
        e2e = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t = time.perf_counter()
            actions = policy.infer(dict(obs), noise=noise)["actions"]
            e2e.append((time.perf_counter() - t) * 1e3)

        inputs = policy._input_transform(dict(obs))  # noqa: SLF001
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to("cuda")[None, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        tn = torch.from_numpy(noise)[None].cuda()
        m = []
        for _ in range(0 if policy._use_cuda_graph else args.iters):  # noqa: SLF001
            torch.cuda.synchronize()
            t = time.perf_counter()
            policy._sample_actions("cuda", observation, noise=tn)  # noqa: SLF001
            torch.cuda.synchronize()
            m.append((time.perf_counter() - t) * 1e3)

    res = {"label": args.label, "framework": "openpi-pytorch", "torch": torch.__version__, "compile": args.compile,
           "quant": args.quant, "quantized_linears": n_q, "policy_load_s": load_s, "first_call_ms": first_ms,
           "infer_p50_ms": float(np.median(e2e)), "infer_p90_ms": float(np.percentile(e2e, 90)),
           "model_p50_ms": float(np.median(m)) if m else None, "cuda_graph": not args.no_cuda_graph, "token_len_buckets": list(buckets), "maxabs_vs_jax_ref": float(np.abs(actions - ref["actions"]).max()),
           "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
    print("RESULT " + json.dumps(res))
    (HERE / "runs" / "opt" / f"{args.label}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
