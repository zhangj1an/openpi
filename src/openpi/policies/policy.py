from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class _TorchChunkGraph:
    """A whole PyTorch inference as one CUDA graph: image conversion, preprocessing, the prefix pass and every
    denoising step. Inputs are copied into static buffers and the graph is replayed, so a call costs one launch
    instead of thousands. Replay is bit-identical to running the same code eagerly.
    """

    def __init__(self, sample_actions, device: str, inputs: dict, noise_shape: tuple[int, ...], sample_kwargs: dict):
        self._sample_actions = sample_actions
        self._device = device
        self._sample_kwargs = sample_kwargs
        self._static = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        self._noise = torch.zeros(noise_shape, dtype=torch.float32, device=device)

        # Run on a side stream first so lazy initialisation (cuBLAS handles, autotuning) happens outside the capture.
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                self._run()
        torch.cuda.current_stream(device).wait_stream(stream)

        self._graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self._graph):
            self._actions = self._run()

    def _run(self) -> torch.Tensor:
        # Observation.from_dict replaces the images in the dict it is given; hand it fresh containers every time.
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in self._static.items()}
        observation = _model.Observation.from_dict(data)
        return self._sample_actions(self._device, observation, noise=self._noise, **self._sample_kwargs)

    def __call__(self, inputs: dict, noise: np.ndarray | None) -> np.ndarray:
        for dst, src in zip(jax.tree.leaves(self._static), jax.tree.leaves(inputs), strict=True):
            dst.copy_(torch.from_numpy(np.asarray(src)).view(dst.shape))
        if noise is None:
            self._noise.normal_()
        else:
            self._noise.copy_(torch.from_numpy(noise).view(self._noise.shape))
        self._graph.replay()
        return self._actions[0].cpu().numpy()


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        warmup_token_lens: Sequence[int] | None = None,
        pytorch_cuda_graph: bool = True,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            warmup_token_lens: Prompt lengths the tokenizer can pad to. On the first request every one of them is
                compiled, so a later prompt of a different length does not stall a running episode.
            pytorch_cuda_graph: For PyTorch models on CUDA, capture each input shape's whole inference into a CUDA
                graph and replay it. Use a compile mode without its own CUDA graphs (e.g. "max-autotune-no-cudagraphs").
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._warmup_token_lens = list(warmup_token_lens or [])

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._use_cuda_graph = pytorch_cuda_graph and str(pytorch_device).startswith("cuda")
            self._torch_graphs: dict = {}
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            self._jax_infer = self._make_jax_infer(model)

    def _make_jax_infer(self, model: _model.BaseModel):
        """One jitted call from host numpy inputs to the unbatched action chunk.

        Batching, uint8 -> float image conversion and the RNG split all run inside the compiled program instead of
        as separate eager device ops (~7 ms of host overhead per call at batch size 1). Same arithmetic and the
        same RNG stream as the eager path.
        """
        graphdef, state = nnx.split(model)
        sample_kwargs = dict(self._sample_kwargs)

        def infer_core(state, rng, inputs, noise):
            module = nnx.merge(graphdef, state)
            inputs = jax.tree.map(lambda x: x[None, ...], inputs)
            rng, sample_rng = jax.random.split(rng)
            kwargs = dict(sample_kwargs)
            if noise is not None:
                kwargs["noise"] = noise[None, ...] if noise.ndim == 2 else noise
            actions = module.sample_actions(sample_rng, _model.Observation.from_dict(inputs), **kwargs)
            return rng, actions[0]

        jitted = jax.jit(infer_core)
        return lambda rng, inputs, noise: jitted(state, rng, inputs, noise)

    def _warmup_prompt_lengths(self, inputs: dict, noise: np.ndarray | None) -> None:
        """Compile every prompt-length shape using this request's structure, with a throwaway RNG key."""
        lens, self._warmup_token_lens = self._warmup_token_lens, []
        current = np.shape(inputs["tokenized_prompt"])[-1]
        for n in lens:
            if n == current:
                continue
            dummy = {**inputs, "tokenized_prompt": np.zeros(n, np.int32), "tokenized_prompt_mask": np.zeros(n, bool)}
            dummy["tokenized_prompt_mask"][0] = True
            start = time.monotonic()
            self._jax_infer(jax.random.key(0), dummy, noise)
            logging.info("Compiled prompt length %d in %.1f s", n, time.monotonic() - start)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        start_time = time.monotonic()
        if not self._is_pytorch_model:
            if noise is not None:
                noise = np.asarray(noise, dtype=np.float32)
            if self._warmup_token_lens:
                self._warmup_prompt_lengths(inputs, noise)
            self._rng, actions = self._jax_infer(self._rng, inputs, noise)
            outputs = {"state": np.asarray(inputs["state"], dtype=np.float32), "actions": np.asarray(actions)}
        elif self._use_cuda_graph:
            if noise is not None:
                noise = np.asarray(noise, dtype=np.float32)
            key = (
                jax.tree.structure(inputs),
                tuple((np.shape(x), np.asarray(x).dtype.str) for x in jax.tree.leaves(inputs)),
            )
            graph = self._torch_graphs.get(key)
            if graph is None:
                cfg = self._model.config
                noise_shape = (1, cfg.action_horizon, cfg.action_dim)
                graph = self._torch_graphs[key] = _TorchChunkGraph(
                    self._sample_actions, self._pytorch_device, inputs, noise_shape, dict(self._sample_kwargs)
                )
            outputs = {"state": np.asarray(inputs["state"]), "actions": graph(inputs, noise)}
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_kwargs = dict(self._sample_kwargs)
            if noise is not None:
                noise = torch.from_numpy(noise).to(self._pytorch_device)
                if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                    noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
                sample_kwargs["noise"] = noise
            observation = _model.Observation.from_dict(inputs)
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(self._pytorch_device, observation, **sample_kwargs),
            }
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
