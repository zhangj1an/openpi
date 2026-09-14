"""Speculative (FLASH) serving for PyTorch pi0 / pi05 policies. See openpi.models_pytorch.flash."""

import dataclasses
import logging
import time
from typing import Any

import jax
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi.models import model as _model
from openpi.models_pytorch import flash as _flash
from openpi.policies import policy as _policy


class _Graph:
    """Capture `fn()` (which reads only static tensors) into a CUDA graph; `replay()` returns the static outputs."""

    def __init__(self, fn, device: str):
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                fn()
        torch.cuda.current_stream(device).wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self._graph):
            self.outputs = fn()

    def replay(self):
        self._graph.replay()
        return self.outputs


@dataclasses.dataclass
class _Slot:
    static: dict
    noise: torch.Tensor
    full: _Graph | None = None
    flash: _Graph | None = None
    cache_ready: bool = False


class FlashPolicy(_base_policy.BasePolicy):
    """Serves full rounds and speculative flash rounds from one PyTorch policy and a trained draft head.

    Request extras: `flash_reset` (bool) forces a full round (send it at the start of an episode); `executed_steps`
    (int) is how many actions of the previous response were executed (default: all returned, up to max_exec_steps).
    With `diagnostics=True`, `flash` also carries the verifier's own result for every flash attempt (`draft_accepted`,
    `gripper_switch`, per-step `step_dist`, and the normalized `draft`), and every response the normalized `actions`.

    Response extras: `accepted_prefix_len` and `flash` statistics. A rejected draft or a predicted gripper switch falls
    back to a full round within the same request, so every response carries at least one action.
    """

    def __init__(
        self,
        policy: _policy.Policy,
        draft: _flash.DraftChunkHead,
        config: _flash.FlashConfig,
        *,
        num_steps: int = 10,
        compile_mode: str | None = None,
        diagnostics: bool = False,
    ):
        if not policy._is_pytorch_model:  # noqa: SLF001
            raise ValueError("FlashPolicy requires a PyTorch policy")
        self._policy = policy
        self._model = policy._model  # noqa: SLF001
        self._device = policy._pytorch_device  # noqa: SLF001
        self._draft = draft.to(self._device, torch.bfloat16).eval()
        self._config = config
        self._num_steps = num_steps
        self._diagnostics = diagnostics
        self._timesteps = torch.tensor(config.verify_timesteps, dtype=torch.float32, device=self._device)
        cfg = self._model.config
        self._noise_shape = (1, cfg.action_horizon, cfg.action_dim)
        self._slots: dict[Any, _Slot] = {}
        self._full_fn = _flash.full_round
        self._flash_fn = _flash.flash_round
        if compile_mode is not None:
            self._full_fn = torch.compile(_flash.full_round, mode=compile_mode)
            self._flash_fn = torch.compile(_flash.flash_round, mode=compile_mode)
        self._prev_chunk: np.ndarray | None = None
        self._last_gripper: float | None = None
        self._flash_since_full = 0
        self.stats = {"full": 0, "flash": 0, "fallback_reject": 0, "fallback_gripper": 0, "accepted": 0}

    def _slot_for(self, inputs: dict) -> _Slot:
        leaves = jax.tree.leaves(inputs)
        key = (jax.tree.structure(inputs), tuple((np.shape(x), np.asarray(x).dtype.str) for x in leaves))
        slot = self._slots.get(key)
        if slot is None:
            static = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._device)[None, ...], inputs)
            noise = torch.zeros(self._noise_shape, dtype=torch.float32, device=self._device)
            slot = self._slots[key] = _Slot(static=static, noise=noise)
        return slot

    def _observation(self, slot: _Slot):
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in slot.static.items()}
        return _model.Observation.from_dict(data)

    def _full(self, slot: _Slot) -> torch.Tensor:
        if slot.full is None:
            slot.full = _Graph(
                lambda: self._full_fn(self._model, self._observation(slot), slot.noise, self._num_steps), self._device
            )
        actions, _, _ = slot.full.replay()
        slot.cache_ready = True
        self._flash_since_full = 0
        self.stats["full"] += 1
        return actions[0]

    def _flash(self, slot: _Slot) -> tuple[torch.Tensor, torch.Tensor]:
        if slot.flash is None:
            _, pad, kv = slot.full.outputs  # static tensors refreshed by every full-round replay
            slot.flash = _Graph(
                lambda: self._flash_fn(
                    self._model, self._draft, self._observation(slot), slot.noise, pad, kv, self._timesteps
                ),
                self._device,
            )
        return slot.flash.replay()

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        obs = dict(obs)
        reset = bool(obs.pop("flash_reset", False))
        executed = obs.pop("executed_steps", None)
        inputs = self._policy._input_transform(jax.tree.map(lambda x: x, obs))  # noqa: SLF001
        inputs = _policy.drop_masked_image_slots(inputs)
        start = time.monotonic()
        slot = self._slot_for(inputs)
        for dst, src in zip(jax.tree.leaves(slot.static), jax.tree.leaves(inputs), strict=True):
            dst.copy_(torch.from_numpy(np.array(src)).view(dst.shape))
        slot.noise.normal_()

        cfg = self._config
        if reset:
            self._prev_chunk, self._last_gripper = None, None
        elif self._prev_chunk is not None:
            n = len(self._prev_chunk) if executed is None else int(executed)
            n = max(1, min(n, len(self._prev_chunk), cfg.max_exec_steps))
            self._last_gripper = float(self._prev_chunk[n - 1, cfg.gripper_dim])

        periodic_full = cfg.full_every_n_flash_rounds > 0 and self._flash_since_full >= cfg.full_every_n_flash_rounds
        use_flash = not reset and slot.cache_ready and self._last_gripper is not None and not periodic_full
        accepted, kind = 0, "full"
        verify: dict[str, Any] = {}
        if use_flash:
            chunk, endpoints = self._flash(slot)
            prev = torch.tensor(self._last_gripper, device=self._device)
            candidates = torch.cat([chunk, endpoints], dim=0)
            accepted_t = _flash.accepted_prefix_len(chunk, endpoints, cfg)
            switch_t = _flash.gripper_switch_in(candidates, prev, cfg, cfg.max_exec_steps)
            accepted, switch = int(accepted_t), bool(switch_t)
            if self._diagnostics:  # the verifier's own result, also for rounds that fall back
                h = min(chunk.shape[-2], cfg.max_exec_steps)
                dist = _flash.step_distance(chunk[:, :h], endpoints[:, :h], cfg).amax(dim=0)  # worst timestep per step
                verify = {
                    "draft_accepted": accepted,
                    "gripper_switch": switch,
                    "step_dist": dist.float().cpu().tolist(),
                    "draft": chunk[0, :h, : cfg.gripper_dim + 1].float().cpu().tolist(),
                }
            if accepted > 0 and not switch:
                actions = chunk[0, :accepted]
                kind = "flash"
                self._flash_since_full += 1
                self.stats["flash"] += 1
                self.stats["accepted"] += accepted
            else:
                self.stats["fallback_gripper" if switch else "fallback_reject"] += 1
                kind = "fallback_gripper" if switch else "fallback_reject"
        if kind != "flash":
            actions = self._full(slot)
            accepted = actions.shape[0]

        actions_np = actions.float().cpu().numpy()
        self._prev_chunk = actions_np
        outputs = self._policy._output_transform(  # noqa: SLF001
            {"state": np.asarray(inputs["state"]), "actions": actions_np}
        )
        outputs["accepted_prefix_len"] = int(accepted)
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - start) * 1000}
        outputs["flash"] = {"round": kind, **verify}
        if self._diagnostics:
            outputs["flash"]["actions"] = actions_np[:, : cfg.gripper_dim + 1].tolist()
        if sum(self.stats.values()) % 200 == 0:
            logging.info("flash stats %s", self.stats)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata
