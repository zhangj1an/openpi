"""Serve a PyTorch pi0 / pi05 policy with FLASH speculative inference over the openpi websocket protocol.

    uv run scripts/flash/serve_flash_policy.py --config pi05_libero --checkpoint-dir /ckpt/pi05_libero_pytorch \
        --draft-dir draft_libero_spatial --pytorch-quantization nvfp4

Clients should send `flash_reset: True` with the first observation of every episode and execute at most
`--max-exec-steps` actions (or `accepted_prefix_len`) per response.
"""

import dataclasses
import json
import logging
import os
import pathlib
import socket

import safetensors.torch
import tyro

from openpi.models_pytorch import flash as _flash
from openpi.policies import flash_policy as _flash_policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    draft_dir: str
    config: str = "pi05_libero"
    port: int = 8000
    pytorch_quantization: str | None = None
    token_len_buckets: tuple[int, ...] = (48,)
    compile_mode: str | None = "max-autotune-no-cudagraphs"
    verify_timesteps: tuple[float, ...] = (0.10, 0.05)
    threshold: float = 0.15
    max_exec_steps: int = 5
    full_every_n_flash_rounds: int = 0


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    # FlashPolicy compiles its own full/flash rounds; the policy's compiled sample_actions is not used.
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
    draft_meta = json.loads((pathlib.Path(args.draft_dir) / "draft_meta.json").read_text())
    draft = _flash.DraftChunkHead(
        model.paligemma_with_expert.paligemma.language_model.config,
        chunk_len=train_config.model.action_horizon,
        action_dim=7,
        confidence=draft_meta.get("confidence", False),
    )
    safetensors.torch.load_model(draft, str(pathlib.Path(args.draft_dir) / "draft.safetensors"))
    flash_config = _flash.FlashConfig(
        verify_timesteps=args.verify_timesteps,
        threshold=args.threshold,
        max_exec_steps=args.max_exec_steps,
        full_every_n_flash_rounds=args.full_every_n_flash_rounds,
    )
    flash = _flash_policy.FlashPolicy(policy, draft, flash_config, compile_mode=args.compile_mode)
    logging.info("Serving FLASH policy on %s:%d (%s)", socket.gethostname(), args.port, flash_config)
    websocket_policy_server.WebsocketPolicyServer(
        policy=flash, host="0.0.0.0", port=args.port, metadata=flash.metadata
    ).serve_forever()


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
