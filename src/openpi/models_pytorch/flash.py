"""Speculative inference for pi0 / pi05 (PyTorch): Realtime-VLA FLASH, arXiv:2605.13778.

A flash round skips the PaliGemma prefill. A small draft model proposes an action chunk from the current prefix
embeddings; the Action Expert then reconstructs the chunk endpoint from a few flow-matching timesteps in parallel,
reusing the KV cache of the most recent full round, and the longest prefix whose reconstructions agree with the draft
is executed. A full round runs when nothing is accepted, when the gripper is about to switch, or periodically.

The draft head and the acceptance rule are adapted from https://github.com/dexmal/realtime-vla-flash (Apache-2.0).

Time convention (openpi): x_t = t * noise + (1 - t) * actions, v = noise - actions, so t = 1 is noise and the endpoint
reconstructed from x_t is x_t - t * v_theta(x_t, t).
"""

from collections.abc import Sequence
import dataclasses
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.cache_utils import DynamicCache
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

from openpi.models_pytorch import triton_ops
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


@dataclasses.dataclass(frozen=True)
class FlashConfig:
    # Flow-matching timesteps used for verification (openpi convention: small t is close to the actions).
    verify_timesteps: tuple[float, ...] = (0.10, 0.05)
    # Per-step acceptance threshold on the RMS distance over the first `dist_dims` normalized action dims.
    threshold: float = 0.15
    dist_dims: int = 6
    # Gripper channel and the normalized value that separates open from closed.
    gripper_dim: int = 6
    gripper_threshold: float = 0.0
    # At most this many actions are executed per round (the client's replan window).
    max_exec_steps: int = 5
    # Force a full round after this many consecutive flash rounds (0 disables).
    full_every_n_flash_rounds: int = 0


class DraftChunkHead(nn.Module):
    """One Gemma decoder block over [prefix embeddings, state token, H action queries] -> H actions.

    With `confidence=True` it also predicts, per action, the probability that the verifier accepts it (the confidence
    head of DSpark, arXiv:2607.05147: c_k = sigmoid(w^T [h_k; embed(x_{k-1})]), with the previous draft action in
    place of the previous token's embedding). The survival probability of a prefix is the cumulative product.
    """

    def __init__(
        self, text_config, *, chunk_len: int, action_dim: int = 7, state_dim: int = 32, confidence: bool = False
    ):
        super().__init__()
        self.chunk_len = int(chunk_len)
        self.action_dim = int(action_dim)
        hidden = int(text_config.hidden_size)
        self.state_token = nn.Linear(int(state_dim), hidden)
        self.action_queries = nn.Embedding(self.chunk_len, hidden)
        block_config = type(text_config)(**{**text_config.to_dict(), "num_hidden_layers": 1})
        block_config._attn_implementation = "sdpa"  # noqa: SLF001
        self.block = GemmaDecoderLayer(block_config, layer_idx=0)
        self.rotary_emb = GemmaRotaryEmbedding(block_config)
        self.action_head = nn.Linear(hidden, self.action_dim)
        self.confidence_head = nn.Linear(hidden + self.action_dim, 1) if confidence else None
        if self.confidence_head is not None:  # start at p = 0.5 without perturbing the action path
            nn.init.zeros_(self.confidence_head.weight)
            nn.init.zeros_(self.confidence_head.bias)

    def init_from_vlm_layer(self, layer: nn.Module) -> None:
        self.block.load_state_dict(layer.state_dict(), strict=True)

    def forward(self, prefix_embs, prefix_pad_masks, prefix_att_masks, state) -> torch.Tensor:
        hidden = self._action_hidden(prefix_embs, prefix_pad_masks, state)
        return self.action_head(hidden.to(self.action_head.weight.dtype)).float()

    def forward_with_confidence(
        self, prefix_embs, prefix_pad_masks, prefix_att_masks, state
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Actions (B, H, action_dim), identical to `forward`, and per-action acceptance logits (B, H)."""
        if self.confidence_head is None:
            raise ValueError("DraftChunkHead was built without a confidence head")
        hidden = self._action_hidden(prefix_embs, prefix_pad_masks, state)
        actions = self.action_head(hidden.to(self.action_head.weight.dtype)).float()
        # Previous draft action (zeros before the first), detached like the sampled token DSpark conditions on.
        prev = torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1].detach()], dim=1)
        head_dtype = self.confidence_head.weight.dtype
        features = torch.cat([hidden.to(head_dtype), prev.to(head_dtype)], dim=-1)
        return actions, self.confidence_head(features).squeeze(-1).float()

    def _action_hidden(self, prefix_embs, prefix_pad_masks, state) -> torch.Tensor:
        """Final hidden states of the action slots; `forward` is bitwise the same as `forward_reference`.

        The action slots are the last attention block, so they attend to every unpadded token, and only their outputs
        are decoded: prefix and state tokens contribute keys and values but need no query, o_proj, or MLP (~1/57 of the
        tokens). Every remaining op runs on the same values, in the same dtype, as in the full block (slicing rows does
        not change a row's linear, attention, or elementwise result), and RMSNorm / rotary embeddings run as Triton
        kernels that reproduce PyTorch's rounding. Prefix attention masks are not needed: the action slots see every block.
        """
        b, device = prefix_embs.shape[0], prefix_embs.device
        block, attn = self.block, self.block.self_attn
        h = self.chunk_len
        dtype = attn.q_proj.weight.dtype
        state_tok = self.state_token(state.to(self.state_token.weight.dtype))[:, None, :].to(dtype)
        queries = self.action_queries.weight[None].expand(b, -1, -1).to(dtype)
        hidden = torch.cat([prefix_embs.to(dtype), state_tok, queries], dim=1)

        ones = torch.ones((b, 1 + h), dtype=torch.bool, device=device)
        pad = torch.cat([prefix_pad_masks, ones], dim=1)
        position_ids = torch.cumsum(pad, dim=1) - 1
        cos, sin = self.rotary_emb(hidden, position_ids)

        # Self-attention (GemmaAttention + transformers' sdpa_attention_forward), queries for the action slots only.
        normed = triton_ops.gemma_rms_norm(hidden, block.input_layernorm.weight, block.input_layernorm.eps)
        t_len = normed.shape[1]
        q = attn.q_proj(normed[:, -h:]).view(b, h, -1, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(normed).view(b, t_len, -1, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(normed).view(b, t_len, -1, attn.head_dim).transpose(1, 2)
        q = triton_ops.apply_rope(q, cos[:, -h:], sin[:, -h:])
        k = triton_ops.apply_rope(k, cos, sin)
        n_rep = attn.num_key_value_groups
        k = k[:, :, None].expand(b, k.shape[1], n_rep, t_len, attn.head_dim).reshape(b, -1, t_len, attn.head_dim)
        v = v[:, :, None].expand(b, v.shape[1], n_rep, t_len, attn.head_dim).reshape(b, -1, t_len, attn.head_dim)
        # The action rows of the full block's additive mask: every unpadded key.
        mask = torch.zeros((b, 1, h, t_len), dtype=dtype, device=device).masked_fill(~pad[:, None, None, :], -1e9)
        out = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=mask, dropout_p=0.0, scale=attn.scaling
        )
        out = attn.o_proj(out.transpose(1, 2).contiguous().reshape(b, h, -1))
        hidden = hidden[:, -h:] + out

        normed = triton_ops.gemma_rms_norm(
            hidden, block.post_attention_layernorm.weight, block.post_attention_layernorm.eps
        )
        return hidden + block.mlp(normed)

    def forward_reference(self, prefix_embs, prefix_pad_masks, prefix_att_masks, state) -> torch.Tensor:
        """The full decoder block over every token (original implementation, kept for tests)."""
        b, device = prefix_embs.shape[0], prefix_embs.device
        dtype = self.block.self_attn.q_proj.weight.dtype
        state_tok = self.state_token(state.to(self.state_token.weight.dtype))[:, None, :].to(dtype)
        queries = self.action_queries.weight[None].expand(b, -1, -1).to(dtype)
        hidden = torch.cat([prefix_embs.to(dtype), state_tok, queries], dim=1)

        # Blocks: [prefix (bidirectional)] [state] [actions]; later blocks see earlier ones, actions see each other.
        ones = torch.ones((b, 1), dtype=torch.bool, device=device)
        pad = torch.cat([prefix_pad_masks, ones, ones.expand(b, self.chunk_len)], dim=1)
        att = torch.cat(
            [
                prefix_att_masks.to(torch.bool),
                ones,
                torch.cat([ones, torch.zeros((b, self.chunk_len - 1), dtype=torch.bool, device=device)], dim=1),
            ],
            dim=1,
        )
        mask = make_att_2d_masks(pad, att)
        # Padded tokens (e.g. an empty camera slot) would otherwise have fully masked rows, whose softmax is NaN in
        # bf16 and poisons gradients. Letting every token see itself does not change any unpadded token's output.
        mask = mask | torch.eye(mask.shape[-1], dtype=torch.bool, device=device)[None]
        attention_mask = torch.zeros(mask[:, None].shape, dtype=dtype, device=device).masked_fill(~mask[:, None], -1e9)
        position_ids = torch.cumsum(pad, dim=1) - 1
        hidden = self.block(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=self.rotary_emb(hidden, position_ids),
        )[0]
        return self.action_head(hidden[:, -self.chunk_len :].to(self.action_head.weight.dtype)).float()


def expand_cache(past_key_values, repeats: int) -> DynamicCache:
    """A read-only cache with every layer's (k, v) expanded along the batch axis (no copy)."""
    legacy = [(k.expand(repeats, *k.shape[1:]), v.expand(repeats, *v.shape[1:])) for k, v in _layers(past_key_values)]
    return DynamicCache.from_legacy_cache(tuple(legacy))


def _layers(past_key_values) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(past_key_values, DynamicCache):
        return [past_key_values[i] for i in range(len(past_key_values))]
    return past_key_values


def reconstruct_endpoints(model, state, prefix_pad_masks, past_key_values, draft, noise, timesteps) -> torch.Tensor:
    """Action Expert endpoint reconstructions of `draft` at each verification timestep, in one batched pass.

    draft, noise: (1, H, D) in the model's normalized action space. Returns (K, H, D).
    """
    k = timesteps.shape[0]
    t = timesteps[:, None, None]
    x_t = t * noise + (1.0 - t) * draft  # (K, H, D)
    v_t = model.denoise_step(
        state.expand(k, *state.shape[1:]),
        prefix_pad_masks.expand(k, -1),
        expand_cache(past_key_values, k),
        x_t,
        timesteps,
    )
    return x_t - t * v_t


def step_accepted(draft, reference, config: FlashConfig) -> torch.Tensor:
    """Whether each draft action is within `threshold` (RMS over the first `dist_dims`) of the reference: (..., H)."""
    d = config.dist_dims
    dist = torch.linalg.vector_norm(reference[..., :d] - draft[..., :d], dim=-1) / math.sqrt(d)
    return dist <= config.threshold


def accepted_prefix_len(draft, endpoints, config: FlashConfig) -> torch.Tensor:
    """Longest leading run of draft actions within `threshold` of every reconstruction (Algorithm 1)."""
    h = min(draft.shape[-2], config.max_exec_steps)
    ok = step_accepted(draft[:, :h], endpoints[:, :h], config).to(torch.int64).cumprod(dim=-1)  # (K, h)
    return ok.sum(dim=-1).min()


def gripper_switch_in(chunks, prev_gripper, config: FlashConfig, horizon: int) -> torch.Tensor:
    """Whether any chunk switches the gripper within the first `horizon` actions (relative to `prev_gripper`)."""
    g = chunks[..., :horizon, config.gripper_dim]
    prev = torch.cat([prev_gripper.expand(*g.shape[:-1], 1), g[..., :-1]], dim=-1)
    thr = config.gripper_threshold
    return ((prev < thr) != (g < thr)).any()


def full_round(model, observation, noise, num_steps: int = 10):
    """PI0Pytorch.sample_actions, also returning the prefix pad masks and KV cache that flash rounds reuse."""
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    dt = torch.full((), -1.0 / num_steps, dtype=torch.float32, device=noise.device)
    time = torch.full((), 1.0, dtype=torch.float32, device=noise.device)
    x_t = noise
    for _ in range(num_steps):
        v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(noise.shape[0]))
        x_t = x_t + dt * v_t
        time = time + dt
    kv = [(k, v) for k, v in _layers(past_key_values)]
    return x_t, prefix_pad_masks, kv


def flash_round(model, draft: DraftChunkHead, observation, noise, prefix_pad_masks_cached, kv_cached, timesteps):
    """Draft a chunk from the current observation and reconstruct its endpoints against the cached full-round context.

    Returns (draft chunk (1, H, D), endpoints (K, H, D)), both in the normalized action space.
    """
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    proposal = draft(prefix_embs, prefix_pad_masks, prefix_att_masks, state.float())
    chunk = torch.zeros_like(noise)
    chunk[..., : proposal.shape[-1]] = proposal
    endpoints = reconstruct_endpoints(model, state, prefix_pad_masks_cached, kv_cached, chunk, noise, timesteps)
    return chunk, endpoints
