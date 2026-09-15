"""Prototype BS=1 speedups for openpi Pi0 (pi05), applied as monkeypatches before the policy is created.

scan_loop       denoise loop as jax.lax.scan with a static trip count instead of jax.lax.while_loop
                (same carry arithmetic, so the result is identical; avoids a per-step predicate on GPU)
batched_siglip  all camera slots through SigLIP in one call (B*views) instead of one call per slot
"""

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0 as _pi0


def _embed_prefix_batched(self, obs):
    names = list(obs.images)
    b = obs.images[names[0]].shape[0]
    pixels = jnp.concatenate([obs.images[n] for n in names], axis=0)
    img_tokens, _ = self.PaliGemma.img(pixels, train=False)  # (views*b, 256, D)
    s = img_tokens.shape[1]
    img_tokens = einops.rearrange(img_tokens, "(v b) s d -> b (v s) d", v=len(names), b=b)
    tokens = [img_tokens]
    input_mask = [einops.repeat(jnp.stack([obs.image_masks[n] for n in names], axis=1), "b v -> b (v s)", s=s)]
    ar_mask = [False] * (len(names) * s)
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]
    return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1), jnp.array(ar_mask)


def _sample_actions_scan(self, rng, observation, *, num_steps=10, noise=None):
    if not isinstance(num_steps, int):
        return _ORIG_SAMPLE(self, rng, observation, num_steps=num_steps, noise=noise)
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry, _):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=pos, kv_cache=kv_cache, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return (x_t + dt * v_t, time + dt), None

    (x_0, _), _ = jax.lax.scan(step, (noise, jnp.asarray(1.0)), None, length=num_steps)
    return x_0


_ORIG_SAMPLE = _pi0.Pi0.sample_actions


def apply(names):
    if "scan_loop" in names:
        _pi0.Pi0.sample_actions = _sample_actions_scan
    if "batched_siglip" in names:
        _pi0.Pi0.embed_prefix = _embed_prefix_batched


# ---------------------------------------------------------------------------------------------------------------
# cudnn_attention: fused cuDNN SDPA (bf16) instead of the fp32 einsum attention in gemma.Attention.
# JAX 0.5.3 only allows head_dim 256 for cuDNN flash attention on Hopper; relax that check for Blackwell.
# ---------------------------------------------------------------------------------------------------------------
def _apply_cudnn_attention():
    import flax.linen as nn

    from jax._src.cudnn import fused_attention_stablehlo as fas
    from openpi.models import gemma as _gemma
    from openpi.models import lora

    _orig_cap = fas.is_cuda_compute_capability_equal
    fas.is_cuda_compute_capability_equal = lambda cap: True if cap == "9.0" else _orig_cap(cap)

    class CudnnAttention(_gemma.Attention):
      @nn.compact
      def __call__(self, xs, positions, attn_mask, kv_cache):
          dtype = next(x.dtype for x in xs if x is not None)
          qkvs = []
          for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
              if x is None:
                  continue
              q_einsum = lora.Einsum(
                  shape=(config.num_heads, config.width, config.head_dim),
                  name=_gemma._name("q_einsum", i),  # noqa: SLF001
                  init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                  lora_config=config.lora_configs.get("attn"),
              )
              q = q_einsum("BTD,NDH->BTNH", x)
              kv_einsum = lora.Einsum(
                  shape=(2, config.num_kv_heads, config.width, config.head_dim),
                  name=_gemma._name("kv_einsum", i),  # noqa: SLF001
                  init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                  lora_config=config.lora_configs.get("attn"),
              )
              k, v = kv_einsum("BSD,2KDH->2BSKH", x)
              qkvs.append((q, k, v))
          q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
          q = _gemma._apply_rope(q, positions=positions)  # noqa: SLF001
          q *= self.configs[0].head_dim ** -0.5
          k = _gemma._apply_rope(k, positions=positions)  # noqa: SLF001
          if kv_cache is not None:
              cache_k, cache_v = kv_cache
              k = jnp.concatenate([cache_k, k], axis=1)
              v = jnp.concatenate([cache_v, v], axis=1)
          encoded = jax.nn.dot_product_attention(
              q, k, v, mask=attn_mask, scale=1.0, implementation="cudnn"
          ).astype(dtype)
          out = []
          start = 0
          for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
              if x is not None:
                  end = start + x.shape[1]
                  out_einsum = lora.Einsum(
                      shape=(config.num_heads, config.head_dim, config.width),
                      name=_gemma._name("attn_vec_einsum", i),  # noqa: SLF001
                      init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                      lora_config=config.lora_configs.get("attn"),
                  )
                  out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                  start = end
              else:
                  out.append(None)
          return out, (k, v)

    _gemma.Attention = CudnnAttention


_orig_apply = apply


def apply(names):  # noqa: F811
    _orig_apply(names)
    if "cudnn_attention" in names:
        _apply_cudnn_attention()
