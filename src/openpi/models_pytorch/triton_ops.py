# ruff: noqa: N803  (Triton kernels conventionally use uppercase pointer and constexpr names)
"""Triton kernels for the FLASH draft head: Gemma RMSNorm and rotary embedding, forward and backward.

The kernels reproduce the rounding of the PyTorch reference implementations below (and of autograd through them):
every elementwise step happens in the same order and dtype, casts happen where autograd casts, multiply-add fusion is
disabled, and reductions (whose summation order a kernel cannot match) stay in PyTorch on the same shapes. Forward and
backward are therefore bitwise identical to the reference, not merely close.

The ops are registered with `torch.library.triton_op`, so they work under autograd, CUDA graphs, and `torch.compile`.
Without CUDA, or for dtype combinations the kernels do not cover, the reference implementations are used.
"""

import torch
from torch.library import triton_op
from torch.library import wrap_triton
import triton
import triton.language as tl

_NO_FUSION = {"enable_fp_fusion": False}  # a fused multiply-add rounds differently from PyTorch's separate ops

# ----------------------------------------------------------------------------------------------------------------------
# Gemma RMSNorm (transformers_replace GemmaRMSNorm, non-adaptive): computed in float32, returned in the input dtype.
# ----------------------------------------------------------------------------------------------------------------------


def gemma_rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps) * (1.0 + weight.float())).to(x.dtype)


@triton.jit
def _rms_norm_fwd_kernel(X, INV, SCALE, Y, n_rows, n_cols, BLOCK: tl.constexpr):
    idx = tl.program_id(0).cast(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_rows * n_cols
    x = tl.load(X + idx, mask=mask).to(tl.float32)
    inv = tl.load(INV + idx // n_cols, mask=mask)
    scale = tl.load(SCALE + idx % n_cols, mask=mask)
    tl.store(Y + idx, (x * inv) * scale, mask=mask)


@triton.jit
def _rms_norm_bwd_a_kernel(G, X, INV, SCALE, GRAD_N, PROD_X, PROD_N, n_rows, n_cols, BLOCK: tl.constexpr):
    # grad_n = g * scale (mul backward of y = n * scale); grad_scale terms g * n; grad_inv terms grad_n * x.
    idx = tl.program_id(0).cast(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_rows * n_cols
    g = tl.load(G + idx, mask=mask)
    x = tl.load(X + idx, mask=mask).to(tl.float32)
    inv = tl.load(INV + idx // n_cols, mask=mask)
    scale = tl.load(SCALE + idx % n_cols, mask=mask)
    grad_n = g * scale
    tl.store(GRAD_N + idx, grad_n, mask=mask)
    tl.store(PROD_X + idx, grad_n * x, mask=mask)
    tl.store(PROD_N + idx, g * (x * inv), mask=mask)


@triton.jit
def _rms_norm_bwd_b_kernel(GRAD_N, X, INV, GRAD_SQ, DX_DIRECT, DX_SQ, n_rows, n_cols, BLOCK: tl.constexpr):
    # Direct path grad_n * inv (mul backward of n = x * inv); square path grad_sq * (2 * x) (pow backward).
    idx = tl.program_id(0).cast(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_rows * n_cols
    x = tl.load(X + idx, mask=mask).to(tl.float32)
    row = idx // n_cols
    tl.store(DX_DIRECT + idx, tl.load(GRAD_N + idx, mask=mask) * tl.load(INV + row, mask=mask), mask=mask)
    tl.store(DX_SQ + idx, tl.load(GRAD_SQ + row, mask=mask) * (2.0 * x), mask=mask)


_BLOCK = 4096


def _launch(kernel, n: int, *args):
    wrap_triton(kernel)[(triton.cdiv(n, _BLOCK),)](*args, BLOCK=_BLOCK, **_NO_FUSION)


@triton_op("openpi::gemma_rms_norm_fwd", mutates_args={})
def _rms_norm_fwd(x: torch.Tensor, scale: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    n_cols = x.shape[-1]
    y = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _launch(_rms_norm_fwd_kernel, x.numel(), x, inv, scale, y, x.numel() // n_cols, n_cols)
    return y


@triton_op("openpi::gemma_rms_norm_bwd", mutates_args={})
def _rms_norm_bwd(
    g: torch.Tensor, x: torch.Tensor, scale: torch.Tensor, inv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    grad_n, prod_x, prod_n = (torch.empty(x.shape, dtype=torch.float32, device=x.device) for _ in range(3))
    _launch(_rms_norm_bwd_a_kernel, x.numel(), g, x, inv, scale, grad_n, prod_x, prod_n, n_rows, n_cols)
    grad_scale = prod_n.sum(dim=tuple(range(x.dim() - 1)))
    grad_inv = prod_x.sum(dim=-1, keepdim=True)
    grad_var = -0.5 * grad_inv * inv.pow(3)  # rsqrt backward
    grad_sq = grad_var / n_cols  # mean backward (per row; expanded inside the kernel)
    dx_direct, dx_sq = (torch.empty(x.shape, dtype=torch.float32, device=x.device) for _ in range(2))
    _launch(_rms_norm_bwd_b_kernel, x.numel(), grad_n, x, inv, grad_sq.contiguous(), dx_direct, dx_sq, n_rows, n_cols)
    return dx_direct, dx_sq, grad_scale


class _GemmaRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        xf = x.float()
        inv = torch.rsqrt(torch.mean(torch.square(xf), dim=-1, keepdim=True) + eps)
        scale = 1.0 + weight.float()
        x = x.contiguous()
        ctx.save_for_backward(x, scale, inv)
        ctx.weight_dtype = weight.dtype
        return _rms_norm_fwd(x, scale, inv).to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        x, scale, inv = ctx.saved_tensors
        dx_direct, dx_sq, grad_scale = _rms_norm_bwd(dy.float().contiguous(), x, scale, inv)
        # Autograd casts each path's gradient to the input dtype before accumulating them.
        dx = dx_direct.to(x.dtype) + dx_sq.to(x.dtype) if ctx.needs_input_grad[0] else None
        return dx, grad_scale.to(ctx.weight_dtype), None


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not x.is_cuda:
        return gemma_rms_norm_reference(x, weight, eps)
    return _GemmaRMSNorm.apply(x, weight, eps)


# ----------------------------------------------------------------------------------------------------------------------
# Rotary embedding (HF apply_rotary_pos_emb for one tensor): x * cos + rotate_half(x) * sin.
# ----------------------------------------------------------------------------------------------------------------------


def apply_rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, heads, T, D); cos/sin: (B, T, D)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return (x * cos[:, None]) + (torch.cat((-x2, x1), dim=-1) * sin[:, None])


@triton.jit
def _rope_kernel(X, COS, SIN, OUT_A, OUT_B, n_rows, n_heads, t_len, d, BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    # Forward: y = x * cos + rotate_half(x) * sin, written to OUT_A.
    # Backward (X = grad): the mul-backward terms g * cos and rotate_half^T(g * sin), kept separate (OUT_A, OUT_B)
    # because autograd casts each to the input dtype before summing them.
    idx = tl.program_id(0).cast(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_rows * n_heads * t_len * d
    half = d // 2
    col = idx % d
    t = (idx // d) % t_len
    b = idx // (d * t_len * n_heads)
    c = tl.load(COS + (b * t_len + t) * d + col, mask=mask)
    s = tl.load(SIN + (b * t_len + t) * d + col, mask=mask)
    first = col < half
    partner = tl.where(first, idx + half, idx - half)
    x = tl.load(X + idx, mask=mask).to(tl.float32)
    xp = tl.load(X + partner, mask=mask).to(tl.float32)
    if BACKWARD:
        sp = tl.load(SIN + (b * t_len + t) * d + tl.where(first, col + half, col - half), mask=mask)
        tl.store(OUT_A + idx, x * c, mask=mask)
        # rotate_half(u) = cat(-u2, u1)  =>  rotate_half^T(v) = cat(v2, -v1), with v = g * sin.
        tl.store(OUT_B + idx, tl.where(first, xp * sp, -(xp * sp)), mask=mask)
    else:
        tl.store(OUT_A + idx, (x * c) + (tl.where(first, -xp, xp) * s), mask=mask)


@triton_op("openpi::rope_fwd", mutates_args={})
def _rope_fwd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    b, heads, t, d = x.shape
    y = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    wrap_triton(_rope_kernel)[(triton.cdiv(x.numel(), _BLOCK),)](
        x, cos, sin, y, y, b, heads, t, d, BACKWARD=False, BLOCK=_BLOCK, **_NO_FUSION
    )
    return y


@triton_op("openpi::rope_bwd", mutates_args={})
def _rope_bwd(g: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, heads, t, d = g.shape
    grad_cos_term = torch.empty(g.shape, dtype=torch.float32, device=g.device)
    grad_sin_term = torch.empty(g.shape, dtype=torch.float32, device=g.device)
    wrap_triton(_rope_kernel)[(triton.cdiv(g.numel(), _BLOCK),)](
        g, cos, sin, grad_cos_term, grad_sin_term, b, heads, t, d, BACKWARD=True, BLOCK=_BLOCK, **_NO_FUSION
    )
    return grad_cos_term, grad_sin_term


class _Rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        ctx.x_dtype = x.dtype
        return _rope_fwd(x.contiguous(), cos, sin)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        grad_cos_term, grad_sin_term = _rope_bwd(dy.float().contiguous(), cos, sin)
        return grad_cos_term.to(ctx.x_dtype) + grad_sin_term.to(ctx.x_dtype), None, None


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, heads, T, D) in float32 or bfloat16; cos/sin: (B, T, D). Same dtype rules as the reference."""
    if not x.is_cuda or cos.dtype != torch.float32 or sin.dtype != torch.float32 or cos.requires_grad:
        return apply_rope_reference(x, cos, sin)  # e.g. bfloat16 cos/sin, where the reference computes in bfloat16
    return _Rope.apply(x, cos.contiguous(), sin.contiguous())
