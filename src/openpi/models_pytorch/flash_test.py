import pytest
import torch
from transformers import GemmaConfig

from openpi.models_pytorch import flash
from openpi.models_pytorch import triton_ops

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _config(hidden=256, heads=4, head_dim=64):
    return GemmaConfig(
        hidden_size=hidden,
        intermediate_size=4 * hidden,
        num_attention_heads=heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        hidden_activation="gelu_pytorch_tanh",
        vocab_size=16,
        max_position_embeddings=2048,
    )


def _assert_equal(actual, expected, name=""):
    assert torch.equal(actual, expected), f"{name}: {(actual != expected).sum().item()} elements differ"


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_matches_reference_bitwise(dtype):
    torch.manual_seed(0)
    x = torch.randn(3, 37, 300, device="cuda", dtype=dtype, requires_grad=True)
    w = (0.1 * torch.randn(300, device="cuda")).requires_grad_()
    dy = torch.randn(3, 37, 300, device="cuda", dtype=dtype)
    ref = triton_ops.gemma_rms_norm_reference(x, w, 1e-6)
    ref_dx, ref_dw = torch.autograd.grad(ref, (x, w), dy)
    out = triton_ops.gemma_rms_norm(x, w, 1e-6)
    dx, dw = torch.autograd.grad(out, (x, w), dy)
    _assert_equal(out, ref, "out")
    _assert_equal(dx, ref_dx, "dx")
    _assert_equal(dw, ref_dw, "dw")

    # Frozen input: only the weight gradient.
    (dw_only,) = torch.autograd.grad(triton_ops.gemma_rms_norm(x.detach(), w, 1e-6), (w,), dy)
    _assert_equal(dw_only, ref_dw, "dw (frozen input)")


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_matches_reference_bitwise(dtype):
    torch.manual_seed(0)
    rotary = flash.GemmaRotaryEmbedding(_config(head_dim=64)).cuda()
    x = torch.randn(2, 3, 29, 64, device="cuda", dtype=dtype, requires_grad=True)
    pos = torch.arange(29, device="cuda")[None].expand(2, -1) + torch.tensor([[0], [7]], device="cuda")
    cos, sin = rotary(torch.zeros(1, device="cuda"), pos)
    dy = torch.randn(2, 3, 29, 64, device="cuda")
    ref = triton_ops.apply_rope_reference(x, cos, sin)
    (ref_dx,) = torch.autograd.grad(ref, (x,), dy)
    out = triton_ops.apply_rope(x, cos, sin)
    (dx,) = torch.autograd.grad(out, (x,), dy)
    assert out.dtype == ref.dtype
    _assert_equal(out, ref, "out")
    _assert_equal(dx, ref_dx, "dx")


def _draft_and_inputs():
    torch.manual_seed(0)
    config = _config()
    b, prefix_len, horizon = 4, 50, 10
    draft = flash.DraftChunkHead(config, chunk_len=horizon, action_dim=7).cuda().float()
    with torch.no_grad():  # the zero-initialized norm weights would hide a wrong (1 + w) scale
        for name, p in draft.named_parameters():
            if "layernorm" in name:
                p.normal_(0, 0.1)
    prefix = torch.randn(b, prefix_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    pad = torch.ones(b, prefix_len, dtype=torch.bool, device="cuda")
    pad[1, 40:] = False  # padded prompt tokens
    pad[2, :10] = False  # a masked camera slot
    att = torch.zeros(b, prefix_len, dtype=torch.bool, device="cuda")
    state = torch.randn(b, 32, device="cuda")
    target = torch.randn(b, horizon, 7, device="cuda")
    return draft, (prefix, pad, att, state), target


@cuda
def test_draft_forward_matches_reference():
    draft, inputs, target = _draft_and_inputs()

    def run(fn, *, autocast):
        draft.zero_grad(set_to_none=True)
        prefix, pad, att, state = inputs
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            pred = fn(prefix if autocast else prefix.float(), pad, att, state)
        torch.nn.functional.smooth_l1_loss(pred, target).backward()
        return pred.detach(), {n: p.grad.clone() for n, p in draft.named_parameters() if p.grad is not None}

    # Training precision: the prediction is bitwise identical. Weight gradients sum over a different number of rows
    # (cuBLAS may pick another algorithm), so they agree up to rounding.
    ref_pred, ref_grads = run(draft.forward_reference, autocast=True)
    pred, grads = run(draft.forward, autocast=True)
    _assert_equal(pred, ref_pred, "pred")
    assert grads.keys() == ref_grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grad, ref_grads[name], rtol=2e-2, atol=2e-4, msg=name)

    # float32: the gradients are the same function, up to float32 roundoff.
    ref_pred, ref_grads = run(draft.forward_reference, autocast=False)
    pred, grads = run(draft.forward, autocast=False)
    torch.testing.assert_close(pred, ref_pred, rtol=1e-5, atol=1e-6)
    for name, grad in grads.items():
        torch.testing.assert_close(grad, ref_grads[name], rtol=1e-4, atol=1e-7, msg=name)


@cuda
def test_draft_forward_compiles_to_the_same_result_as_reference():
    # Inductor re-rounds the surrounding ops, so compare compiled against compiled; the kernels must not break the graph.
    draft, inputs, _ = _draft_and_inputs()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        ref = torch.compile(draft.forward_reference)(*inputs)
        out = torch.compile(draft.forward, fullgraph=True)(*inputs)
    _assert_equal(out, ref, "compiled pred")


@cuda
def test_draft_serving_precision_runs():
    # FlashPolicy serves the draft in bfloat16 without autocast (the attention mask must match the query dtype).
    draft, inputs, _ = _draft_and_inputs()
    draft = draft.to(torch.bfloat16).eval()
    with torch.inference_mode():
        ref = draft.forward_reference(*inputs)
        out = draft.forward(*inputs)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.isfinite(ref).all()
