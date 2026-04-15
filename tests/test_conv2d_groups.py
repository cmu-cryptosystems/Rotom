"""Unit tests for grouped :func:`TensorTerm.conv2d` (matches PyTorch ``groups`` semantics)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from frontends.tensor import TensorTerm
from frontends.tensor_args import Conv2dArgs
from ir.analysis.shape import Shape


def _eval_term(term: TensorTerm, inputs: dict) -> np.ndarray:
    return np.asarray(term.eval(inputs), dtype=np.float64)


def _torch_grouped(
    x_chw: np.ndarray,
    w_oikk: np.ndarray,
    *,
    stride: int,
    padding_mode: str,
    groups: int,
) -> np.ndarray:
    """``x_chw``: ``[C_in, H, W]``; ``w_oikk``: ``[C_out, C_in/groups, kH, kW]``."""
    with torch.no_grad():
        x = torch.from_numpy(x_chw.astype(np.float64)).unsqueeze(0)
        w = torch.from_numpy(w_oikk.astype(np.float64))
        if padding_mode == "same":
            y = F.conv2d(x, w, stride=stride, padding="same", groups=groups)
        else:
            y = F.conv2d(x, w, stride=stride, padding=0, groups=groups)
        return y.squeeze(0).cpu().numpy()


@pytest.mark.parametrize("padding", ("same", "valid"))
def test_conv2d_depthwise_matches_torch(padding: str) -> None:
    """Depthwise (``groups == C_in``) 3×3, stride 1, matches ``torch.nn.functional.conv2d``."""
    rng = np.random.default_rng(0)
    c = 8
    h = w = 12 if padding == "same" else 14
    x = rng.standard_normal((c, h, w))
    wgt = rng.standard_normal((c, 1, 3, 3))

    inp = TensorTerm.Tensor("x", [c, h, w], True)
    wt = TensorTerm.Tensor("w", list(wgt.shape), False)
    term = TensorTerm.conv2d(inp, wt, 1, padding, groups=c)
    got = _eval_term(term, {"x": x, "w": wgt})
    ref = _torch_grouped(x, wgt, stride=1, padding_mode=padding, groups=c)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-9)


def test_conv2d_groups_two_stride_two_valid() -> None:
    """``groups=2``, stride 2, ``padding='valid'`` (avoids PyTorch ``same`` + stride>1 quirks)."""
    rng = np.random.default_rng(1)
    cin, cout, g = 4, 6, 2
    h = w = 17
    x = rng.standard_normal((cin, h, w))
    wgt = rng.standard_normal((cout, cin // g, 3, 3))

    inp = TensorTerm.Tensor("x", [cin, h, w], True)
    wt = TensorTerm.Tensor("w", list(wgt.shape), False)
    term = TensorTerm.conv2d(inp, wt, 2, "valid", groups=g)
    got = _eval_term(term, {"x": x, "w": wgt})
    ref = _torch_grouped(x, wgt, stride=2, padding_mode="valid", groups=g)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-9)


def test_shape_analyzer_grouped_conv2d() -> None:
    """:class:`ir.analysis.shape.Shape` reports the same logical output shape as PyTorch.

    Channel counts are powers of two so ``get_padded_shape`` agrees with divisibility checks.
    """
    cin, cout, g, h, w, k = 8, 8, 2, 8, 8, 3
    inp = TensorTerm.Tensor("x", [cin, h, w], True)
    wt = TensorTerm.Tensor("w", [cout, cin // g, k, k], False)
    term = TensorTerm.conv2d(inp, wt, 1, "same", groups=g)
    sa = Shape(term)
    sa.run()
    logical = sa.get_shape(term)
    assert logical[0] == cout
    assert logical[1] == h and logical[2] == w


def test_conv2d_args_wire_format_groups_then_padding() -> None:
    """After layout gen, padding sits at index 5 when ``groups`` is stored at index 4."""
    inp = TensorTerm.Tensor("x", [4, 8, 8], True)
    wt = TensorTerm.Tensor("w", [6, 2, 3, 3], False)
    term = TensorTerm.conv2d(inp, wt, 1, "same", groups=2)
    term.cs.append([0, 0, 0, 0])
    assert Conv2dArgs.from_term(term).groups == 2
    assert Conv2dArgs.get_computed_padding(term) == [0, 0, 0, 0]


def test_conv2d_groups_one_equivalent_to_ungrouped() -> None:
    """Explicit ``groups=1`` matches building without a ``groups`` argument."""
    rng = np.random.default_rng(2)
    cin, cout = 3, 5
    h = w = 7
    x = rng.standard_normal((cin, h, w))
    wgt = rng.standard_normal((cout, cin, 3, 3))

    inp = TensorTerm.Tensor("x", [cin, h, w], True)
    wt = TensorTerm.Tensor("w", list(wgt.shape), False)
    a = TensorTerm.conv2d(inp, wt, 1, "same")
    b = TensorTerm.conv2d(inp, wt, 1, "same", groups=1)
    inputs = {"x": x, "w": wgt}
    ya = _eval_term(a, inputs)
    yb = _eval_term(b, inputs)
    assert np.allclose(ya, yb, rtol=0, atol=0)
