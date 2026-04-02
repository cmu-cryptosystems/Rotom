"""Unit tests for each unique Conv2d in CIFAR ResNet-20 (SiLU model).

``benchmarks.e2e.resnet.resnet_model.resnet20`` uses Option-A shortcuts, so every
learnable conv is 3×3 with ``padding=1``. Distinct layer types are exactly six
``(in_channels, out_channels, stride)`` pairs:

1. Stem ``3 → 16``, stride 1, on ``32×32``
2. Same-width block ``16 → 16``, stride 1, on ``32×32`` (layer1)
3. Downsample ``16 → 32``, stride 2, on ``32×32`` (layer2 block0 conv1)
4. Same-width ``32 → 32``, stride 1, on ``16×16`` (layer2 blocks 1–2)
5. Downsample ``32 → 64``, stride 2, on ``16×16`` (layer3 block0 conv1)
6. Same-width ``64 → 64``, stride 1, on ``8×8`` (layer3 blocks 1–2)

Each test builds a single ``TensorTerm.conv2d`` with weights taken from that
representative ``nn.Conv2d``, compares LayoutAssignment → Lower → Toy against
``tensor_ir.eval`` via ``check_results`` (same pattern as ``test_resnet_silu``).
``test_resnet20_stem_plus_layer1_first_conv_toy`` chains stem + layer1.0's first
conv (no BN/SiLU) on ``32×32``. Ring dimension ``n=32768`` matches the other tests.
"""

from __future__ import annotations

import numpy as np
import torch

import pytest

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from benchmarks.e2e.resnet.resnet_model import resnet20
from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.test_util import get_default_args
from util.checker import check_results

_RESNET20_CONV_N = 32768


def _run_conv_matches_toy(
    *,
    conv: torch.nn.Conv2d,
    height: int,
    width: int,
    benchmark: str,
) -> None:
    assert conv.kernel_size == (3, 3) and conv.padding == (1, 1)
    stride = conv.stride[0]
    cin = conv.in_channels
    _cout = conv.out_channels

    torch.manual_seed(0)
    x = torch.randn(1, cin, height, width, dtype=torch.float32)

    w = conv.weight.detach().cpu().numpy().astype("float64")
    x_np = x.squeeze(0).cpu().numpy().astype("float64")

    inp = TensorTerm.Tensor("input", [cin, height, width], True)
    wt = TensorTerm.Tensor("weight", list(w.shape), False)
    tensor_ir = TensorTerm.conv2d(inp, wt, stride, "same")

    inputs = {"input": x_np, "weight": w}

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET20_CONV_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = benchmark

    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(tensor_ir, inputs, kernel, backend_results, 0, args)


@pytest.mark.slow
def test_resnet20_stem_plus_layer1_first_conv_toy() -> None:
    """Stem (3→16) then layer1.0 conv1 (16→16), both 3×3 stride-1 same as in ``resnet20``."""
    model = resnet20()
    w_stem = model.conv1.weight.detach().cpu().numpy().astype(np.float64)
    w_l1 = model.layer1[0].conv1.weight.detach().cpu().numpy().astype(np.float64)

    torch.manual_seed(0)
    x_np = (
        torch.randn(1, 3, 32, 32, dtype=torch.float32)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    inp = TensorTerm.Tensor("input", [3, 32, 32], True)
    t_stem = TensorTerm.Tensor("conv1_w", list(w_stem.shape), False)
    t_l1 = TensorTerm.Tensor("l1_0_conv1_w", list(w_l1.shape), False)
    h = TensorTerm.conv2d(inp, t_stem, 1, "same")
    tensor_ir = TensorTerm.conv2d(h, t_l1, 1, "same")

    inputs = {"input": x_np, "conv1_w": w_stem, "l1_0_conv1_w": w_l1}

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET20_CONV_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_conv_stem_plus_l1_0"

    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(tensor_ir, inputs, kernel, backend_results, 0, args)


@pytest.mark.slow
def test_resnet20_unique_conv_stem_3_to_16_stride1() -> None:
    model = resnet20()
    _run_conv_matches_toy(
        conv=model.conv1, height=32, width=32, benchmark="resnet20_conv_stem"
    )


@pytest.mark.slow
def test_resnet20_unique_conv_16_to_16_stride1() -> None:
    model = resnet20()
    _run_conv_matches_toy(
        conv=model.layer1[0].conv1,
        height=32,
        width=32,
        benchmark="resnet20_conv_16_16_s1",
    )


@pytest.mark.slow
def test_resnet20_unique_conv_16_to_32_stride2() -> None:
    model = resnet20()
    _run_conv_matches_toy(
        conv=model.layer2[0].conv1,
        height=32,
        width=32,
        benchmark="resnet20_conv_16_32_s2",
    )


@pytest.mark.slow
def test_resnet20_unique_conv_32_to_32_stride1() -> None:
    model = resnet20()
    _run_conv_matches_toy(
        conv=model.layer2[1].conv1,
        height=16,
        width=16,
        benchmark="resnet20_conv_32_32_s1",
    )


@pytest.mark.slow
def test_resnet20_unique_conv_32_to_64_stride2() -> None:
    model = resnet20()
    _run_conv_matches_toy(
        conv=model.layer3[0].conv1,
        height=16,
        width=16,
        benchmark="resnet20_conv_32_64_s2",
    )


@pytest.mark.slow
def test_resnet20_unique_conv_64_to_64_stride1() -> None:
    model = resnet20()
    _run_conv_matches_toy(
        conv=model.layer3[1].conv1,
        height=8,
        width=8,
        benchmark="resnet20_conv_64_64_s1",
    )
