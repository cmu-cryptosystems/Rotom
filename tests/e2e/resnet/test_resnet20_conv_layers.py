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
Ring dimension ``n=32768`` matches the user's request.
"""

from __future__ import annotations

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
