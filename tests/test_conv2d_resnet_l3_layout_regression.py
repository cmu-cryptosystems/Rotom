"""Regression tests for the ResNet-20 layer3 conv output packing that exposes Toy mismatch.

On CIFAR ResNet-20 with ``n=32768``, ``rolls=True``, and
``channel_gap_align_weight=0.5`` (same as fused SiLU e2e tests), layout assignment
picks this **CONV2D output** layout for ``l3_0_conv1_w`` (stride-2, ``32→64``,
``16×16 → 8×8``):

``[0:64:1];[G:32][1:8:1][G:4][2:8:1][G:4]``

Output channel (logical dim ``0``) lives on the ciphertext axis (64 ciphertexts);
spatial dims ``1``/``2`` are 8 with interleaved gap slots. Layer3 Toy vs eval failures
on ``l3_0_conv2`` were traced to ``gen_conv2d`` using unit stride on the weight
``C_in`` slot axis while activations used a split channel stride (fixed by copying
``dim.stride`` from activation ``dim 0`` onto ``Dim(1, …)``).

The tests here pin that assignment choice and validate basic layout invariants so
future changes to ``gen_conv2d`` / heuristics are explicit.
"""

from __future__ import annotations

import torch

import pytest

from assignment.assignment import LayoutAssignment
from benchmarks.e2e.resnet.resnet_model import resnet20
from benchmarks.e2e.resnet.resnet20_tensor_ir import (
    build_resnet20_silu_poly_graph_through_layer3,
    populate_resnet20_inputs,
)
from frontends.tensor import TensorOp, TensorTerm
from ir.dim import DimType
from ir.kernel import KernelOp
from ir.layout import Layout
from tests.test_util import get_default_args

# Golden string observed for ``l3_0_conv1_w`` under fused ResNet-20 SiLU settings below.
RESNET20_L3_0_CONV1_OUTPUT_LAYOUT = "[0:64:1];[G:32][1:8:1][G:4][2:8:1][G:4]"

_RESNET_TOY_N = 32768


def test_resnet_l3_conv1_failing_layout_string_parses_and_has_channel_on_ct_axis() -> (
    None
):
    """Sanity-check the documented layout string: Cout on ``ct_dims``, 8×8 spatial in slots."""
    layout = Layout.from_string(
        RESNET20_L3_0_CONV1_OUTPUT_LAYOUT, _RESNET_TOY_N, secret=True
    )
    assert layout.num_ct() == 64
    assert len(layout.ct_dims) == 1
    assert layout.ct_dims[0].dim == 0 and layout.ct_dims[0].extent == 64
    slot = layout.slot_dims
    assert [d.dim for d in slot if d.dim is not None] == [1, 2]
    assert sum(1 for d in slot if d.dim_type == DimType.EMPTY) == 3


@pytest.mark.slow
def test_resnet20_silu_through_layer3_l3_0_conv1_kernel_layout_matches_documented_packing() -> (
    None
):
    """LayoutAssignment on the full stem–layer3 SiLU graph picks the known ``l3_0_conv1_w`` packing."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    inputs["input"] = torch.randn(3, 32, 32, dtype=torch.float64).numpy()

    tensor_ir = build_resnet20_silu_poly_graph_through_layer3(inputs)

    args = get_default_args()
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_l3_conv1_layout_regression"
    args.channel_gap_align_weight = 0.5

    kernel = LayoutAssignment(tensor_ir, args).run()

    conv_k = None
    for k in kernel.post_order():
        if k.op != KernelOp.CONV2D:
            continue
        term = k.layout.term
        if not isinstance(term.cs[1], TensorTerm):
            continue
        if term.cs[1].op == TensorOp.TENSOR and term.cs[1].cs[0] == "l3_0_conv1_w":
            conv_k = k
            break

    assert conv_k is not None, "expected a CONV2D kernel for weight name l3_0_conv1_w"
    assert conv_k.layout.layout_str() == RESNET20_L3_0_CONV1_OUTPUT_LAYOUT


@pytest.mark.slow
def test_resnet20_silu_through_layer3_l3_0_conv2_weight_stride_matches_activation_channel_split() -> (
    None
):
    """``gen_conv2d`` must copy activation channel strides onto ``C_in`` (e.g. ``[1:32:2]``)."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    inputs["input"] = torch.randn(3, 32, 32, dtype=torch.float64).numpy()

    tensor_ir = build_resnet20_silu_poly_graph_through_layer3(inputs)

    args = get_default_args()
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_l3_conv2_stride_regression"
    args.channel_gap_align_weight = 0.5

    kernel = LayoutAssignment(tensor_ir, args).run()

    conv_k = None
    for k in kernel.post_order():
        if k.op != KernelOp.CONV2D:
            continue
        term = k.layout.term
        if not isinstance(term.cs[1], TensorTerm):
            continue
        if term.cs[1].op == TensorOp.TENSOR and term.cs[1].cs[0] == "l3_0_conv2_w":
            conv_k = k
            break

    assert conv_k is not None, "expected CONV2D for l3_0_conv2_w"
    b_str = conv_k.cs[1].layout.layout_str()
    assert "[1:32:2]" in b_str, b_str
