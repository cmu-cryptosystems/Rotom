"""Layout checks for stem conv → COMPACT on the fused ResNet-20 stem+layer1 graph.

``test_resnet20_silu_poly_stem_layer1_fused_toy_matches_tensor_eval`` lowers a
``CONV2D`` then ``COMPACT`` on the first CIFAR conv; these assertions document
expected ciphertext counts and target packing.
"""

from __future__ import annotations

import torch

import pytest
from assignment.assignment import LayoutAssignment
from benchmarks.e2e.resnet.resnet_model import resnet20
from benchmarks.e2e.resnet.resnet20_tensor_ir import (
    build_resnet20_silu_poly_graph_through_layer1,
    populate_resnet20_inputs,
)
from ir.kernel import KernelOp
from tests.test_util import get_default_args


@pytest.mark.slow
def test_stem_conv_compact_layout_counts_and_order() -> None:
    """Stem CONV2D uses 16 CTs; COMPACT target uses 1 CT; channel is dim 0 in slots."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()
    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    inputs["input"] = torch.randn(3, 32, 32, dtype=torch.float64).numpy()

    tensor_ir = build_resnet20_silu_poly_graph_through_layer1(inputs)
    args = get_default_args()
    args.backend = "toy"
    args.n = 32768
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_stem_layer1_fused"
    args.channel_gap_align_weight = 0.5

    kernel = LayoutAssignment(tensor_ir, args).run()
    conv_k = compact_k = None
    for k in kernel.post_order():
        if k.op == KernelOp.CONV2D and k.layout.term.cs[1].cs[0] == "conv1_w":
            conv_k = k
        if (
            k.op == KernelOp.COMPACT
            and k.cs[0].op == KernelOp.CONV2D
            and k.cs[0].layout.term.cs[1].cs[0] == "conv1_w"
        ):
            compact_k = k
            break

    assert conv_k is not None and compact_k is not None

    # (1) [0:16:1] on the ciphertext side ⇒ 16 ciphertexts.
    assert conv_k.layout.layout_str() == "[0:16:1];[G:32][1:32:1][2:32:1]"
    assert conv_k.layout.num_ct() == 16

    # (3) After COMPACT, all axes pack into one ciphertext (empty ct_dims product).
    assert compact_k.layout.num_ct() == 1
    assert compact_k.layout.ct_dims == []

    # (4) Target slot order: leading gap, then channel 16, then 32×32 spatial.
    assert compact_k.layout.layout_str() == "[G:2][0:16:1][1:32:1][2:32:1]"
    slot = compact_k.layout.slot_dims
    assert [d.dim for d in slot if d.dim is not None] == [0, 1, 2]
