"""CIFAR ResNet-20 TensorTerm e2e tests (SiLU via ``PolyCall('silu', ...)``, affine BN).

By default ``tensor_ir.eval`` and Toy both call ``util.silu_polycall_eval.eval_silu_polycall``
with the same ``inputs`` (poly degree / nodes pinned by ``populate_resnet20_inputs``) and
bounds from each ``PolyCall("silu", ...)``. For PyTorch comparison (exact SiLU), set
``inputs["__rotom_silu_eval_mode"] = "exact"``.

- **Core coverage only:** stem, stem+layer1, stem+layer1+layer2, stem through layer3, and full end-to-end.
- For each of stem and stem+layer1 we keep both comparisons:
  - Toy vs ``tensor_ir.eval`` (same SiLU poly implementation)
  - ``tensor_ir.eval`` vs PyTorch (exact SiLU at eval)
- **Full graph:** Toy vs ``tensor_ir.eval`` is opt-in via ``ROTOM_RUN_HEAVY_E2E=1`` (memory/time).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from benchmarks.e2e.resnet import resnet20_tensor_ir as R
from benchmarks.e2e.resnet.resnet_model import resnet20
from benchmarks.e2e.resnet.resnet20_tensor_ir import (
    build_resnet20_silu_poly_graph,
    build_resnet20_silu_poly_graph_through_layer1,
    build_resnet20_silu_poly_graph_through_layer2,
    build_resnet20_silu_poly_graph_through_layer3,
    populate_resnet20_inputs,
)
from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.test_util import get_default_args
from util.checker import check_results

_RESNET_TOY_N = 32768
_SKIP_ON_CI = os.environ.get("CI", "").strip().lower() in {"1", "true", "yes", "on"}
_RUN_HEAVY_E2E = os.environ.get("ROTOM_RUN_HEAVY_E2E", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _resnet20_stem_ir_upto(
    inputs: dict, stop: Literal["conv", "bn", "silu"]
) -> TensorTerm:
    """ResNet-20 stem prefix: ``conv1`` only, ``+ bn1`` (affine), or ``+ SiLU`` poly (full stem)."""
    h, w = 32, 32
    t = TensorTerm.Tensor("input", [3, 32, 32], True)
    t = R._conv_same(t, "conv1_w", inputs, 1)
    h, w = R._spatial_hw_after_conv3(h, w, 1)
    if stop == "conv":
        return t
    t = R._bn_affine_hw(t, "bn1", 16, h, w, inputs)
    if stop == "bn":
        return t
    return R._silu_poly(t)


@pytest.mark.slow
def test_resnet20_silu_poly_stem_toy_matches_tensor_eval() -> None:
    """Stem (conv+BN+SiLU poly): Toy matches ``tensor_ir.eval``."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)

    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    t = _resnet20_stem_ir_upto(inputs, "silu")

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_poly_stem"

    kernel = LayoutAssignment(t, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(t, inputs, kernel, backend_results, 0, args)


@pytest.mark.slow
def test_resnet20_silu_poly_stem_tensor_eval_matches_pytorch() -> None:
    """Stem (SiLU poly IR): ``tensor_ir.eval`` matches PyTorch (exact SiLU at eval)."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()

    t = _resnet20_stem_ir_upto(inputs, "silu")
    dense = np.asarray(t.eval(inputs))

    with torch.no_grad():
        pt_stem = model.act_conv1(model.bn1(model.conv1(x)))[0].cpu().numpy()

    assert dense.shape == pt_stem.shape
    assert np.allclose(dense, pt_stem, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
def test_resnet20_silu_poly_stem_layer1_fused_toy_matches_tensor_eval() -> None:
    """Fused stem + all ``layer1`` blocks: Toy matches ``tensor_ir.eval``.

    One IR from ``input`` through three BasicBlocks needs channel dim 0 adjacent to leading
    ``[G:*]`` after replication/rolls, or the first layer1 conv mismatches Toy. A small
    ``channel_gap_align_weight`` encodes that packing preference in layout assignment.
    """
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    tensor_ir = build_resnet20_silu_poly_graph_through_layer1(inputs)

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_stem_layer1_fused"
    args.channel_gap_align_weight = 0.5

    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(tensor_ir, inputs, kernel, backend_results, 0, args)


@pytest.mark.slow
def test_resnet20_silu_poly_stem_layer1_layer2_fused_toy_matches_tensor_eval() -> None:
    """Fused stem + layer1 + layer2: Toy matches ``tensor_ir.eval``.

    Includes layer2.0 stride-2 downsample and two 32→32 blocks on 16×16. Same layout
    hint as stem+layer1 fused (channel dim 0 near leading gap after rolls).
    """
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    tensor_ir = build_resnet20_silu_poly_graph_through_layer2(inputs)

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_stem_layer1_layer2_fused"
    args.channel_gap_align_weight = 0.5

    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(tensor_ir, inputs, kernel, backend_results, 0, args)


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "Toy vs eval mismatch on layer3 stride-2 convs at chosen layouts "
        "(e.g. ``[0:64:1];[G:32][1:8:1]…``); ``channel_gap_align_weight`` does not "
        "steer assignment away. Remove xfail when ``lower_conv2d`` handles this case."
    ),
    raises=AssertionError,
    strict=False,
)
def test_resnet20_silu_poly_stem_through_layer3_fused_toy_matches_tensor_eval() -> None:
    """Fused stem + layer1 + layer2 + layer3: Toy matches ``tensor_ir.eval``.

    Output is ``[64, 8, 8]`` activations before pooling/FC. Expect long runtime (Toy
    over the full prefix); same ``channel_gap_align_weight`` as other fused SiLU tests.
    """
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    tensor_ir = build_resnet20_silu_poly_graph_through_layer3(inputs)

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_stem_through_layer3_fused"
    args.channel_gap_align_weight = 0.5

    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(tensor_ir, inputs, kernel, backend_results, 0, args)


@pytest.mark.slow
def test_resnet20_silu_poly_layer1_tensor_eval_matches_pytorch() -> None:
    """Stem + layer1 SiLU-poly IR: ``tensor_ir.eval`` matches PyTorch (exact SiLU at eval)."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()

    tensor_ir = build_resnet20_silu_poly_graph_through_layer1(inputs)
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        t = model.act_conv1(model.bn1(model.conv1(x)))
        for i in range(3):
            t = model.layer1[i](t)
        pt = t[0].cpu().numpy()

    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
@pytest.mark.skipif(
    _SKIP_ON_CI or not _RUN_HEAVY_E2E,
    reason="too heavy (memory/time) for default test runs; set ROTOM_RUN_HEAVY_E2E=1 to opt in",
)
def test_resnet20_silu_poly_toy_matches_tensor_eval() -> None:
    """Full SiLU-poly ResNet-20 tensor graph: layout, lower, Toy, then ``check_results``."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    os.environ.setdefault(
        "ROTOM_APPLY_LAYOUT_PLAN_CACHE_DIR",
        str(
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "rotom"
            / "apply_layout_plans"
        ),
    )

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)

    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    tensor_ir = build_resnet20_silu_poly_graph(inputs)

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_poly_l2_0"

    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()

    check_results(tensor_ir, inputs, kernel, backend_results, 0, args)
