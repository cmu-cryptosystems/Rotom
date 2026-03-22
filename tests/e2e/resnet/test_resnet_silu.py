"""CIFAR ResNet-20 TensorTerm e2e tests (SiLU via ``PolyCall('silu', ...)``, affine BN).

TensorEvaluator implements ``func == 'silu'`` as exact SiLU for ``tensor_ir.eval``; Toy
uses the clipped polynomial for HE. ``check_results`` tolerances apply Toy vs eval.

- **Stem (checkpoint):** conv + BN + SiLU poly, DaCapo weights when available; Toy vs eval
  and dense vs PyTorch ``act(bn(conv(x)))``.
- **Stem (random init):** same IR without checkpoint (always runs in slow suite).
- **Layer1:** dense eval vs PyTorch; Toy path for the full stem+layer1 graph is **xfail**
  until Toy matches eval on that chain.
- **Full graph:** opt-in via ``ROTOM_RUN_HEAVY_E2E=1`` (memory/time).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from benchmarks.e2e.resnet import resnet20_tensor_ir as R
from benchmarks.e2e.resnet.resnet_data import CKPT_FILE, load_checkpoint
from benchmarks.e2e.resnet.resnet_model import resnet20
from benchmarks.e2e.resnet.resnet20_tensor_ir import (
    build_resnet20_silu_poly_graph,
    build_resnet20_silu_poly_graph_through_layer1,
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


def _load_dacapo_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is None or not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise FileNotFoundError(
            f"Missing/invalid DaCapo checkpoint at {ckpt_path} (need dict with 'state_dict')."
        )
    state: Any = ckpt["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected 'state_dict' type: {type(state)}")
    if any(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)


@pytest.mark.slow
def test_resnet20_silu_poly_stem_toy_matches_tensor_eval() -> None:
    """Stem + BN + SiLU poly through Toy matches tensor eval (random init weights)."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)

    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    h, w = 32, 32
    t = TensorTerm.Tensor("input", [3, 32, 32], True)
    t = R._conv_same(t, "conv1_w", inputs, 1)
    h, w = R._spatial_hw_after_conv3(h, w, 1)
    t = R._bn_affine_hw(t, "bn1", 16, h, w, inputs)
    t = R._silu_poly(t)

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


@pytest.mark.e2e
@pytest.mark.slow
def test_resnet20_silu_poly_stem_ckpt_matches_tensor_eval_and_pytorch() -> None:
    """Stem + SiLU poly: Toy and ``tensor_ir.eval`` vs PyTorch (DaCapo checkpoint)."""
    if not os.path.exists(CKPT_FILE):
        pytest.skip(f"Missing checkpoint at {CKPT_FILE}")

    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    _load_dacapo_weights(model, CKPT_FILE)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()

    h, w = 32, 32
    t = TensorTerm.Tensor("input", [3, 32, 32], True)
    t = R._conv_same(t, "conv1_w", inputs, 1)
    h, w = R._spatial_hw_after_conv3(h, w, 1)
    t = R._bn_affine_hw(t, "bn1", 16, h, w, inputs)
    t = R._silu_poly(t)

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

    dense = t.eval(inputs)
    with torch.no_grad():
        pt_stem = model.act(model.bn1(model.conv1(x)))[0].cpu().numpy()
    assert dense.shape == pt_stem.shape
    assert np.allclose(dense, pt_stem, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "Toy diverges from tensor_ir.eval for stem+layer1 in one graph; "
        "see test_resnet20_silu_poly_layer1_tensor_eval_matches_pytorch."
    ),
    strict=False,
)
def test_resnet20_silu_poly_layer1_toy_matches_tensor_eval() -> None:
    """Stem + layer1 (three BasicBlocks, 32×32) through Toy matches tensor eval."""
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
    args.benchmark = "resnet20_silu_poly_l1"

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

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()

    tensor_ir = build_resnet20_silu_poly_graph_through_layer1(inputs)
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        t = model.act(model.bn1(model.conv1(x)))
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
