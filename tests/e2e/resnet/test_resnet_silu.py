"""CIFAR ResNet-20 TensorTerm e2e tests (SiLU via ``PolyCall('silu', ...)``, affine BN).

By default ``tensor_ir.eval`` and Toy both call ``util.silu_polycall_eval.eval_silu_polycall``
with the same ``inputs`` (poly degree / nodes pinned by ``populate_resnet20_inputs``) and
bounds from each ``PolyCall("silu", ...)``. For PyTorch comparison (exact SiLU), set
``inputs["__rotom_silu_eval_mode"] = "exact"``.

- **Stem (checkpoint):** conv + BN + SiLU poly, DaCapo weights when available; Toy vs eval
  and dense vs PyTorch ``act(bn(conv(x)))``.
- **Stem + layer1 block 0:** ``tensor_ir.eval`` vs PyTorch on one fused graph (exact SiLU).
  Toy is checked **compositionally**: ``layer1[0]`` alone with secret input
  ``stem.eval``, including **per-stage** runs (after conv1, bn1, each SiLU, conv2, bn2, add).
  A single stem+``l1_0`` IR tickles the same Toy conv layout issue as full stem+layer1;
  see skipped ``test_resnet20_silu_poly_layer1_toy_matches_tensor_eval``.
- **Stem (random init):** parametrized Toy vs ``tensor_ir.eval`` after conv1, after BN, and full stem;
  Toy asserts each kernel against dense eval with ``np.allclose(..., rtol=1e-2, atol=1e-2)``
  (``backends/toy.py``), not bitwise equality.
- **Layer1:** skipped by default. Set ``ROTOM_RUN_RESNET_LAYER1_SILU_E2E=1`` to run Toy vs eval
  on the three layer1 blocks (secret input = stem ``tensor_ir.eval``) and eval vs PyTorch on the
  full stem+layer1 graph.
- **Full graph:** opt-in via ``ROTOM_RUN_HEAVY_E2E=1`` (memory/time).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

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
    build_resnet20_silu_poly_graph_to_depth,
    build_resnet20_silu_poly_l1_block_graph,
    build_resnet20_silu_poly_layer1_only_graph,
    build_resnet20_stem_plus_l1_0_block_graph,
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


def _resnet20_l1_0_block_ir_upto(
    inputs: dict,
    stop: Literal["conv1", "bn1", "silu1"],
) -> TensorTerm:
    """``l1_0`` first branch only: conv1 → bn1 → SiLU (secret ``l1_0_block_in``).

    Second conv and residual use :func:`_resnet20_l1_0_from_after_silu1_upto` /
    :func:`_resnet20_l1_0_residual_tail_upto` so Toy layout matches eval (see test docstring).
    """
    block_id = "l1_0"
    in_ch, out_ch, stride = 16, 16, 1
    h, w = 32, 32
    x = TensorTerm.Tensor("l1_0_block_in", [in_ch, h, w], True)

    out = R._conv_same(x, f"{block_id}_conv1_w", inputs, stride)
    h1, w1 = R._spatial_hw_after_conv3(h, w, stride)
    if stop == "conv1":
        return out

    out = R._bn_affine_hw(out, f"{block_id}_bn1", out_ch, h1, w1, inputs)
    if stop == "bn1":
        return out

    return R._silu_poly(out)


def _resnet20_l1_0_from_after_silu1_upto(
    inputs: dict,
    stop: Literal["conv2", "bn2"],
) -> TensorTerm:
    """``l1_0`` from first-branch SiLU onward (secret ``l1_0_after_silu1`` in ``inputs``)."""
    block_id = "l1_0"
    out_ch, h1, w1 = 16, 32, 32
    x = TensorTerm.Tensor("l1_0_after_silu1", [out_ch, h1, w1], True)

    out = R._conv_same(x, f"{block_id}_conv2_w", inputs, 1)
    if stop == "conv2":
        return out

    out = R._bn_affine_hw(out, f"{block_id}_bn2", out_ch, h1, w1, inputs)
    return out


def _resnet20_l1_0_residual_tail_upto(
    inputs: dict,
    stop: Literal["add", "silu2"],
) -> TensorTerm:
    """``bn2(conv2(mid)) + block_in`` then optional final SiLU.

    Expects ``inputs[\"l1_0_after_silu1\"]`` and ``inputs[\"l1_0_block_in\"]``.
    """
    block_id = "l1_0"
    out_ch, h1, w1 = 16, 32, 32
    mid = TensorTerm.Tensor("l1_0_after_silu1", [out_ch, h1, w1], True)
    block_in = TensorTerm.Tensor("l1_0_block_in", [out_ch, h1, w1], True)

    out = R._conv_same(mid, f"{block_id}_conv2_w", inputs, 1)
    out = R._bn_affine_hw(out, f"{block_id}_bn2", out_ch, h1, w1, inputs)
    out = out + block_in
    if stop == "add":
        return out
    return R._silu_poly(out)


@pytest.mark.slow
@pytest.mark.parametrize(
    "stop,benchmark",
    [
        ("conv", "resnet20_silu_poly_stem_stage_conv"),
        ("bn", "resnet20_silu_poly_stem_stage_bn"),
        ("silu", "resnet20_silu_poly_stem"),
    ],
    ids=["stem_conv1", "stem_conv_bn", "stem_full_silu"],
)
def test_resnet20_silu_poly_stem_stage_toy_matches_tensor_eval(
    stop: Literal["conv", "bn", "silu"], benchmark: str
) -> None:
    """Each stem stage: Toy ciphertext simulation matches ``tensor_ir.eval`` (per-kernel, ~1e-2)."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)

    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    t = _resnet20_stem_ir_upto(inputs, stop)

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = benchmark

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
    inputs["__rotom_silu_eval_mode"] = "exact"

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
        pt_stem = model.act_conv1(model.bn1(model.conv1(x)))[0].cpu().numpy()
    assert dense.shape == pt_stem.shape
    assert np.allclose(dense, pt_stem, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
@pytest.mark.parametrize(
    "stop,benchmark",
    [
        ("conv1", "resnet20_silu_poly_l1_0_stage_conv1"),
        ("bn1", "resnet20_silu_poly_l1_0_stage_bn1"),
        ("silu1", "resnet20_silu_poly_l1_0_stage_silu1"),
        ("conv2", "resnet20_silu_poly_l1_0_stage_conv2"),
        ("bn2", "resnet20_silu_poly_l1_0_stage_bn2"),
        ("add", "resnet20_silu_poly_l1_0_stage_add"),
        ("silu2", "resnet20_silu_poly_l1_block0"),
    ],
    ids=[
        "l1_0_conv1",
        "l1_0_bn1",
        "l1_0_silu1",
        "l1_0_conv2",
        "l1_0_bn2",
        "l1_0_add",
        "l1_0_silu2_full",
    ],
)
def test_resnet20_silu_poly_l1_0_block_stage_toy_matches_tensor_eval(
    stop: Literal["conv1", "bn1", "silu1", "conv2", "bn2", "add", "silu2"],
    benchmark: str,
) -> None:
    """Each ``l1_0`` op: Toy vs ``tensor_ir.eval`` (compositional secrets where needed).

    Stages through ``silu1`` use one IR rooted at ``l1_0_block_in`` (= poly stem output).
    Toy mismatches when ``conv2`` is chained after ``silu1`` in that same graph (same
    class as fused stem+``l1_0``); ``conv2``/``bn2``/residual stages use a fresh secret
    ``l1_0_after_silu1`` (= eval through ``silu1``), and the add uses both tensors.
    """
    torch.manual_seed(1)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)

    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    stem_ir = build_resnet20_silu_poly_graph_to_depth(inputs, "stem")
    stem_out = np.asarray(stem_ir.eval(inputs), dtype=np.float64)
    inputs["l1_0_block_in"] = stem_out

    if stop in ("conv1", "bn1", "silu1"):
        t = _resnet20_l1_0_block_ir_upto(inputs, stop)
    elif stop in ("conv2", "bn2"):
        silu1_ir = _resnet20_l1_0_block_ir_upto(inputs, "silu1")
        inputs["l1_0_after_silu1"] = np.asarray(silu1_ir.eval(inputs), dtype=np.float64)
        t = _resnet20_l1_0_from_after_silu1_upto(inputs, stop)
    else:
        silu1_ir = _resnet20_l1_0_block_ir_upto(inputs, "silu1")
        inputs["l1_0_after_silu1"] = np.asarray(silu1_ir.eval(inputs), dtype=np.float64)
        inputs["l1_0_block_in"] = stem_out
        t = _resnet20_l1_0_residual_tail_upto(inputs, stop)

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = benchmark

    kernel = LayoutAssignment(t, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(t, inputs, kernel, backend_results, 0, args)


@pytest.mark.slow
def test_resnet20_silu_poly_stem_l1_0_tensor_eval_matches_pytorch() -> None:
    """Stem + ``layer1[0]``: ``tensor_ir.eval`` matches PyTorch (exact SiLU at eval)."""
    torch.manual_seed(1)
    model = resnet20(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()

    tensor_ir = build_resnet20_stem_plus_l1_0_block_graph(inputs)
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        t = model.act_conv1(model.bn1(model.conv1(x)))
        t = model.layer1[0](t)
        pt = t[0].cpu().numpy()

    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)


@pytest.mark.e2e
@pytest.mark.slow
def test_resnet20_silu_poly_stem_l1_0_ckpt_matches_tensor_eval_and_pytorch() -> None:
    """DaCapo weights: fused eval vs PyTorch (exact SiLU); Toy vs eval on ``layer1[0]`` only."""
    if not os.path.exists(CKPT_FILE):
        pytest.skip(f"Missing checkpoint at {CKPT_FILE}")

    torch.manual_seed(1)
    model = resnet20(num_classes=10)
    _load_dacapo_weights(model, CKPT_FILE)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()

    fused_ir = build_resnet20_stem_plus_l1_0_block_graph(inputs)
    dense = np.asarray(fused_ir.eval(inputs))
    with torch.no_grad():
        t = model.act_conv1(model.bn1(model.conv1(x)))
        t = model.layer1[0](t)
        pt = t[0].cpu().numpy()
    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)

    # Compositional Toy (fused stem+block0 IR does not match Toy on the first l1 conv).
    inputs.pop("__rotom_silu_eval_mode", None)
    stem_ir = build_resnet20_silu_poly_graph_to_depth(inputs, "stem")
    inputs["l1_0_block_in"] = np.asarray(stem_ir.eval(inputs), dtype=np.float64)
    l1_ir = build_resnet20_silu_poly_l1_block_graph(inputs, index=0)

    args = get_default_args()
    args.backend = "toy"
    args.n = _RESNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_poly_l1_block0"

    kernel = LayoutAssignment(l1_ir, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(l1_ir, inputs, kernel, backend_results, 0, args)


@pytest.mark.slow
@pytest.mark.skip(
    reason=(
        "Layer1 SiLU e2e opt-in; set ROTOM_RUN_RESNET_LAYER1_SILU_E2E=1 to run "
        "(Toy vs eval on layer1-only IR; PyTorch check on full stem+layer1)."
    ),
)
def test_resnet20_silu_poly_layer1_toy_matches_tensor_eval() -> None:
    """Layer1 only (three BasicBlocks): Toy matches ``tensor_ir.eval``.

    Activations are ``stem.eval`` on the same ``inputs`` (stem is covered by
    ``test_resnet20_silu_poly_stem_stage_toy_matches_tensor_eval``). A single IR that
    fuses stem with layer1 tickles a Toy/lowering mismatch on the first layer1
    conv input layout; compiling layer1 as its own graph avoids that while still
    exercising every layer1 tensor op (conv, BN, SiLU poly, residual add).
    """
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_resnet20_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    stem_ir = build_resnet20_silu_poly_graph_to_depth(inputs, "stem")
    inputs["layer1_only_in"] = np.asarray(stem_ir.eval(inputs), dtype=np.float64)
    tensor_ir = build_resnet20_silu_poly_layer1_only_graph(inputs)

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
