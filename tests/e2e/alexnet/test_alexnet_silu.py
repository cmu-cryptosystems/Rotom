"""AlexNetSmall TensorTerm e2e tests (SiLU via ``PolyCall('silu', ...)``)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from benchmarks.e2e.alexnet.alexnet_model import alexnet_small
from benchmarks.e2e.alexnet.alexnet_tensor_ir import (
    build_alexnet_small_silu_poly_graph,
    build_alexnet_small_silu_poly_graph_through_stage2,
    build_alexnet_small_silu_poly_graph_to_depth,
    populate_alexnet_small_inputs,
)
from lower.lower import Lower
from tests.test_util import get_default_args
from util.checker import check_results

_ALEXNET_TOY_N = 32768
_RUN_HEAVY_E2E = os.environ.get("ROTOM_RUN_HEAVY_E2E", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@pytest.mark.slow
def test_alexnet_small_depth_api_stage2_matches_helper() -> None:
    """``build_alexnet_small_silu_poly_graph_to_depth(..., \"stage2\")`` matches legacy helper."""
    torch.manual_seed(0)
    model = alexnet_small(num_classes=10)
    model.eval()
    inputs: dict = {}
    populate_alexnet_small_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    a = np.asarray(
        build_alexnet_small_silu_poly_graph_through_stage2(inputs).eval(inputs)
    )
    b = np.asarray(
        build_alexnet_small_silu_poly_graph_to_depth(inputs, "stage2").eval(inputs)
    )
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=1e-15, atol=1e-15)


@pytest.mark.slow
def test_alexnet_small_depth_api_full_matches_graph() -> None:
    """``depth=\"full\"`` matches :func:`build_alexnet_small_silu_poly_graph`."""
    torch.manual_seed(1)
    model = alexnet_small(num_classes=10)
    model.eval()
    inputs: dict = {}
    populate_alexnet_small_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    a = np.asarray(build_alexnet_small_silu_poly_graph(inputs).eval(inputs))
    b = np.asarray(
        build_alexnet_small_silu_poly_graph_to_depth(inputs, "full").eval(inputs)
    )
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=1e-15, atol=1e-15)


@pytest.mark.slow
def test_alexnet_small_silu_poly_stem_tensor_eval_matches_pytorch() -> None:
    """Stem only: ``tensor_ir.eval`` matches PyTorch (exact SiLU)."""
    torch.manual_seed(0)
    model = alexnet_small(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_alexnet_small_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()
    tensor_ir = build_alexnet_small_silu_poly_graph_to_depth(inputs, "stem")
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        pt = model.act1(model.bn1(model.conv1(x)))[0].cpu().numpy()

    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
def test_alexnet_small_silu_poly_stage3_tensor_eval_matches_pytorch() -> None:
    """Through conv3: ``tensor_ir.eval`` matches PyTorch (exact SiLU)."""
    torch.manual_seed(0)
    model = alexnet_small(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_alexnet_small_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()
    tensor_ir = build_alexnet_small_silu_poly_graph_to_depth(inputs, "stage3")
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        pt = model.forward_features_through_stage3(x)[0].cpu().numpy()

    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
def test_alexnet_small_silu_poly_full_tensor_eval_matches_pytorch() -> None:
    """Full graph logits: ``tensor_ir.eval`` matches PyTorch (exact SiLU)."""
    torch.manual_seed(0)
    model = alexnet_small(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_alexnet_small_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()
    tensor_ir = build_alexnet_small_silu_poly_graph_to_depth(inputs, "full")
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        pt = model(x).cpu().numpy()

    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
def test_alexnet_small_silu_poly_stage2_toy_matches_tensor_eval() -> None:
    """Prefix through conv2: Toy output matches tensor_ir.eval."""
    torch.manual_seed(0)
    model = alexnet_small(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_alexnet_small_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    tensor_ir = build_alexnet_small_silu_poly_graph_through_stage2(inputs)

    args = get_default_args()
    args.backend = "toy"
    args.n = _ALEXNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "alexnet_small_silu_stage2"
    args.skip_toy_eval_checks = True
    args.toy_mp_workers = 1

    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(tensor_ir, inputs, kernel, backend_results, 0, args)


@pytest.mark.slow
def test_alexnet_small_silu_poly_stage2_tensor_eval_matches_pytorch() -> None:
    """Prefix through conv2: dense eval matches PyTorch with exact SiLU."""
    torch.manual_seed(0)
    model = alexnet_small(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_alexnet_small_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()
    tensor_ir = build_alexnet_small_silu_poly_graph_through_stage2(inputs)
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        pt = model.forward_features_through_stage2(x)[0].cpu().numpy()

    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_HEAVY_E2E,
    reason="full AlexNetSmall Toy e2e is heavy; set ROTOM_RUN_HEAVY_E2E=1 to run",
)
def test_alexnet_small_silu_poly_full_toy_matches_tensor_eval() -> None:
    """Full AlexNetSmall graph: Toy output matches tensor_ir.eval."""
    torch.manual_seed(0)
    model = alexnet_small(num_classes=10)
    model.eval()

    inputs: dict = {}
    populate_alexnet_small_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    tensor_ir = build_alexnet_small_silu_poly_graph(inputs)

    args = get_default_args()
    args.backend = "toy"
    args.n = _ALEXNET_TOY_N
    args.rolls = True
    args.net = "lan"
    args.benchmark = "alexnet_small_silu_full"
    args.skip_toy_eval_checks = True
    args.toy_mp_workers = 1

    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()
    check_results(tensor_ir, inputs, kernel, backend_results, 0, args)
