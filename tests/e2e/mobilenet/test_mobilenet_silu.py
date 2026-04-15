"""MobileNetSmall TensorTerm e2e tests (SiLU via ``PolyCall('silu', ...)``)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.e2e.mobilenet.mobilenet_model import mobilenet_small
from benchmarks.e2e.mobilenet.mobilenet_tensor_ir import (
    build_mobilenet_small_silu_poly_graph,
    build_mobilenet_small_silu_poly_graph_to_depth,
    populate_mobilenet_small_inputs,
)


@pytest.mark.slow
def test_mobilenet_small_depth_api_full_matches_graph() -> None:
    """``depth=\"full\"`` matches :func:`build_mobilenet_small_silu_poly_graph`."""
    torch.manual_seed(1)
    model = mobilenet_small(num_classes=10)
    model.eval()
    inputs: dict = {}
    populate_mobilenet_small_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    a = np.asarray(build_mobilenet_small_silu_poly_graph(inputs).eval(inputs))
    b = np.asarray(
        build_mobilenet_small_silu_poly_graph_to_depth(inputs, "full").eval(inputs)
    )
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=1e-15, atol=1e-15)


@pytest.mark.slow
def test_mobilenet_small_silu_poly_stem_tensor_eval_matches_pytorch() -> None:
    """Stem: ``tensor_ir.eval`` matches PyTorch (exact SiLU)."""
    torch.manual_seed(0)
    model = mobilenet_small(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_mobilenet_small_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()
    tensor_ir = build_mobilenet_small_silu_poly_graph_to_depth(inputs, "stem")
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        pt = model.act_stem(model.bn_stem(model.conv_stem(x)))[0].cpu().numpy()

    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
def test_mobilenet_small_silu_poly_stage2_tensor_eval_matches_pytorch() -> None:
    """Through block1: ``tensor_ir.eval`` matches PyTorch (exact SiLU)."""
    torch.manual_seed(0)
    model = mobilenet_small(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_mobilenet_small_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()
    tensor_ir = build_mobilenet_small_silu_poly_graph_to_depth(inputs, "stage2")
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        pt = model.forward_features_through_stage2(x)[0].cpu().numpy()

    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
def test_mobilenet_small_silu_poly_stage3_tensor_eval_matches_pytorch() -> None:
    """Through block2: ``tensor_ir.eval`` matches PyTorch (exact SiLU)."""
    torch.manual_seed(0)
    model = mobilenet_small(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_mobilenet_small_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()
    tensor_ir = build_mobilenet_small_silu_poly_graph_to_depth(inputs, "stage3")
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        pt = model.forward_features_through_stage3(x)[0].cpu().numpy()

    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)


@pytest.mark.slow
def test_mobilenet_small_silu_poly_full_tensor_eval_matches_pytorch() -> None:
    """Full logits: ``tensor_ir.eval`` matches PyTorch (exact SiLU)."""
    torch.manual_seed(0)
    model = mobilenet_small(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_mobilenet_small_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()
    tensor_ir = build_mobilenet_small_silu_poly_graph_to_depth(inputs, "full")
    dense = np.asarray(tensor_ir.eval(inputs))

    with torch.no_grad():
        pt = model(x).cpu().numpy()

    assert dense.shape == pt.shape
    assert np.allclose(dense, pt, rtol=1e-9, atol=1e-7)
