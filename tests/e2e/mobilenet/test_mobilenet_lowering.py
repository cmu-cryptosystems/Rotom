"""MobileNetSmall lowering/inference parity tests on Toy backend."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.e2e.mobilenet.mobilenet_model import mobilenet_small
from benchmarks.e2e.mobilenet.mobilenet_tensor_ir import (
    build_mobilenet_small_silu_poly_graph_to_depth,
    populate_mobilenet_small_inputs,
)
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Current layout assignment cannot fully lower the MobileNet graph to Toy yet.",
    strict=False,
)
def test_mobilenet_small_full_toy_lowering_matches_eval() -> None:
    """Full MobileNetSmall TensorIR compiles through lowering and matches eval on Toy."""
    torch.manual_seed(3)
    model = mobilenet_small(num_classes=10)
    model.eval()
    model.double()

    inputs: dict = {}
    populate_mobilenet_small_inputs(model, inputs)
    inputs["__rotom_silu_eval_mode"] = "exact"

    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    inputs["input"] = x[0].cpu().numpy()
    tensor_ir = build_mobilenet_small_silu_poly_graph_to_depth(inputs, "full")
    expected = np.asarray(tensor_ir.eval(inputs))

    args = get_default_args()
    args.backend = "toy"
    args.n = 4096
    args.rolls = True
    args.conv_roll = False
    args.benchmark = "mobilenet_small_toy_lowering"

    results, kernel = run_compiler_and_backend(tensor_ir, inputs, args, "toy")
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, "toy")
