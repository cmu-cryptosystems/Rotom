"""Tests that TensorTerm.layout is honored during layout assignment."""

import numpy as np
import pytest

from assignment.assignment import LayoutAssignment
from assignment.frontend_layout import filter_kernels_by_frontend_layout
from assignment.gen.gen_tensor import gen_tensor
from frontends.tensor import TensorTerm
from ir.layout import Layout
from ir.layout_utils import dimension_merging
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args


def test_filter_keeps_matching_kernel_only():
    term = TensorTerm.Tensor("a", [4, 4], True, layout="[0:4:1][1:4:1]")
    n = 16
    secret = True
    kernels = gen_tensor(term, secret, [4, 4], n)
    assert len(kernels) > 1

    filtered = filter_kernels_by_frontend_layout(kernels, term, n, secret)
    assert len(filtered) == 1
    ref = Layout.from_string("[0:4:1][1:4:1]", n, secret)
    ref.term = term
    want = dimension_merging(ref)
    assert dimension_merging(filtered[0].layout).layout_str() == want.layout_str()


def test_filter_raises_when_no_candidate_matches():
    # Valid layout string, but gen_tensor does not emit roll kernels for plain 4x4 inputs.
    term = TensorTerm.Tensor("a", [4, 4], True, layout="roll(0,1) [1:4:1][0:4:1]")
    n = 16
    kernels = gen_tensor(term, True, [4, 4], n)
    with pytest.raises(ValueError, match="does not match any candidate"):
        filter_kernels_by_frontend_layout(kernels, term, n, True)


def test_layout_assignment_pins_matmul_output_layout():
    args = get_default_args()
    args.n = 16
    args.fuzz = False

    a = TensorTerm.Tensor("a", [4, 4], True, layout="[0:4:1][1:4:1]")
    b = TensorTerm.Tensor("b", [4, 4], False, layout="[0:4:1][1:4:1]")
    # For this graph, binop + matmul emits a single matmul output layout candidate.
    expr = (a + b).matmul(a, layout="[0:4:1][1:4:1]")

    kernel = LayoutAssignment(expr, args).run()
    assert dimension_merging(kernel.layout).layout_str() == "[0:4:1][1:4:1]"


def test_end_to_end_toy_backend_respects_input_layout():
    args = get_default_args()
    args.n = 16
    args.fuzz = False
    args.backend = "toy"

    a = TensorTerm.Tensor("a", [4, 4], True, layout="[0:4:1][1:4:1]")
    b = TensorTerm.Tensor("b", [4, 4], False, layout="[0:4:1][1:4:1]")
    expr = a + b

    inputs = {
        "a": np.arange(16, dtype=float).reshape(4, 4),
        "b": np.ones((4, 4)),
    }
    expected = expr.eval(inputs)
    results, kernel = run_compiler_and_backend(expr, inputs, args, "toy")

    from util.layout_util import apply_layout

    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, "toy")
