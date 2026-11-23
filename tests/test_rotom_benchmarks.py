"""The registered benchmark graphs build, and the small ones run end to end."""

import pytest

from assignment.assignment import LayoutAssignment
from benchmarks.rotom_benchmarks.mlp_mnist_square import mlp_mnist_square
from benchmarks.rotom_benchmarks.resnet_silu import resnet_silu, resnet_silu_one_layer
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def _args(n, rolls, name):
    args = get_default_args()
    args.n = n
    args.rolls = rolls
    args.backend = "toy"
    args.benchmark = name
    return args


def _assert_matches_eval(term, inputs, args):
    results, kernel = run_compiler_and_backend(term, inputs, args, "toy")
    expected = apply_layout(term.eval(inputs), kernel.layout)
    assert_results_equal(expected, results, "toy")


@pytest.mark.parametrize("rolls", [False, True])
def test_mlp_mnist_square_matches_eval(rolls):
    term, inputs = mlp_mnist_square(hidden_size=16)
    _assert_matches_eval(term, inputs, _args(256, rolls, f"mlp_mnist_{rolls}"))


@pytest.mark.parametrize("rolls", [False, True])
def test_resnet_silu_one_layer_matches_eval(rolls):
    term, inputs, n = resnet_silu_one_layer()
    _assert_matches_eval(term, inputs, _args(n, rolls, f"resnet_layer_{rolls}"))


def test_resnet_silu_gets_a_layout():
    term, _, n = resnet_silu()
    kernel = LayoutAssignment(term, _args(n, False, "resnet_silu")).run()
    assert kernel.layout.term == term
