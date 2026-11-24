"""The layout search and terms with more than one consumer."""

import collections

import numpy as np

from frontends.tensor import TensorTerm
from ir.kernel import KernelOp
from lower.lower import Lower
from tests.test_conv_direct import _compile, _he_ops, _run

_MOVES = (KernelOp.TENSOR, KernelOp.CONVERSION, KernelOp.REPLICATE, KernelOp.COMPACT)


def _layouts_per_term(kernel):
    """The layouts each term is computed in."""
    layouts = collections.defaultdict(set)
    for k in kernel.post_order():
        if k.op not in _MOVES:
            layouts[k.layout.term].add(k.layout.layout_str())
    return layouts


def _gate(stride):
    """Features times a function of their own mean: the features have two
    consumers, the mean and the product."""
    x = TensorTerm.Tensor("x", [4, 8, 8], True)
    w1 = TensorTerm.Tensor("w1", [4, 4, 3, 3], False)
    w2 = TensorTerm.Tensor("w2", [4, 4, 1, 1], False)
    features = TensorTerm.conv2d(x, w1, stride, "same").hard_swish()
    gate = TensorTerm.conv2d(features.mean((1, 2)), w2, 1, "same")
    return features * gate


def test_two_consumers_read_one_layout_of_a_term():
    kernel = _compile(_gate(1), 1024)
    assert all(len(found) == 1 for found in _layouts_per_term(kernel).values())
    ops = [k.op for k in kernel.post_order()]
    # the 3x3 conv once, and the gate's 1x1 conv
    assert ops.count(KernelOp.CONV_DIRECT) == 2
    assert _he_ops(Lower(kernel).run()) == {"mul": 52, "rot": 39, "add": 63}


def test_a_strided_conv_with_two_consumers_is_computed_once():
    kernel = _compile(_gate(2), 1024)
    assert all(len(found) == 1 for found in _layouts_per_term(kernel).values())
    assert [k.op for k in kernel.post_order()].count(KernelOp.CONV_DIRECT) == 2


def test_the_gate_matches_eval():
    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.normal(size=(4, 8, 8)),
        "w1": rng.normal(size=(4, 4, 3, 3)),
        "w2": rng.normal(size=(4, 4, 1, 1)),
    }
    _run(_gate(1), inputs, 1024)
    _run(_gate(2), inputs, 1024)
