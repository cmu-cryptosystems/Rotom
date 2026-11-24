"""Convolutions whose channel counts are not powers of two.

A ciphertext dim holds exactly as many indices as the tensor has (``[c:12:1]``
is 12 ciphertexts); in slots the dim takes the next power of two. Compaction
has to move such a dim into gaps, and its lowering has to follow.
"""

from copy import deepcopy

import numpy as np
import pytest

from assignment.assignment import LayoutAssignment
from assignment.gen.gen_align import _conversions_like, _holds_a_dim_exactly
from assignment.gen.gen_compaction import find_compaction
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from lower.lower import Lower
from tests.test_conv_direct import _he_ops, _run
from tests.test_util import get_default_args
from util.checker import check_results
from util.layout_util import parse_layout
from util.window import masked_groups, relayout_ops


@pytest.mark.parametrize(
    "layout,n,compacted",
    [
        # the gap holds every index: the dim takes the next power of two
        ("[0:6:1];[G:8][1:8:1][2:8:1]", 512, "[0:8:1][1:8:1][2:8:1]"),
        # the gap is smaller: it takes the fine digit and 24 indices stay on
        # ceil(24 / 16) ciphertexts
        ("[0:24:1];[G:16][1:4:1][2:4:1]", 256, "[0:2:16];[0:16:1][1:4:1][2:4:1]"),
        (
            "[0:6:1];[1:4:1][G:2][2:4:1][G:2]",
            64,
            "[0:2:4];[1:4:1][0:2:1][2:4:1][0:2:2]",
        ),
        # a power of two keeps the coarse digit in the gap, as before
        ("[0:8:1];[G:4][1:4:1][2:4:1]", 64, "[0:2:1];[0:4:2][1:4:1][2:4:1]"),
    ],
)
def test_compaction_of_a_ciphertext_dim(layout, n, compacted):
    source = parse_layout(layout, n, True)
    kernel = find_compaction(Kernel(KernelOp.TENSOR, [], source))
    assert kernel.op == KernelOp.COMPACT
    assert kernel.layout.layout_str() == compacted

    # Ciphertexts move whole into zero gaps: a rotation and an add each, and
    # a mask only where a ciphertext would be split.
    ops = relayout_ops(source, kernel.layout, zero_gaps=True)
    moved = source.num_ct() - kernel.layout.num_ct()
    assert ops == {"mul": 0, "rot": moved, "add": moved}
    assert KernelCost(kernel, "lan").ops() == ops
    assert not masked_groups(source, kernel.layout, True).any()
    assert masked_groups(source, kernel.layout, False).all()


def _conv(c_in, c_out, size, k, stride):
    rng = np.random.default_rng(0)
    a = TensorTerm.Tensor("a", [c_in, size, size], True)
    b = TensorTerm.Tensor("b", [c_out, c_in, k, k], False)
    inputs = {
        "a": rng.normal(size=(c_in, size, size)),
        "b": rng.normal(size=(c_out, c_in, k, k)),
    }
    return TensorTerm.conv2d(a, b, stride, "same"), inputs


@pytest.mark.parametrize(
    "c_in,c_out,size,k,stride,n",
    [
        (8, 24, 8, 1, 1, 256),
        (6, 12, 8, 3, 1, 1024),
        (24, 6, 8, 1, 1, 512),
        (12, 6, 8, 3, 2, 1024),
        (4, 12, 8, 3, 2, 512),
        (5, 3, 8, 3, 1, 512),
        (4, 6, 8, 1, 2, 64),
    ],
)
def test_every_plan_computes_the_convolution(c_in, c_out, size, k, stride, n):
    """Every plan the search keeps, compacted ones included, not only the
    cheapest."""
    term, inputs = _conv(c_in, c_out, size, k, stride)
    args = get_default_args()
    args.n = n
    args.backend = "toy"
    search = LayoutAssignment(term, args)
    search.run()

    plans = [
        search.combine_kernels(deepcopy(plan)) for plan in search.kernels[term].values()
    ]
    assert any(kernel.op == KernelOp.COMPACT for kernel in plans)
    for kernel in plans:
        circuit = Lower(kernel).run()
        results = Toy(circuit, inputs, args).run()
        check_results(term, inputs, kernel, results, 0, args)
        if kernel.op == KernelOp.COMPACT:
            lowered = Lower(kernel)
            lowered.lower()
            # the circuit is the convolution's, plus the compaction's
            # rotations and adds: no multiply
            compact = KernelCost(kernel, "lan").ops()
            conv = KernelCost(kernel.cs[0], "lan").ops()
            assert compact["mul"] == 0
            assert _he_ops({0: lowered.env[kernel]}) == {
                op: conv[op] + compact[op] for op in conv
            }


@pytest.mark.parametrize("channels,n", [(12, 256), (24, 1024), (40, 1024), (40, 4096)])
def test_every_plan_of_the_sum_of_two_convs(channels, n):
    """Each conv has layouts that hold its channels exactly on ciphertexts and
    layouts padded to a power of two in slots. Operands that hold them
    differently are added after one is converted to the other's layout."""
    rng = np.random.default_rng(channels)
    x = TensorTerm.Tensor("x", [8, 4, 4], True)
    w1 = TensorTerm.Tensor("w1", [channels, 8, 1, 1], False)
    w2 = TensorTerm.Tensor("w2", [channels, 8, 3, 3], False)
    inputs = {
        "x": rng.normal(size=(8, 4, 4)),
        "w1": rng.normal(size=(channels, 8, 1, 1)),
        "w2": rng.normal(size=(channels, 8, 3, 3)),
    }
    term = TensorTerm.conv2d(x, w1, 1, "same") + TensorTerm.conv2d(x, w2, 1, "same")
    args = get_default_args()
    args.n = n
    args.backend = "toy"
    search = LayoutAssignment(term, args)
    search.run()

    plans = [
        search.combine_kernels(deepcopy(plan)) for plan in search.kernels[term].values()
    ]
    for kernel in plans:
        results = Toy(Lower(kernel).run(), inputs, args).run()
        check_results(term, inputs, kernel, results, 0, args)


@pytest.mark.parametrize("channels,n", [(12, 256), (40, 1024)])
def test_a_conv_output_converted_to_each_of_its_other_layouts(channels, n):
    """The conversion the sum above is offered for operands that hold the
    channels differently. The search never keeps it there (two operands in
    one layout are cheaper), so it is run here on its own."""
    term, inputs = _conv(8, channels, 4, 3, 1)
    args = get_default_args()
    args.n = n
    args.backend = "toy"
    search = LayoutAssignment(term, args)
    search.run()
    plans = [
        search.combine_kernels(deepcopy(plan)) for plan in search.kernels[term].values()
    ]

    same_dims = {(0, 0), (1, 1), (2, 2)}
    converted = 0
    for source in plans:
        for target in plans:
            if source.layout == target.layout:
                continue
            if not (_holds_a_dim_exactly(source) or _holds_a_dim_exactly(target)):
                continue
            (_, _), (conversion, _) = _conversions_like(
                same_dims,
                Kernel(KernelOp.CS, [0], source.layout),
                Kernel(KernelOp.CS, [1], target.layout),
            )
            kernel = Kernel(
                KernelOp.CONVERSION,
                [conversion.cs[0], conversion.cs[1], source],
                conversion.layout,
            )
            assert kernel.layout.get_dims() == target.layout.get_dims()
            results = Toy(Lower(kernel).run(), inputs, args).run()
            check_results(term, inputs, kernel, results, 0, args)
            converted += 1
    assert converted


def test_a_public_operand_on_conv3d_channels_held_exactly_on_ciphertexts():
    """80 channels on 5 ciphertexts of 16: the bias is padded to 128 on its
    own, and is laid out like the conv output instead."""
    rng = np.random.default_rng(6)
    x = TensorTerm.Tensor("x", [8, 2, 4, 4], True, "[0:8:1][1:2:1][2:4:1][3:4:1]")
    w = TensorTerm.Tensor("w", [80, 8, 1, 1, 1], False)
    inputs = {
        "x": rng.normal(size=(8, 2, 4, 4)),
        "w": rng.normal(size=(80, 8, 1, 1, 1)),
    }
    conv = TensorTerm.conv3d(x, w, 1, "same")
    for name, shape in (("bias", (80, 1, 1, 1)), ("full", (80, 2, 4, 4))):
        inputs[name] = rng.normal(size=shape)
        public = TensorTerm.Tensor(name, list(shape), False)
        _run(conv + public, inputs, 4096)
        _run(conv * public, inputs, 4096)
