"""A kernel's string, and the copies of dimensions, that the layout search
makes hundreds of thousands of times."""

from copy import deepcopy

import pytest

from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging

LAYOUTS = [
    "[0:4:1][1:4:1]",
    "[0:2:2][0:2:1][1:4:1]",
    "[0:4:1];[1:4:1][G:2][G:2]",
    "[1:2:1][G:2];[0:2:2][0:2:1][1:2:2][R:2:1]",
    "roll(0,1) [0:4:1][1:4:1]",
]


@pytest.mark.parametrize("layout_str", LAYOUTS)
@pytest.mark.parametrize("n", [4, 16, 64])
def test_a_kernel_prints_its_layout_with_merged_dims(layout_str, n):
    layout = Layout.from_string(layout_str, n, True)
    for _ in range(2):  # the second time from the cache
        kernel = Kernel(KernelOp.ADD, [], layout)
        assert (
            repr(kernel) == f"{KernelOp.ADD}: {dimension_merging(layout).layout_str()}"
        )


def test_the_same_pieces_under_another_n_print_differently():
    """`n` decides which pieces are ciphertext dims, so it is part of the key."""
    pieces = "[0:4:1][1:4:1]"
    slots = Kernel(KernelOp.ADD, [], Layout.from_string(pieces, 16, True))
    cts = Kernel(KernelOp.ADD, [], Layout.from_string(pieces, 4, True))
    assert repr(slots) == f"{KernelOp.ADD}: [0:4:1][1:4:1]"
    assert repr(cts) == f"{KernelOp.ADD}: [0:4:1];[1:4:1]"


def test_a_copied_dim_is_equal_and_separate():
    for dim in (Dim(1, 4, 2), Dim(None, 8, 1, DimType.EMPTY), Dim(None, 2, 4)):
        copied = deepcopy(dim)
        assert copied == dim and copied is not dim
        assert vars(copied) == vars(dim)
        copied.extent *= 2
        assert copied != dim


def test_copying_a_list_keeps_a_dim_that_appears_twice_as_one():
    dim = Dim(0, 4, 1)
    first, second = deepcopy([dim, dim])
    assert first is second and first is not dim
