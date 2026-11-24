"""A deep copy of a kernel or a layout shares its TensorTerm."""

from copy import deepcopy

from frontends.tensor import TensorTerm
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from opt.opt import Optimizer


def _chain(depth):
    x = TensorTerm.Tensor("x", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 4], False)
    for _ in range(depth):
        x = x + b
    return x


def test_a_copied_layout_shares_its_term_and_copies_its_dims():
    term = _chain(3)
    layout = Layout.from_string("[0:4:1][1:4:1]", 16, True)
    layout.term = term
    copied = deepcopy(layout)
    assert copied.term is term
    assert copied.get_dims() == layout.get_dims()
    assert copied.get_dims()[0] is not layout.get_dims()[0]


def test_a_copied_term_is_the_term():
    term = _chain(3)
    assert deepcopy(term) is term
    assert deepcopy([term, term])[0] is term


def test_the_optimizer_copies_a_kernel_of_a_deep_program():
    """The roll passes deep-copy every candidate kernel. Copying the term of a
    kernel 400 operations into a program used to exceed the recursion limit."""
    term = _chain(400)
    layout = Layout(term, [], Layout.from_string("[0:4:1][1:4:1]", 16).get_dims(), 16)
    operand = Kernel(KernelOp.CS, [0], Layout(term.cs[0], [], layout.get_dims(), 16))
    kernel = Kernel(KernelOp.ADD, [operand, operand], layout)
    (optimized,) = Optimizer(True).run([kernel])
    assert optimized.layout.term is term
    assert optimized.layout.get_dims() == layout.get_dims()
