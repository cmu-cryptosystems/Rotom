import numpy as np

from frontends.tensor import TensorTerm
from ir.dim import *
from assignment.assignment import LayoutAssignment
from lower.lower import Lower
from backends.toy import Toy
from util.layout_util import apply_layout
from tests.test_util import get_default_args


def reshape():
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 32], False)
    s = a @ b
    s2 = s.reshape(1, {1: 4, 2: 8}).permute({0: 1, 1: 0, 2: 2})
    return s2


def test_reshape():
    # create args
    args = get_default_args()
    args.n = 32

    # create inputs
    inputs = {}
    inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
    inputs["b"] = np.array([[i * 4 + j for j in range(32)] for i in range(4)])
    # generate test case
    tensor_ir = reshape()

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected = tensor_ir.eval(inputs)
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results
