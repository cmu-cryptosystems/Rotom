import numpy as np

from frontends.tensor import TensorTerm
from ir.dim import *
from assignment.assignment import LayoutAssignment
from lower.lower import Lower
from backends.toy import Toy
from util.layout_util import apply_layout
from tests.test_util import get_default_args


## Test matrix multiply with one secret input and one public input
def matvecmul_ct_pt(inputs):
    a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
    b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
    return a @ b, inputs["a"] @ inputs["b"]


def test_matvecmul_ct_pt_1():
    # create args
    args = get_default_args()
    args.n = 16
    args.rolls = True

    # create inputs
    size = 4
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["b"] = np.array([i for i in range(size)])

    print(inputs["a"].shape)
    print(inputs["b"].shape)

    # generate test case
    tensor_ir, expected = matvecmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matvecmul_ct_pt_2():
    # create args
    args = get_default_args()
    args.n = 4096
    args.rolls = True   

    # create inputs
    size = 64
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["b"] = np.array([i for i in range(size)])

    print(inputs["a"].shape)
    print(inputs["b"].shape)

    # generate test case
    tensor_ir, expected = matvecmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results
