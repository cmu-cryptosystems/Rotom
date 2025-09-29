import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


## Test matrix multiply with secret inputs
def matmul_add(inputs):
    a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
    b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
    c = TensorTerm.Tensor("c", list(inputs["c"].shape), False)
    return (a @ b) + c, (inputs["a"] @ inputs["b"]) + inputs["c"]


def test_matmul_add_1():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "matmul_add_1"

    # create inputs
    size = 4
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["b"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["c"] = np.array([j for j in range(size)])

    # generate test case
    tensor_ir, expected = matmul_add(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_add_2():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "matmul_add_2"
    args.rolls = True

    # create inputs
    size = 4
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["b"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["c"] = np.array([j for j in range(size)])

    # generate test case
    tensor_ir, expected = matmul_add(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_add_3():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "matmul_add_3"

    # create inputs
    size = 8
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["b"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["c"] = np.array([j for j in range(size)])

    # generate test case
    tensor_ir, expected = matmul_add(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_add_4():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "matmul_add_4"
    args.rolls = True

    # create inputs
    size = 8
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["b"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["c"] = np.array([j for j in range(size)])

    # generate test case
    tensor_ir, expected = matmul_add(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results
