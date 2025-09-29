import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


## Test index
def index(inputs, i):
    a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
    return a[i]


def test_index_1():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "index_1"

    # create inputs
    size = 4
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])

    # generate test case
    tensor_ir = index(inputs, 0)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected = tensor_ir.eval(inputs)
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_index_2():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "index_2"

    # create inputs
    size = 4
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])

    # generate test case
    tensor_ir = index(inputs, 0)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected = tensor_ir.eval(inputs)
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_index_3():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "index_3"

    # create inputs
    size = 4
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])

    # generate test case
    tensor_ir = index(inputs, 0)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected = tensor_ir.eval(inputs)
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_index_4():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "index_4"

    # create inputs
    size = 4
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])

    # generate test case
    tensor_ir = index(inputs, 0)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected = tensor_ir.eval(inputs)
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def index_2():
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 32], False)
    s = a @ b
    s2 = s.reshape(1, {1: 4, 2: 8}).permute({0: 1, 1: 0, 2: 2})
    return s2[0]


def test_index_5():
    # create args
    args = get_default_args()
    args.n = 32
    args.benchmark = "index_5"

    # create inputs
    inputs = {}
    inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
    inputs["b"] = np.array([[i * 4 + j for j in range(32)] for i in range(4)])
    # generate test case
    tensor_ir = index_2()

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected = tensor_ir.eval(inputs)
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results
