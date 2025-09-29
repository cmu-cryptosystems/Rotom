import random

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


# Test matrix multiply with one secret input and one public input
def matmul_ct_pt(inputs):
    a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
    b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
    return a @ b, inputs["a"] @ inputs["b"]


def test_matmul_ct_pt_1():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "matmul_ct_pt_1"
    # args.fuzz = True

    # create inputs
    size = 4
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )

    # generate test case
    tensor_ir, expected = matmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_pt_2():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "matmul_ct_pt_2"
    # args.fuzz = True

    # create inputs
    size = 16
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )

    # generate test case
    tensor_ir, expected = matmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_pt_3():
    # create args # create args
    args = get_default_args()
    args.n = 4096
    args.benchmark = "matmul_ct_pt_3"

    # create inputs
    size = 64
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )

    # generate test case
    tensor_ir, expected = matmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_pt_4():
    # create args
    args = get_default_args()
    args.n = 4096
    args.benchmark = "matmul_ct_pt_4"

    # create inputs
    size = 64
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )

    # generate test case
    tensor_ir, expected = matmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_pt_5():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "matmul_ct_pt_5"
    # args.fuzz = True

    # create inputs
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for j in range(4)] for i in range(4)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for j in range(16)] for i in range(4)]
    )

    # generate test case
    tensor_ir, expected = matmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results
