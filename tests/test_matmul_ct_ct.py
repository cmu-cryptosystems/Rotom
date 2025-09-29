import random

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


# Test matrix multiply with secret inputs
def matmul_ct_ct(inputs):
    a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
    b = TensorTerm.Tensor("b", list(inputs["b"].shape), True)
    return a @ b, inputs["a"] @ inputs["b"]


def test_matmul_ct_ct_1():
    # create args# create args
    args = get_default_args()
    args.n = 16

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
    tensor_ir, expected = matmul_ct_ct(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_ct_2():
    # create args# create args
    args = get_default_args()
    args.n = 16

    # create inputs
    size = 8
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )

    # generate test case
    tensor_ir, expected = matmul_ct_ct(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_ct_3():
    # create args# create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "matmul_ct_ct_3"

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
    tensor_ir, expected = matmul_ct_ct(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_ct_4():
    # create args# create args
    args = get_default_args()
    args.rolls = True
    args.benchmark = "matmul_ct_ct_4"

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
    tensor_ir, expected = matmul_ct_ct(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_ct_5():
    # create args
    args = get_default_args()
    args.n = 16
    args.rolls = True

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
    tensor_ir, expected = matmul_ct_ct(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_ct_6():
    # create args# create args
    args = get_default_args()
    args.n = 16
    args.rolls = True

    # create inputs
    size = 8
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for j in range(size)] for i in range(size)]
    )

    # generate test case
    tensor_ir, expected = matmul_ct_ct(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_ct_7():
    # create args# create args# create args
    args = get_default_args()
    args.n = 16
    args.rolls = True
    args.benchmark = "matmul_ct_ct_7"

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
    tensor_ir, expected = matmul_ct_ct(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    for k in kernel.post_order():
        print(k)
    print()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_matmul_ct_ct_8():
    # create args
    args = get_default_args()
    args.rolls = True
    args.benchmark = "matmul_ct_ct_8"

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
    tensor_ir, expected = matmul_ct_ct(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results
