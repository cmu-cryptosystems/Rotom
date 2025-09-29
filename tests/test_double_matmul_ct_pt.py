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


def double_matmul_ct_pt(inputs):
    a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
    b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
    c = TensorTerm.Tensor("c", list(inputs["c"].shape), False)
    return a @ b @ c


def test_double_matmul_ct_pt_1():
    # create args
    args = get_default_args()
    args.n = 16
    args.rolls = True
    args.benchmark = "double_matmul_ct_pt_1"

    # create inputs
    size = 4
    inputs = {}
    inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["b"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
    inputs["c"] = np.array([[i * size + j for j in range(size)] for i in range(size)])

    # generate test case
    tensor_ir = double_matmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(tensor_ir.eval(inputs), kernel.layout)
    assert expected_cts == results


def test_double_matmul_ct_pt_2():
    # create args
    args = get_default_args()
    args.n = 16
    args.rolls = False
    args.benchmark = "double_matmul_ct_pt_2"

    # create inputs
    size = 16
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for i in range(size)] for j in range(size)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for i in range(size)] for j in range(size)]
    )
    inputs["c"] = np.array(
        [[random.choice(range(2)) for i in range(size)] for j in range(size)]
    )

    # generate test case
    tensor_ir = double_matmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    print("kernel:", kernel)
    for k in kernel.post_order():
        print("k:", k)
    print()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(tensor_ir.eval(inputs), kernel.layout)
    assert expected_cts == results


def test_double_matmul_ct_pt_3():
    # create args
    args = get_default_args()
    args.rolls = True
    args.benchmark = "double_matmul_ct_pt_3"

    # create inputs
    size = 64
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
    )
    inputs["c"] = np.array(
        [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
    )

    # generate test case
    tensor_ir = double_matmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(tensor_ir.eval(inputs), kernel.layout)
    assert expected_cts == results


def test_double_matmul_ct_pt_4():
    # create args
    args = get_default_args()
    args.rolls = True
    args.benchmark = "double_matmul_ct_pt_4"

    # create inputs
    size = 64
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
    )
    inputs["c"] = np.array(
        [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
    )

    # generate test case
    tensor_ir = double_matmul_ct_pt(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(tensor_ir.eval(inputs), kernel.layout)
    assert expected_cts == results
