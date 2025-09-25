import numpy as np
import random

from frontends.tensor import TensorTerm
from ir.dim import *
from assignment.assignment import LayoutAssignment
from lower.lower import Lower
from backends.toy import Toy
from util.layout_util import apply_layout

from tests.test_util import get_default_args


## Test matrix multiply with secret inputs
def index_matmul(index):
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 16], False)
    c = TensorTerm.Tensor("c", [4, 16], False)
    s = a @ b
    s2 = s.reshape(1, {1: 4, 2: 4}).permute({0: 1, 1: 2, 2: 0})
    t = a @ c
    t2 = t.reshape(1, {1: 4, 2: 4}).permute({0: 2, 1: 1, 2: 0})

    res = []
    for i in range(4):
        res.append(s2[i] @ t2[i])
    return s2[index] @ t2[index]


def test_index_matmul_1():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "index_matmul_1"
    args.rolls = True 

    # create inputs
    inputs = {}
    inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
    inputs["b"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])
    inputs["c"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])

    # generate test case
    tensor_ir = index_matmul(0)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected = tensor_ir.eval(inputs)
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_index_matmul_2():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "index_matmul_2"
    args.rolls = True 

    # create inputs
    inputs = {}
    inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
    inputs["b"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])
    inputs["c"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])

    # generate test case
    tensor_ir = index_matmul(1)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected = tensor_ir.eval(inputs)
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def index_matmul_2(index):
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 16], False)
    b = b.reshape(1, {1: 4, 2: 4})
    return a @ b[index]


def test_index_matmul_3():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "index_matmul_3"
    args.rolls = True 

    # create inputs
    inputs = {}
    inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
    inputs["b"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])
    inputs["c"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])

    # generate test case
    tensor_ir = index_matmul_2(0)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected = tensor_ir.eval(inputs)
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_index_matmul_4():
    # create args
    args = get_default_args()
    args.n = 16
    args.benchmark = "index_matmul_4"
    args.rolls = True 

    # create inputs
    inputs = {}
    inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
    inputs["b"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])
    inputs["c"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])

    # generate test case
    tensor_ir = index_matmul_2(1)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected = tensor_ir.eval(inputs)
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def index_matmul_3(index):
    h = TensorTerm.Tensor("h", [4, 48], True)
    wq = TensorTerm.Tensor("wq", [48, 48], False)
    bq = TensorTerm.Tensor("bq", [48], False)
    wk = TensorTerm.Tensor("wk", [48, 48], False)
    bk = TensorTerm.Tensor("bk", [48], False)

    q = h @ wq + bq
    k = h @ wk + bk

    # reshape q and k
    # [4, 48] -> [4, 3, 16]
    blocked_q = q.reshape(1, {1: 3, 2: 16}).permute({0: 1, 1: 2, 2: 0})
    blocked_k = k.reshape(1, {1: 3, 2: 16}).permute({0: 2, 1: 1, 2: 0})

    # q @ k.t
    res = []
    for i in range(16):
        res.append(blocked_q[i] @ blocked_k[i])
    return res[index]



# def test_index_matmul_4():
#     # create args
#     args = get_default_args()
#     args.benchmark = "index_matmul_4"
#     args.rolls = True 

#     # create inputs
#     size = 48
#     inputs = {}
#     inputs["h"] = np.array(
#         [[random.choice(range(2)) for _ in range(size)] for _ in range(4)]
#     )
#     inputs["wq"] = np.array(
#         [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
#     )
#     inputs["bq"] = np.array([random.choice(range(2)) for _ in range(size)])
#     inputs["wk"] = np.array(
#         [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
#     )
#     inputs["bk"] = np.array([random.choice(range(2)) for _ in range(size)])

#     # generate test case
#     i = random.randint(0, 15)
#     tensor_ir = index_matmul_3(i)

#     # run compiler
#     kernel = LayoutAssignment(tensor_ir, args).run()
#     circuit_ir = Lower(kernel).run()
#     results = Toy(circuit_ir, inputs, args).run()

#     # check result
#     expected = tensor_ir.eval(inputs)
#     expected_cts = apply_layout(expected, kernel.layout)
#     assert expected_cts == results


# def test_index_matmul_5():
#     # create args
#     args = get_default_args()
#     args.benchmark = "index_matmul_5"
#     args.rolls = True 

#     # create inputs
#     size = 48
#     inputs = {}
#     inputs["h"] = np.array(
#         [[random.choice(range(2)) for _ in range(size)] for _ in range(4)]
#     )
#     inputs["wq"] = np.array(
#         [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
#     )
#     inputs["bq"] = np.array([random.choice(range(2)) for _ in range(size)])
#     inputs["wk"] = np.array(
#         [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
#     )
#     inputs["bk"] = np.array([random.choice(range(2)) for _ in range(size)])

#     # generate test case
#     i = random.randint(0, 15)
#     tensor_ir = index_matmul_3(i)

#     # run compiler
#     kernel = LayoutAssignment(tensor_ir, args).run()
#     circuit_ir = Lower(kernel).run()
#     results = Toy(circuit_ir, inputs, args).run()

#     # check result
#     expected = tensor_ir.eval(inputs)
#     expected_cts = apply_layout(expected, kernel.layout)
#     assert expected_cts == results