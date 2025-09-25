import numpy as np

from frontends.tensor import TensorTerm
from ir.dim import *
from assignment.assignment import LayoutAssignment
from lower.lower import Lower
from backends.toy import Toy
from util.layout_util import apply_layout
from tests.test_util import get_default_args


def convolution(input_size, input_channel, f_out, f_in, f_h, f_w, stride, padding):
    """convolution example, precursor to resnet
    https://proceedings.mlr.press/v162/lee22e.html
    """
    input_Tensor = TensorTerm.Tensor(
        "a", [input_channel, input_size, input_size], True)
    weight_Tensor = TensorTerm.Tensor("b", [f_out, f_in, f_h, f_w], False)
    output_Tensor = TensorTerm.conv2d(
        input_Tensor, weight_Tensor, stride, padding)
    return output_Tensor


def test_conv2d_1():
    # create args
    args = get_default_args()
    args.n = 16
    args.rolls = True
    args.benchmark = "conv2d_1"

    # create inputs
    dim_size = 4
    f_size = 2
    padding = "same"
    inputs = {}
    inputs["a"] = np.array(
        [
            [[i + j * dim_size for i in range(dim_size)]
             for j in range(dim_size)]
            for _ in range(1)
        ]
    )
    inputs["b"] = np.array([[[[1 for i in range(f_size)]
                           for j in range(f_size)]]])

    # generate test case
    tensor_ir = convolution(dim_size, 1, 1, 1, f_size, f_size, 1, padding)
    expected = tensor_ir.eval(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_conv2d_2():
    # create args
    args = get_default_args()
    args.n = 16
    args.rolls = True
    args.benchmark = "conv2d_2"

    # create inputs
    dim_size = 4
    f_size = 3
    padding = "same"
    inputs = {}
    inputs["a"] = np.array(
        [
            [[i + j * dim_size for i in range(dim_size)]
             for j in range(dim_size)]
            for _ in range(1)
        ]
    )
    inputs["b"] = np.array([[[[1 for i in range(f_size)]
                           for j in range(f_size)]]])

    # generate test case
    tensor_ir = convolution(dim_size, 1, 1, 1, f_size, f_size, 1, padding)
    expected = tensor_ir.eval(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_conv2d_3():
    # create args
    args = get_default_args()
    args.n = 64
    args.rolls = True
    args.benchmark = "conv2d_3"

    # create inputs
    input_channels = 3
    dim_size = 4
    f_size = 3
    padding = "same"
    inputs = {}
    inputs["a"] = np.array(
        [
            [[i + j * dim_size for i in range(dim_size)]
             for j in range(dim_size)]
            for _ in range(input_channels)
        ]
    )
    inputs["b"] = np.array([[[[1 for i in range(f_size)]
                           for j in range(f_size)]]])

    # generate test case
    tensor_ir = convolution(dim_size, input_channels, 1,
                            1, f_size, f_size, 1, padding)
    expected = tensor_ir.eval(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results


def test_conv2d_4():
    # create args
    args = get_default_args()
    args.n = 4096
    args.rolls = True
    args.benchmark = "conv2d_4"

    # create inputs
    input_channels = 3
    dim_size = 32
    f_size = 3
    padding = "same"
    inputs = {}
    inputs["a"] = np.array(
        [
            [[i + j * dim_size for i in range(dim_size)]
             for j in range(dim_size)]
            for _ in range(input_channels)
        ]
    )
    inputs["b"] = np.array([[[[1 for i in range(f_size)]
                           for j in range(f_size)]]])

    # generate test case
    tensor_ir = convolution(dim_size, input_channels, 1,
                            1, f_size, f_size, 1, padding)
    expected = tensor_ir.eval(inputs)

    # run compiler
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    # check result
    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results
