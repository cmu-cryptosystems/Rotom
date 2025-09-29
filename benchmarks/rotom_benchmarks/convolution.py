from frontends.tensor import TensorTerm
import numpy as np
import random


def run_convolution(input_size, input_channel, f_out, f_in, f_h, f_w, stride, padding):
    input_tensor = TensorTerm.Tensor("a", [input_channel, input_size, input_size], True)
    weight_tensor = TensorTerm.Tensor("b", [f_out, f_in, f_h, f_w], False)
    output_tensor = TensorTerm.conv2d(input_tensor, weight_tensor, stride, padding)
    return output_tensor


def convolution():
    input_channels = 8
    dim_size = 32
    f_size = 3
    padding = "same"

    inputs = {}
    inputs["a"] = np.array(
        [
            [[i + j * dim_size for i in range(dim_size)] for j in range(dim_size)]
            for _ in range(input_channels)
        ]
    )
    inputs["b"] = np.array([[[[1 for i in range(f_size)] for j in range(f_size)]]])

    tensor_ir = run_convolution(
        dim_size, input_channels, 1, 1, f_size, f_size, 1, padding
    )
    return tensor_ir, inputs, 8192 
