from frontends.tensor import TensorTerm
import numpy as np
import random


def matmul_256_128():
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for _ in range(128)] for _ in range(256)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for _ in range(256)] for _ in range(128)]
    )
    a = TensorTerm.Tensor("a", inputs["a"].shape, True)
    b = TensorTerm.Tensor("b", inputs["b"].shape, False)
    return a @ b, inputs
