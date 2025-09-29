import random

import numpy as np

from frontends.tensor import TensorTerm


def matmul_128_128():
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for _ in range(128)] for _ in range(128)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for _ in range(128)] for _ in range(128)]
    )
    a = TensorTerm.Tensor("a", inputs["a"].shape, True)
    b = TensorTerm.Tensor("b", inputs["b"].shape, True)
    return a @ b, inputs
