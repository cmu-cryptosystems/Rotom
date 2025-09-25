from frontends.tensor import TensorTerm
import numpy as np
import random


def matmul_128_64_ct_ct():
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(5)) for _ in range(64)] for _ in range(128)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for _ in range(128)] for _ in range(64)]
    )
    a = TensorTerm.Tensor("a", inputs["a"].shape, True)
    b = TensorTerm.Tensor("b", inputs["b"].shape, True)
    return a @ b, inputs
