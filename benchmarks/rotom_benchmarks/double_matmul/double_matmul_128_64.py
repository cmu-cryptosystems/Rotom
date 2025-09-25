from frontends.tensor import TensorTerm
import numpy as np
import random


def double_matmul_128_64():
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(128)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for _ in range(128)] for _ in range(64)]
    )
    inputs["c"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(128)]
    )

    
    a = TensorTerm.Tensor("a", inputs["a"].shape, True)
    b = TensorTerm.Tensor("b", inputs["b"].shape, True)
    c = TensorTerm.Tensor("c", inputs["c"].shape, True)
    return a @ b @ c, inputs
