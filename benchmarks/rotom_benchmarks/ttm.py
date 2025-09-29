import random

import numpy as np

from frontends.tensor import TensorTerm


def ttm():
    inputs = {}
    inputs["t"] = np.array(
        [
            [[random.choice(range(2)) for _ in range(64)] for _ in range(64)]
            for _ in range(64)
        ]
    )
    inputs["m"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(64)]
    )

    t = TensorTerm.Tensor("t", inputs["t"].shape, True)
    m = TensorTerm.Tensor("m", inputs["m"].shape, True)
    return t @ m, inputs
