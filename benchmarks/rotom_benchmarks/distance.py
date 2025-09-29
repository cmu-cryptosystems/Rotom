import random

import numpy as np

from frontends.tensor import TensorTerm


def distance():
    inputs = {}
    inputs["point"] = np.array([random.choice(range(2)) for _ in range(64)])
    inputs["tests"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(128)]
    )

    point = TensorTerm.Tensor("point", [64], True)
    tests = TensorTerm.Tensor(f"tests", [128, 64], False)
    prod = tests * point
    comp = (prod - prod).sum(1)
    return comp, inputs
