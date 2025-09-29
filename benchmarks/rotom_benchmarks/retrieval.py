import random

import numpy as np

from frontends.tensor import TensorTerm


def retrieval():
    inputs = {}
    inputs["keys"] = np.array(
        [[random.choice(range(2)) for _ in range(10)] for _ in range(1024)]
    )
    inputs["values"] = np.array([random.choice(range(2)) for _ in range(1024)])
    inputs["query"] = np.array([random.choice(range(2)) for _ in range(10)])

    keys = TensorTerm.Tensor("point", [1024, 10], True)
    values = TensorTerm.Tensor(f"tests", [1024], True)
    query = TensorTerm.Tensor(f"tests", [10], True)

    diff = TensorTerm.const(1) - ((keys - query) * (keys - query))
    diff = diff.product(1)
    return diff, inputs
