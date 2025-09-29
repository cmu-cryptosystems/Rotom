import random

import numpy as np

from frontends.tensor import TensorTerm


def bert_attention():
    inputs = {}
    inputs["h"] = np.array(
        [[random.choice(range(2)) for _ in range(768)] for _ in range(128)]
    )
    inputs["wq"] = np.array(
        [[random.choice(range(2)) for _ in range(768)] for _ in range(768)]
    )
    inputs["bq"] = np.array([random.choice(range(2)) for _ in range(768)])
    inputs["wk"] = np.array(
        [[random.choice(range(2)) for _ in range(768)] for _ in range(768)]
    )
    inputs["bk"] = np.array([random.choice(range(2)) for _ in range(768)])
    inputs["wv"] = np.array(
        [[random.choice(range(2)) for _ in range(768)] for _ in range(768)]
    )
    inputs["bv"] = np.array([random.choice(range(2)) for _ in range(768)])

    h = TensorTerm.Tensor("h", [128, 768], True)
    wq = TensorTerm.Tensor("wq", [768, 768], False)
    bq = TensorTerm.Tensor("bq", [768], False)
    wk = TensorTerm.Tensor("wk", [768, 768], False)
    bk = TensorTerm.Tensor("bk", [768], False)
    wv = TensorTerm.Tensor("wv", [768, 768], False)
    bv = TensorTerm.Tensor("bv", [768], False)

    q = h @ wq + bq
    k = h @ wk + bk
    qk = q @ k.T
    # qk = softmax(qk)
    v = h @ wv + bv
    result = qk @ v
    return result, inputs, 8192
