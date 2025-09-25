from frontends.tensor import TensorTerm
import numpy as np
import random


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
    k = (h @ wk) + bk
    v = (h @ wv) + bv

    # reshape q and k
    blocked_q = q.reshape(1, {1: 12, 2: 64}).permute({0: 1, 1: 0, 2: 2})
    blocked_kt = k.reshape(1, {1: 12, 2: 64}).permute({0: 2, 1: 0, 2: 1})
    blocked_v = v.reshape(1, {1: 12, 2: 64}).permute({0: 1, 1: 0, 2: 2})

    q_kt = blocked_q.block_matmul(blocked_kt)

    q_kt_v = q_kt.block_matmul(blocked_v)
    return q_kt_v, inputs
