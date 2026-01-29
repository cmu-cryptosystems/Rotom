import numpy as np

from frontends.tensor import TensorTerm


def logreg():
    inputs = {}
    inputs["v"] = np.array([[np.random.randint(2)] for _ in range(256)])
    inputs["m_0"] = np.array(
        [[np.random.randint(2) for _ in range(256)] for _ in range(1024)]
    )
    inputs["m_1"] = np.array(
        [[np.random.randint(2) for _ in range(1024)] for _ in range(256)]
    )

    v = TensorTerm.Tensor("v", [256, 1], True)
    m = TensorTerm.Tensor(f"m_0", [1024, 256], False)
    m1 = TensorTerm.Tensor(f"m_1", [256, 1024], False)
    v2 = m @ v
    tensor_ir = m1 @ v2
    return tensor_ir, inputs
    # return v2, inputs
