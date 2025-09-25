from frontends.tensor import TensorTerm
import numpy as np
import random


def logreg():
    inputs = {}
    inputs["v"] = np.array([random.choice(range(2)) for _ in range(197)])
    inputs["m_0"] = np.array(
        [[random.choice(range(2)) for _ in range(197)] for _ in range(1024)]
    )
    inputs["m_1"] = np.array(
        [[random.choice(range(2)) for _ in range(1024)] for _ in range(197)]
    )

    v = TensorTerm.Tensor("v", [197], True)
    m = TensorTerm.Tensor(f"m_0", [1024, 197], False)
    m1 = TensorTerm.Tensor(f"m_1", [197, 1024], False)
    v2 = m @ v
    tensor_ir = m1 @ v2
    return tensor_ir, inputs
