from frontends.tensor import TensorTerm
import numpy as np
import random

def strassens():
    inputs = {}
    inputs["a1"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(64)]
    )
    inputs["a2"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(64)]
    )
    inputs["a3"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(64)]
    )
    inputs["a4"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(64)]
    )
    inputs["b1"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(64)]
    )
    inputs["b2"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(64)]
    )
    inputs["b3"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(64)]
    )
    inputs["b4"] = np.array(
        [[random.choice(range(2)) for _ in range(64)] for _ in range(64)]
    )
    
    a1 = TensorTerm.Tensor("a1", inputs["a1"].shape, True)
    a2 = TensorTerm.Tensor("a2", inputs["a2"].shape, True)
    a3 = TensorTerm.Tensor("a3", inputs["a3"].shape, True)
    a4 = TensorTerm.Tensor("a4", inputs["a4"].shape, True)

    b1 = TensorTerm.Tensor("b1", inputs["b1"].shape, True)
    b2 = TensorTerm.Tensor("b2", inputs["b2"].shape, True)
    b3 = TensorTerm.Tensor("b3", inputs["b3"].shape, True)
    b4 = TensorTerm.Tensor("b4", inputs["b4"].shape, True)

    M1 = (a1 + a4) @ (b1 + b4)
    M2 = (a3 + a4) @ b1
    M3 = a1 @ (b2 - b4)
    M4 = a4 @ (b3 - b1)
    M5 = (a1 + a2) @ b4
    M6 = (a3 - a1) @ (b1 + b2)
    M7 = (a2 - a4) @ (b3 + b4)

    C1 = M1 + M4 - M5 + M7
    C2 = M3 + M5
    C3 = M2 + M4
    C4 = M1 - M2 + M3 + M6
    # return C1 + C2 + C3 + C4, inputs
    # return C1 + C2 + C3 + C4, inputs
    # return C1, inputs
    return C2 + C3, inputs 
    # return a1 @ b1, inputs
