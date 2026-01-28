import random

import numpy as np

from frontends.tensor import TensorTerm


def strassens_matmul():
    """Strassen's algorithm matrix multiplication benchmark.
    
    Creates a 128x128 matrix multiplication using Strassen's algorithm.
    Both input matrices are secret (encrypted).
    
    Returns:
        tuple: (tensor_ir, inputs) where tensor_ir is the matrix multiplication
               expression and inputs is a dictionary of input matrices
    """
    size = 128
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
    )
    
    # Create tensor terms - both are secret for Strassen's algorithm
    a = TensorTerm.Tensor("a", inputs["a"].shape, True)
    b = TensorTerm.Tensor("b", inputs["b"].shape, True)
    
    # Return matrix multiplication expression
    return a @ b, inputs
