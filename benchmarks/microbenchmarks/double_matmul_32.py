from frontends.tensor import TensorTerm
from ir.dim import Dim
from ir.roll import Roll
from ir.layout import Layout
from ir.kernel import Kernel, KernelOp
import numpy as np
import random


def double_matmul_32_micro(n):
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(2)) for _ in range(128)] for _ in range(256)]
    )
    inputs["b"] = np.array(
        [[random.choice(range(2)) for _ in range(256)] for _ in range(128)]
    )
    inputs["c"] = np.array(
        [[random.choice(range(2)) for _ in range(128)] for _ in range(256)]
    )


    # create tensor term
    a_term = TensorTerm.Tensor("a", inputs["a"].shape, True)
    tensor_layout = Layout(
        a_term, [], [Dim.parse(f"[1:256:1]"), Dim.parse(f"[128]"), Dim.parse(f"[0:256:1]")], {}, n, True
    )
    a = Kernel(KernelOp.TENSOR, [], layout=tensor_layout)
    
    # replicate a 
    replicated_a_layout = Layout(
        a_term, [], [Dim.parse(f"[2]"), Dim.parse(f"[1:256:1]"), Dim.parse(f"[128]"), Dim.parse(f"[0:256:1]")], {}, n, True
    )
    replicated_a = Kernel(KernelOp.REPLICATE, [a], replicated_a_layout)

    # create tensor b 
    b_term = TensorTerm.Tensor("b", inputs["b"].shape, True)
    tensor_layout = Layout(
        b_term, [], [Dim.parse(f"[1:2:128]"), Dim.parse(f"[0:256:1]"), Dim.parse(f"[1:128:1]"), Dim.parse(f"[256]")], {}, n, True
    )
    b = Kernel(KernelOp.TENSOR, [], layout=tensor_layout)

    matmul_layout = Layout(
        a_term @ b_term, [], [Dim.parse(f"[1:2:128]"), Dim.parse(f"[1:128:1]"), Dim.parse(f"[0:256:1]")], {}, n, True
    )
    matmul = Kernel(KernelOp.MATMUL, [replicated_a, b], layout=matmul_layout)


    # replicate matmul
    replicated_layout = Layout(
        a_term @ b_term, [], [Dim.parse(f"[128]"), Dim.parse(f"[1:2:128]"), Dim.parse(f"[1:128:1]"), Dim.parse(f"[0:256:1]")], {}, n, True
    )
    r = Kernel(KernelOp.REPLICATE, [matmul], replicated_layout)

    # roll layout 
    roll = Roll(Dim.parse(f"[1:128:1]"), Dim.parse(f"[128]"))
    roll_layout = Layout(
        a_term @ b_term, [roll], [Dim.parse(f"[1:128:1]"), Dim.parse(f"[1:2:128]"), Dim.parse(f"[128]"), Dim.parse(f"[0:256:1]")], {}, n, True
    )
    rolled = Kernel(KernelOp.ROT_ROLL, [roll, r], roll_layout)

    # create tensor c 
    c_term = TensorTerm.Tensor("c", inputs["c"].shape, True)
    tensor_layout = Layout(
        c_term, [Roll(Dim.parse(f"[0:128:1]"), Dim.parse(f"[1:128:1]"))], [Dim.parse(f"[0:128:1]"), Dim.parse(f"[0:2:128]"), Dim.parse(f"[1:128:1]"), Dim.parse(f"[256]")], {}, n, True
    )
    c = Kernel(KernelOp.TENSOR, [], layout=tensor_layout)


    matmul_layout = Layout(
        a_term @ b_term @ c_term, [], [Dim.parse(f"[1:128:1]"), Dim.parse(f"[0:256:1]")], {}, n, True
    )
    result = Kernel(KernelOp.MATMUL, [rolled, c], layout=matmul_layout)

    return result, inputs
