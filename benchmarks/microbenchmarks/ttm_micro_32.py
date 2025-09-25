from frontends.tensor import TensorTerm
from ir.dim import Dim
from ir.roll import Roll
from ir.layout import Layout
from ir.kernel import Kernel, KernelOp
import numpy as np
import random


def ttm_micro_32(n):
    size = 64
    inputs = {}
    inputs["t"] = np.array(
        [[[random.choice(range(5)) for _ in range(size)] for _ in range(size)] for _ in range(size)]
    )
    inputs["m"] = np.array(
        [[random.choice(range(5)) for _ in range(size)] for _ in range(size)]
    )

    # create tensor term
    term = TensorTerm.Tensor("t", inputs["t"].shape, True)
    tensor_layout = Layout(
        term, [], [Dim.parse(f"[1:8:8]"), Dim.parse(f"[0:64:1]"), Dim.parse(f"[1:8:1]"), Dim.parse(f"[2:64:1]")], {}, n, True
    )
    t = Kernel(KernelOp.TENSOR, [], layout=tensor_layout)

    # create tensor m 
    term = TensorTerm.Tensor("m", inputs["m"].shape, True)
    tensor_layout = Layout(
        term, [], [Dim.parse(f"[0:64:1]"), Dim.parse(f"[8]"), Dim.parse(f"[1:64:1]")], {}, n, True
    )
    m = Kernel(KernelOp.TENSOR, [], layout=tensor_layout)


    # replicate 
    replicated_t_layout = Layout(
        term, [], [Dim.parse(f"[64]"), Dim.parse(f"[1:8:8]"), Dim.parse(f"[0:64:1]"), Dim.parse(f"[1:8:1]"), Dim.parse(f"[2:64:1]")], {}, n, True
    )
    t = Kernel(KernelOp.REPLICATE, [t], layout=replicated_t_layout)

    replicated_m_layout = Layout(
        term, [], [Dim.parse(f"[64]"), Dim.parse(f"[8]"), Dim.parse(f"[0:64:1]"), Dim.parse(f"[8]"), Dim.parse(f"[1:64:1]")], {}, n, True
    )
    m = Kernel(KernelOp.REPLICATE, [m], layout=replicated_m_layout)

    # roll m 
    rolled_m_layout = Layout(
        term, [Roll(Dim.parse(f"[0:64:1]"), Dim.parse(f"[64]"))], [Dim.parse(f"[0:64:1]"), Dim.parse(f"[8]"), Dim.parse(f"[64]"), Dim.parse(f"[8]"), Dim.parse(f"[1:64:1]")], {}, n, True
    )
    m = Kernel(KernelOp.ROT_ROLL, [Roll(Dim.parse(f"[1:64:1]"), Dim.parse(f"[64]")), m], layout=rolled_m_layout)

    
    # mul layout 
    matmul_layout = Layout(
        term, [Roll(Dim.parse(f"[2:64:1]"), Dim.parse(f"[0:64:1]"))], [Dim.parse(f"[2:64:1]"), Dim.parse(f"[1:8:8]"), Dim.parse(f"[0:64:1]"), Dim.parse(f"[1:8:1]"), Dim.parse(f"[G:64]")], {}, n, True
    )
    result = Kernel(KernelOp.MATMUL, [t, m], layout=matmul_layout)

    
    return result, inputs
