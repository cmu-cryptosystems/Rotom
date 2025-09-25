from frontends.tensor import TensorTerm
from ir.dim import Dim
from ir.roll import Roll
from ir.layout import Layout
from ir.kernel import Kernel, KernelOp
import numpy as np
import random


def slot_bsgs_roll(n, size):
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(5)) for _ in range(64)] for _ in range(64)]
    )

    # create tensor term
    term = TensorTerm.Tensor("a", inputs["a"].shape, True)

    # create conversion
    tensor_layout = Layout(
        term, [], [Dim.parse(f"[0:{size}:1]"), Dim.parse(f"[1:{size}:1]")], {}, n, True
    )
    tensor_kernel = Kernel(KernelOp.TENSOR, [], layout=tensor_layout)

    # rolled_layout
    roll = Roll(Dim.parse(f"[0:{size}:1]"), Dim.parse(f"[1:{size}:1]"))
    rolled_layout = Layout(
        term,
        [roll],
        [Dim.parse(f"[0:{size}:1]"), Dim.parse(f"[1:{size}:1]")],
        {},
        n,
        True,
    )
    kernel = Kernel(
        KernelOp.BSGS_ROLL,
        [
            roll,
            tensor_kernel,
        ],
        rolled_layout,
    )

    return kernel, inputs
