import random

import numpy as np

from frontends.tensor import TensorTerm
from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.roll import Roll


def rot_roll(n, size):
    inputs = {}
    inputs["a"] = np.array(
        [[random.choice(range(5)) for _ in range(64)] for _ in range(64)]
    )

    # create tensor term
    term = TensorTerm.Tensor("a", inputs["a"].shape, True)

    # create conversion
    tensor_layout = Layout(
        term, [], [Dim.parse(f"[0:{size}:1]"), Dim.parse(f"[1:{size}:1]")], n, True
    )
    tensor_kernel = Kernel(KernelOp.TENSOR, [], layout=tensor_layout)

    # replicate kernel
    replicated_layout = Layout(
        term,
        [],
        [
            Dim.parse(f"[{size}]"),
            Dim.parse(f"[0:{size}:1]"),
            Dim.parse(f"[1:{size}:1]"),
        ],
        n,
        True,
    )
    replicated_kernel = Kernel(KernelOp.REPLICATE, [tensor_kernel], replicated_layout)

    # rolled_layout
    roll = Roll(Dim.parse(f"[1:{size}:1]"), Dim.parse(f"[{size}]"))
    rolled_layout = Layout(
        term,
        [roll],
        [
            Dim.parse(f"[1:{size}:1]"),
            Dim.parse(f"[0:{size}:1]"),
            Dim.parse(f"[{size}]"),
        ],
        n,
        True,
    )
    kernel = Kernel(
        KernelOp.ROT_ROLL,
        [
            roll,
            replicated_kernel,
        ],
        rolled_layout,
    )

    return kernel, inputs
