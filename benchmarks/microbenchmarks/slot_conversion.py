import random

import numpy as np

from frontends.tensor import TensorTerm
from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def slot_conversion(n, size):
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

    # converted_layout
    converted_layout = Layout(
        term,
        [],
        [Dim.parse(f"[1:{size}:1]"), Dim.parse(f"[{size}]")],
        n,
        True,
    )
    kernel = Kernel(
        KernelOp.CONVERSION,
        [
            tuple([Dim.parse(f"[0:{size}:1]"), Dim.parse(f"[1:{size}:1]")]),
            tuple([Dim.parse(f"[1:{size}:1]"), Dim.parse(f"[{size}]")]),
            tensor_kernel,
        ],
        converted_layout,
    )

    return kernel, inputs
