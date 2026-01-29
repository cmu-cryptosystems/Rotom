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

    # Step 1: [0:size][1:size] -> [1:size];[0:size][G:size]
    tensor_layout = Layout(
        term, [], [Dim.parse(f"[0:{size}:1]"), Dim.parse(f"[1:{size}:1]")], n, True
    )
    tensor_kernel = Kernel(KernelOp.TENSOR, [], layout=tensor_layout)

    layout_1 = Layout(
        term,
        [],
        [
            Dim.parse(f"[1:{size}:1]"),
            Dim.parse(f"[0:{size}:1]"),
            Dim.parse(f"[G:{size}]"),
        ],
        n,
        True,
    )
    kernel_1 = Kernel(
        KernelOp.CONVERSION,
        [
            tuple([Dim.parse(f"[0:{size}:1]"), Dim.parse(f"[1:{size}:1]")]),
            tuple(
                [
                    Dim.parse(f"[1:{size}:1]"),
                    Dim.parse(f"[0:{size}:1]"),
                    Dim.parse(f"[G:{size}]"),
                ]
            ),
            tensor_kernel,
        ],
        layout_1,
    )

    # Step 2: [1:size];[0:size][G:size] -> [1:size];[G:size][G:size]
    layout_2 = Layout(
        term,
        [],
        [
            Dim.parse(f"[1:{size}:1]"),
            Dim.parse(f"[G:{size}]"),
            Dim.parse(f"[G:{size}]"),
        ],
        n,
        True,
    )
    kernel_2 = Kernel(
        KernelOp.CONVERSION,
        [
            tuple(
                [
                    Dim.parse(f"[1:{size}:1]"),
                    Dim.parse(f"[0:{size}:1]"),
                    Dim.parse(f"[G:{size}]"),
                ]
            ),
            tuple(
                [
                    Dim.parse(f"[1:{size}:1]"),
                    Dim.parse(f"[G:{size}]"),
                    Dim.parse(f"[G:{size}]"),
                ]
            ),
            kernel_1,
        ],
        layout_2,
    )

    # Step 3: [1:size];[G:size][G:size] -> [1:size][G:size]
    layout_3 = Layout(
        term,
        [],
        [Dim.parse(f"[1:{size}:1]"), Dim.parse(f"[G:{size}]")],
        n,
        True,
    )
    kernel_3 = Kernel(
        KernelOp.CONVERSION,
        [
            tuple(
                [
                    Dim.parse(f"[1:{size}:1]"),
                    Dim.parse(f"[G:{size}]"),
                    Dim.parse(f"[G:{size}]"),
                ]
            ),
            tuple([Dim.parse(f"[1:{size}:1]"), Dim.parse(f"[G:{size}]")]),
            kernel_2,
        ],
        layout_3,
    )

    # Step 4: [1:size][G:size] -> [1:size][size] (replication)
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
            tuple([Dim.parse(f"[1:{size}:1]"), Dim.parse(f"[G:{size}]")]),
            tuple([Dim.parse(f"[1:{size}:1]"), Dim.parse(f"[{size}]")]),
            kernel_3,
        ],
        converted_layout,
    )

    return kernel, inputs
