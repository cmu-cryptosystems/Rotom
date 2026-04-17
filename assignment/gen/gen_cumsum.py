"""Layout generation for cumulative sum (same rank as input)."""

from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def gen_cumsum(term, kernels):
    """Generate CUMSUM kernels preserving the input layout (axis is lowering metadata)."""
    output_kernels = set()
    for kernel in kernels:
        if kernel.layout.rolls:
            continue
        cs_placeholder = Kernel(KernelOp.CS, [0], kernel.layout)
        new_layout = Layout(
            term,
            kernel.layout.rolls,
            kernel.layout.get_dims(),
            kernel.layout.n,
            kernel.layout.secret,
        )
        output_kernels.add(Kernel(KernelOp.CUMSUM, [cs_placeholder], new_layout))
    return output_kernels
