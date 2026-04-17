"""Layout generation for CAST (dtype view; layout unchanged)."""

from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def gen_cast(term, kernels):
    """CAST preserves packing; dtype is handled in evaluation and lowering is a no-op."""
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
        output_kernels.add(Kernel(KernelOp.CAST, [cs_placeholder], new_layout))
    return output_kernels
