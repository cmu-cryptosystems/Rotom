from copy import deepcopy as copy

from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging


def gen_tile(term, kernels):
    """Generate TILE kernels by expanding selected logical extents."""
    reps = [int(x) for x in term.cs[1]]
    output_kernels = set()
    for kernel in kernels:
        if kernel.layout.rolls:
            continue
        new_dims = [copy(d) for d in kernel.layout.get_dims()]

        # If a logical dim is split over multiple layout dims, grow only the first fragment.
        seen = set()
        for i, d in enumerate(new_dims):
            if d.dim is None:
                continue
            if d.dim in seen:
                continue
            seen.add(d.dim)
            r = reps[d.dim]
            if r > 1:
                new_dims[i].extent *= r

        tiled_layout = dimension_merging(
            Layout(
                term,
                copy(kernel.layout.rolls),
                new_dims,
                kernel.layout.n,
                kernel.layout.secret,
            )
        )
        cs_placeholder = Kernel(KernelOp.CS, [0], kernel.layout)
        tiled_kernel = Kernel(KernelOp.TILE, [cs_placeholder], tiled_layout)
        output_kernels.add(tiled_kernel)
    return output_kernels
