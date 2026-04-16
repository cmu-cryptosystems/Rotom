from copy import deepcopy as copy

from ir.dim import Dim
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

        # If a tiled logical dim was absent in layout (common when its input extent==1),
        # materialize it explicitly.
        existing_dims = {d.dim for d in new_dims if d.dim is not None}
        for logical_dim, r in enumerate(reps):
            if r <= 1 or logical_dim in existing_dims:
                continue

            higher = [d for d in new_dims if d.dim is not None and d.dim > logical_dim]
            if higher:
                next_dim = min(higher, key=lambda d: d.dim)
                stride = int(next_dim.stride) * int(next_dim.extent)
            else:
                stride = 1

            insert_at = None
            for idx, d in enumerate(new_dims):
                if d.dim is not None and d.dim > logical_dim:
                    insert_at = idx
                    break
            if insert_at is None:
                for idx, d in enumerate(new_dims):
                    if d.dim is None:
                        insert_at = idx
                        break
            if insert_at is None:
                insert_at = len(new_dims)
            new_dims.insert(insert_at, Dim(logical_dim, r, stride))

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
