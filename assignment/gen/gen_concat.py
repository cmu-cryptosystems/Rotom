from copy import deepcopy as copy

from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging


def gen_concat(term, cs_kernels):
    """Generate CONCAT kernels for children sharing a compatible layout."""
    axis = int(term.cs[1])
    input_terms = term.cs[0]
    input_shapes = [t.cs[1] for t in input_terms]

    if not cs_kernels:
        return set()

    # Build candidate lookup per input by merged layout string.
    by_layout = []
    for kernel_list in cs_kernels:
        m = {}
        for k in kernel_list:
            if k.layout.rolls:
                continue
            lk = dimension_merging(k.layout)
            m[lk.layout_str()] = k
        by_layout.append(m)

    common_layouts = set(by_layout[0].keys())
    for m in by_layout[1:]:
        common_layouts &= set(m.keys())

    output_kernels = set()
    for layout_key in common_layouts:
        chosen = [m[layout_key] for m in by_layout]
        base_layout = chosen[0].layout
        new_dims = [copy(d) for d in base_layout.get_dims()]

        # Scale first fragment of concat axis by total/base extent.
        base_extent = int(input_shapes[0][axis])
        out_extent = sum(int(s[axis]) for s in input_shapes)
        factor = out_extent // base_extent if base_extent > 0 else 1

        axis_frag_idx = None
        for i, d in enumerate(new_dims):
            if d.dim == axis:
                axis_frag_idx = i
                break
        if axis_frag_idx is None:
            # Materialize missing axis dim (e.g., extent-1 elided in input layouts).
            new_dims.append(Dim(axis, out_extent, 1))
        else:
            new_dims[axis_frag_idx].extent *= max(1, factor)

        concat_layout = dimension_merging(
            Layout(
                term,
                [],
                new_dims,
                base_layout.n,
                base_layout.secret,
            )
        )
        cs_placeholders = [
            Kernel(KernelOp.CS, [i], chosen[i].layout) for i in range(len(chosen))
        ]
        concat_kernel = Kernel(KernelOp.CONCAT, cs_placeholders, concat_layout)
        output_kernels.add(concat_kernel)
    return output_kernels
