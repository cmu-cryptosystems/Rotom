"""
Constant

Key functions:
- gen_constant: Main function for generating plaintext constants
"""

from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def gen_const(term, n):
    """Generates kernels for plaintext constants.

    This function creates kernels for plaintext constants that maintain the same layout
    as the input tensor.
    """
    output_kernels = set()
    value = term.cs[0]
    layout_spec = getattr(term, "layout", None)
    if layout_spec is not None and str(layout_spec).strip():
        layout = Layout.from_string(str(layout_spec).strip(), n, False)
        layout.term = term
        layout.ct_dims = sorted(
            [ct_dim for ct_dim in layout.ct_dims],
            key=lambda x: (x.dim, x.stride),
        )
        const_kernel = Kernel(KernelOp.CONST, [value], layout)
        output_kernels.add(const_kernel)
        return output_kernels

    new_layout = Layout(
        term,
        [],
        [Dim(0, n, 1)],
        n,
        False,
    )
    const_kernel = Kernel(KernelOp.CONST, [value], new_layout)
    output_kernels.add(const_kernel)
    return output_kernels
