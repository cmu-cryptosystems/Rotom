"""
Polynomial approximations.

Poly_Call applies an element-wise polynomial or named function (e.g. identity,
silu, batchnorm). The output has the same layout as the input.

Key functions:
- gen_poly: Main function for generating Poly operation kernels
"""

from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def gen_poly_call(term, kernels):
    """Generates kernels for Poly operations.

    Creates kernels that preserve the input layout (element-wise operation).

    Args:
        term: TensorTerm representing the Poly operation (input = term.cs[0];
              use PolyCallArgs.from_term(term) for .name, .lower_bound, .upper_bound)
        kernels: List of input kernels to generate Poly kernels for

    Returns:
        Set of Kernel objects representing Poly operation kernels
    """
    output_kernels = set()
    for kernel in kernels:
        cs_placeholder = Kernel(KernelOp.CS, [0], kernel.layout)

        new_layout = Layout(
            term,
            kernel.layout.rolls,
            kernel.layout.get_dims(),
            kernel.layout.n,
            kernel.layout.secret,
        )

        poly_kernel = Kernel(KernelOp.POLY_CALL, [cs_placeholder], new_layout)
        output_kernels.add(poly_kernel)
    return output_kernels
