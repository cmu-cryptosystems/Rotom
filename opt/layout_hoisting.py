"""
Layout hoisting optimization for FHE kernels.

This module implements layout hoisting optimization that moves layout
assignments to optimal positions in the computation graph. The optimization
focuses on hoisting layout assignments for matrix multiplication operations
to improve efficiency and reduce redundant computations.

Key Concepts:
- Layout Hoisting: Moving layout assignments to optimal positions
- Matrix Multiplication Optimization: Specialized optimization for matmul
- Tensor Placeholder Creation: Creating optimized tensor representations
- Computation Efficiency: Improving overall computation efficiency
"""

from frontends.tensor import TensorOp
from ir.kernel import Kernel, KernelOp
from util.kernel_util import get_cs_op_kernels


def run_layout_hoisting(candidate):
    """
    Apply layout hoisting optimization to improve computation efficiency.

    This function hoists layout assignments for matrix multiplication
    operations, creating optimized tensor representations and improving
    overall computation efficiency.

    Args:
        candidate: Kernel to apply layout hoisting to

    Returns:
        List of optimized kernels with hoisted layouts
    """
    hoisted = {}
    output_kernels = [candidate]
    match candidate.op:
        case KernelOp.MATMUL:
            cs_kernels = get_cs_op_kernels(candidate)
            a_cs_placeholder = None
            b_cs_placeholder = None

            cs_kernel_ids = set()
            for cs_kernel in cs_kernels:
                cs_kernel_ids.add(cs_kernel.cs[0])
            # create new tensor term
            if 0 in cs_kernel_ids and cs_kernels[0].layout.term.op == TensorOp.TENSOR:
                a_term = candidate.cs[0].layout.term
                hoisted_tensor_kernel = Kernel(
                    KernelOp.TENSOR, [], candidate.cs[0].layout
                )
                if a_term not in hoisted:
                    hoisted[a_term] = set()
                hoisted[a_term].add(hoisted_tensor_kernel)
                a_cs_placeholder = Kernel(KernelOp.CS, [0], candidate.cs[0].layout)

            if 1 in cs_kernel_ids and cs_kernels[1].layout.term.op == TensorOp.TENSOR:
                b_term = candidate.cs[1].layout.term
                hoisted_tensor_kernel = Kernel(
                    KernelOp.TENSOR, [], candidate.cs[1].layout
                )
                if b_term not in hoisted:
                    hoisted[b_term] = set()
                hoisted[b_term].add(hoisted_tensor_kernel)
                b_cs_placeholder = Kernel(KernelOp.CS, [1], candidate.cs[1].layout)

            # create new matmul operation
            if a_cs_placeholder and b_cs_placeholder:
                matmul_kernel = Kernel(
                    KernelOp.MATMUL,
                    [a_cs_placeholder, b_cs_placeholder],
                    candidate.layout,
                )
                output_kernels.append(matmul_kernel)
            if a_cs_placeholder:
                matmul_kernel = Kernel(
                    KernelOp.MATMUL,
                    [a_cs_placeholder, candidate.cs[1]],
                    candidate.layout,
                )
                output_kernels.append(matmul_kernel)
            if b_cs_placeholder:
                matmul_kernel = Kernel(
                    KernelOp.MATMUL,
                    [candidate.cs[0], b_cs_placeholder],
                    candidate.layout,
                )
                output_kernels.append(matmul_kernel)
            return output_kernels, hoisted

        case KernelOp.BSGS_MATMUL:
            cs_kernels = get_cs_op_kernels(candidate)
            a_cs_placeholder = None
            b_cs_placeholder = None

            cs_kernel_ids = set()
            for cs_kernel in cs_kernels:
                cs_kernel_ids.add(cs_kernel.cs[0])
            # create new tensor term
            if 0 in cs_kernel_ids and cs_kernels[0].layout.term.op == TensorOp.TENSOR:
                a_term = candidate.cs[1].layout.term
                hoisted_tensor_kernel = Kernel(
                    KernelOp.TENSOR, [], candidate.cs[1].layout
                )
                if a_term not in hoisted:
                    hoisted[a_term] = set()
                hoisted[a_term].add(hoisted_tensor_kernel)
                a_cs_placeholder = Kernel(KernelOp.CS, [0], candidate.cs[1].layout)

            if 1 in cs_kernel_ids and cs_kernels[1].layout.term.op == TensorOp.TENSOR:
                b_term = candidate.cs[2].layout.term
                hoisted_tensor_kernel = Kernel(
                    KernelOp.TENSOR, [], candidate.cs[2].layout
                )
                if b_term not in hoisted:
                    hoisted[b_term] = set()
                hoisted[b_term].add(hoisted_tensor_kernel)
                b_cs_placeholder = Kernel(KernelOp.CS, [1], candidate.cs[2].layout)

            # create new matmul operation
            if a_cs_placeholder and b_cs_placeholder:
                matmul_kernel = Kernel(
                    KernelOp.BSGS_MATMUL,
                    [candidate.cs[0], a_cs_placeholder, b_cs_placeholder],
                    candidate.layout,
                )
                output_kernels.append(matmul_kernel)
            if a_cs_placeholder:
                matmul_kernel = Kernel(
                    KernelOp.BSGS_MATMUL,
                    [candidate.cs[0], a_cs_placeholder, candidate.cs[2]],
                    candidate.layout,
                )
                output_kernels.append(matmul_kernel)
            if b_cs_placeholder:
                matmul_kernel = Kernel(
                    KernelOp.BSGS_MATMUL,
                    [candidate.cs[0], candidate.cs[1], b_cs_placeholder],
                    candidate.layout,
                )
                output_kernels.append(matmul_kernel)
            return output_kernels, hoisted

        case _:
            # TODO: add other operation as well
            return output_kernels, hoisted
