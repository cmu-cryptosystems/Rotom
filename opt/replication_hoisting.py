"""
Replication hoisting optimization for FHE kernels.

This module implements replication hoisting optimization that moves
replication operations to optimal positions in the computation graph.
The optimization focuses on hoisting replications that are performed
on input ciphertexts to reduce redundant computations.

Key Concepts:
- Replication Hoisting: Moving replication operations to optimal positions
- Input Optimization: Optimizing replications on input ciphertexts
- Redundancy Reduction: Eliminating redundant replication operations
- Cost Minimization: Reducing the cost of data replication
"""

from copy import deepcopy as copy

from frontends.tensor import TensorOp
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def run_replication_hoisting(candidate):
    """
    Apply replication hoisting optimization to reduce redundant replications.

    This optimization tries to hoist out replications that are performed
    on input ciphertexts, reducing redundant computations and improving
    overall efficiency.

    Args:
        candidate: Kernel to apply replication hoisting to

    Returns:
        Tuple of (optimized kernel, hoisted map) with replications hoisted
    """
    hoisted_map = {}
    update_map = {}

    for kernel in candidate.post_order():
        # update kernel cs
        for i, cs in enumerate(kernel.cs):
            if cs in update_map:
                kernel.cs[i] = update_map[cs]

        # update kernel with rewrites
        match kernel.op:
            case KernelOp.REPLICATE:
                if (
                    kernel.cs[0].op == KernelOp.CS
                    and kernel.cs[0].layout.term.op == TensorOp.TENSOR
                ):
                    # get cs kernel
                    cs_kernel = kernel.cs[0]

                    # create a new CS kernel with replicated slot dims
                    replicated_slot_dims = kernel.layout.slot_dims
                    new_cs_layout = Layout(
                        cs_kernel.layout.term,
                        cs_kernel.layout.rolls,
                        copy(replicated_slot_dims),
                        cs_kernel.layout.n,
                        cs_kernel.layout.secret,
                    )
                    new_cs_kernel = Kernel(
                        KernelOp.CS, [cs_kernel.cs[0]], new_cs_layout
                    )

                    # create a new input Tensor kernel with replicated slot dims
                    new_tensor_layout = copy(new_cs_layout)
                    new_tensor_kernel = Kernel(KernelOp.TENSOR, [], new_tensor_layout)

                    # add new tensor kernel to hoisted
                    if new_tensor_kernel.layout.term not in hoisted_map:
                        hoisted_map[new_tensor_kernel.layout.term] = set()
                    hoisted_map[new_tensor_kernel.layout.term].add(new_tensor_kernel)

                    # update kernel
                    if new_cs_layout.get_dims() != kernel.layout.get_dims():
                        # receate replication
                        replicated_kernel = Kernel(
                            KernelOp.REPLICATE, [new_cs_kernel], copy(kernel.layout)
                        )
                        update_map[kernel] = replicated_kernel
                    else:
                        update_map[kernel] = new_cs_kernel
                else:
                    update_map[kernel] = kernel
            case _:
                update_map[kernel] = kernel
    return candidate, hoisted_map
