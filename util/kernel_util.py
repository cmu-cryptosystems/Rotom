from ir.kernel import KernelOp


def get_cs_op_kernels(kernel):
    cs_kernels = []
    for term in kernel.post_order():
        if term.op == KernelOp.CS:
            cs_kernels.append(term)
    return cs_kernels
