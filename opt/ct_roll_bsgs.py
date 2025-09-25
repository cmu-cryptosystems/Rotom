from ir.kernel import KernelOp


def run_ct_roll_bsgs(candidate):
    """
    BSGS can be applied to a roll on a secret tensor
    """
    update_map = {}
    for kernel in candidate.post_order():
        # update kernel cs
        for i, cs in enumerate(kernel.cs):
            if cs in update_map:
                kernel.cs[i] = update_map[cs]

        # update kernel with rewrites
        match kernel.op:
            case KernelOp.ROLL:
                if kernel.layout.secret and len(kernel.layout.dims) == 2:
                    kernel.op = KernelOp.BSGS_ROLL
                update_map[kernel] = kernel
            case _:
                update_map[kernel] = kernel
    return candidate
