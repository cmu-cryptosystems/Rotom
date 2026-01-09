"""
Lowering should account for layout and dimensions at each step
"""

from ir.kernel import KernelOp
from lower.circuit_opts.mask_opts import (
    mask_identity_opt,
    zero_mask_identity_opt,
    zero_mask_opt,
)
from lower.circuit_opts.rot_opts import rot_zero_opt
from lower.lower_add import lower_add
from lower.lower_combine import lower_combine
from lower.lower_compact import lower_compact
from lower.lower_conv2d import lower_conv2d
from lower.lower_conversion import lower_conversion
from lower.lower_cs_pack import lower_cs_pack
from lower.lower_index import lower_index
from lower.lower_matmul import lower_bsgs_matmul, lower_matmul
from lower.lower_mul import lower_mul
from lower.lower_permute import lower_permute
from lower.lower_reorder import lower_reorder
from lower.lower_replicate import lower_replicate
from lower.lower_rescale import lower_rescale
from lower.lower_reshape import lower_reshape
from lower.lower_roll import (
    lower_bsgs_roll,
    lower_bsgs_rot_roll,
    lower_roll,
    lower_rot_roll,
    lower_split_roll,
)
from lower.lower_sub import lower_sub
from lower.lower_sum import lower_sum
from lower.lower_tensor import lower_tensor
from lower.lower_transpose import lower_transpose


class Lower:
    """
    Convert tensor_ir and layout assignments to high-level FHE circuits
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.env = {}

    def lower(self):
        for term in self.kernel.post_order():
            match term.op:
                case KernelOp.TENSOR:
                    self.env[term] = lower_tensor(term)
                case KernelOp.CS:
                    self.env[term] = lower_cs_pack(term)
                case KernelOp.REPLICATE:
                    self.env[term] = lower_replicate(self.env, term)
                case KernelOp.ADD:
                    self.env[term] = lower_add(self.env, term)
                case KernelOp.SUB:
                    self.env[term] = lower_sub(self.env, term)
                case KernelOp.MUL:
                    self.env[term] = lower_mul(self.env, term)
                case KernelOp.CONV2D:
                    self.env[term] = lower_conv2d(self.env, term)
                case KernelOp.SUM:
                    self.env[term] = lower_sum(self.env, term)
                case KernelOp.CONVERSION:
                    self.env[term] = lower_conversion(self.env, term)
                case KernelOp.REORDER:
                    self.env[term] = lower_reorder(self.env, term)
                case KernelOp.MATMUL:
                    self.env[term] = lower_matmul(self.env, term)
                case KernelOp.BSGS_MATMUL:
                    self.env[term] = lower_bsgs_matmul(self.env, term)
                case KernelOp.ROLL:
                    self.env[term] = lower_roll(self.env, term)
                case KernelOp.ROT_ROLL:
                    self.env[term] = lower_rot_roll(self.env, term)
                case KernelOp.BSGS_ROT_ROLL:
                    self.env[term] = lower_bsgs_rot_roll(self.env, term)
                case KernelOp.BSGS_ROLL:
                    self.env[term] = lower_bsgs_roll(self.env, term)
                case KernelOp.SPLIT_ROLL:
                    self.env[term] = lower_split_roll(self.env, term)
                case KernelOp.TRANSPOSE:
                    self.env[term] = lower_transpose(self.env, term)
                case KernelOp.RESHAPE:
                    self.env[term] = lower_reshape(self.env, term)
                case KernelOp.PERMUTE:
                    self.env[term] = lower_permute(self.env, term)
                case KernelOp.COMPACT:
                    self.env[term] = lower_compact(self.env, term)
                case KernelOp.INDEX:
                    self.env[term] = lower_index(self.env, term)
                case KernelOp.COMBINE:
                    self.env[term] = lower_combine(self.env, term)
                case KernelOp.RESCALE:
                    self.env[term] = lower_rescale(self.env, term)
                case _:
                    raise NotImplementedError(term.op)

    def opt(self):
        layout_cts = self.env[self.kernel]
        opt_cts = {}
        for ct_idx, ct in layout_cts.items():
            opt_ct = rot_zero_opt(ct)
            opt_ct = zero_mask_opt(ct)
            opt_ct = mask_identity_opt(ct)
            opt_ct = zero_mask_identity_opt(ct)
            opt_cts[ct_idx] = opt_ct

        # replace with optimized cts
        from lower.layout_cts import LayoutCiphertexts

        self.env[self.kernel] = LayoutCiphertexts(layout=layout_cts.layout, cts=opt_cts)

    def run(self):
        self.lower()
        self.opt()
        return self.env
