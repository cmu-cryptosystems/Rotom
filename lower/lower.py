"""
Lowering from layout-assigned kernels to HE circuit terms.

Conventions:
- Most lower functions have the signature ``lower_foo(env, kernel)`` where
  ``env`` is a mapping from input kernels to their lowered ``LayoutCiphertexts``.
- A few leaf lowers that do not depend on prior env entries take just
  ``(kernel)`` (e.g. ``lower_tensor``, ``lower_const``, ``lower_cs_pack``,
  ``lower_punctured_tensor``).

The ``Lower`` class is the single entrypoint used by tests and backends; any
new kernel op should be wired in here with a corresponding ``lower_*`` helper.
"""

from ir.kernel import KernelOp
from lower.circuit_opts.combined_opts import fused_circuit_opts

# replace with optimized cts
from lower.layout_cts import LayoutCiphertexts
from lower.lower_add import lower_add
from lower.lower_combine import lower_combine
from lower.lower_compact import lower_compact
from lower.lower_concat import lower_concat
from lower.lower_const import lower_const
from lower.lower_conv2d import lower_conv2d
from lower.lower_conv2d_roll import lower_conv2d_roll
from lower.lower_conv3d import lower_conv3d
from lower.lower_conversion import lower_conversion
from lower.lower_cs_pack import lower_cs_pack
from lower.lower_index import lower_index
from lower.lower_mask_tensor import lower_punctured_tensor
from lower.lower_matmul import lower_bsgs_matmul, lower_matmul
from lower.lower_mul import lower_mul
from lower.lower_permute import lower_permute
from lower.lower_poly_call import lower_poly_call
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
from lower.lower_select import lower_select
from lower.lower_sub import lower_sub
from lower.lower_sum import lower_sum
from lower.lower_tensor import lower_tensor
from lower.lower_tile import lower_tile
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
                case KernelOp.PUNCTURED_TENSOR:
                    self.env[term] = lower_punctured_tensor(term)
                case KernelOp.CS:
                    self.env[term] = lower_cs_pack(term)
                case KernelOp.CONST:
                    self.env[term] = lower_const(term)
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
                case KernelOp.CONV2D_ROLL:
                    self.env[term] = lower_conv2d_roll(self.env, term)
                case KernelOp.CONV3D:
                    self.env[term] = lower_conv3d(self.env, term)
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
                case KernelOp.TILE:
                    self.env[term] = lower_tile(self.env, term)
                case KernelOp.CONCAT:
                    self.env[term] = lower_concat(self.env, term)
                case KernelOp.SELECT:
                    self.env[term] = lower_select(self.env, term)
                case KernelOp.COMBINE:
                    self.env[term] = lower_combine(self.env, term)
                case KernelOp.RESCALE:
                    self.env[term] = lower_rescale(self.env, term)
                case KernelOp.POLY_CALL:
                    self.env[term] = lower_poly_call(self.env, term)
                case _:
                    raise NotImplementedError(term.op)

    def opt(self):
        layout_cts = self.env[self.kernel]
        opt_cts = {}
        for ct_idx, ct in layout_cts.items():
            # One fused post-order pass (same semantics as the former four-pass pipeline).
            opt_ct = fused_circuit_opts(ct)

            # Lift rotations to packing phase (optimization for convolution)
            # This replaces ROT(CS(PACK(...)), rot_amt) with pre-rotated PACK operations
            # The backend can then rotate during packing instead of homomorphically
            # opt_ct = lift_rotations_to_pack(opt_ct)

            opt_cts[ct_idx] = opt_ct

        self.env[self.kernel] = LayoutCiphertexts(layout=layout_cts.layout, cts=opt_cts)

    def run(self):
        self.lower()
        self.opt()
        return self.env
