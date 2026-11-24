"""Lower a CONV_DIRECT kernel (see :mod:`util.window`).

Each group is one rotation of an input ciphertext and one multiply by a
plaintext of filter elements. A ``Z`` ciphertext sums its groups, and the
output sums ``Z`` over the labels it drops: whole ciphertexts first, then a
rotate-and-sum over each summed slot piece and a mask that clears the slots
the sum leaves behind.
"""

from frontends.tensor import TensorOp, TensorTerm
from ir.dim import Dim
from ir.he import HEOp, HETerm
from ir.layout import Layout
from lower.layout_cts import LayoutCiphertexts
from lower.lower_util import rotate_and_sum
from util.layout_util import convert_layout_to_mask
from util.window import GroupWeights, output_ct, summed_pieces, window_keys


def _weights(filter_term, window, in_layout, z_layout, key):
    """The plaintext of one group, packed from the filter the backend is given."""
    n = z_layout.n
    weights = GroupWeights(window, in_layout, z_layout, key)
    gather = TensorTerm(TensorOp.GATHER, [filter_term, weights])
    return HETerm.pack(Layout(gather, [], [Dim(0, n, 1)], n, False), "0 conv weights")


def lower_conv_direct(env, kernel):
    a_kernel, z_layout, window = kernel.cs
    in_layout = a_kernel.layout
    in_cts = env[a_kernel]
    filter_term = kernel.layout.term.cs[1]
    n = z_layout.n
    secret = kernel.layout.secret

    # Z: each ciphertext is the sum of its groups' weighted rotations.
    rotated = {}
    z_cts = {}
    for key in window_keys(window, in_layout, z_layout):
        z_ct, in_ct, shift = (int(k) for k in key)
        source = in_cts[in_ct]
        if source.op == HEOp.ZERO_MASK:
            continue
        if shift:
            if (in_ct, shift) not in rotated:
                amount = shift if shift <= n // 2 else shift - n
                rotated[(in_ct, shift)] = HETerm(
                    HEOp.ROT, [source, amount], secret, "conv shift"
                )
            source = rotated[(in_ct, shift)]
        term = source * _weights(filter_term, window, in_layout, z_layout, key)
        z_cts[z_ct] = term if z_ct not in z_cts else z_cts[z_ct] + term

    # Sum over the ciphertext pieces the output drops.
    summed = {}
    for z_ct in sorted(z_cts):
        out_ct = int(output_ct(window, z_layout, z_ct))
        summed[out_ct] = (
            z_cts[z_ct] if out_ct not in summed else summed[out_ct] + z_cts[z_ct]
        )

    # Sum over the slot pieces the output drops, then clear what is left there.
    _, slot_pieces = summed_pieces(window, z_layout)
    mask = HETerm.mask([convert_layout_to_mask(kernel.layout)]) if slot_pieces else None
    cts = {}
    for out_ct in range(kernel.layout.num_ct()):
        if out_ct not in summed:
            cts[out_ct] = HETerm(HEOp.ZERO_MASK, [], False)
            continue
        ct = summed[out_ct]
        for _, extent, stride in slot_pieces:
            ct = rotate_and_sum(ct, extent, stride)
        cts[out_ct] = ct * mask if mask is not None else ct
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
