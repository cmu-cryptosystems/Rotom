from ir.he import HETerm
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import get_dim_indices_by_dim
from util.util import split_lists


def lower_index(env, kernel):
    input_cts = env[kernel.cs[0]]
    # map kernel dims to layout indices
    dim_map = get_dim_indices_by_dim(kernel.cs[0].layout.get_dims())
    ct_indices = split_lists(dim_map[0], kernel.cs[0].layout.num_ct())
    cts = {}
    cts_index = 0
    for i, ct_index in enumerate(ct_indices):
        # create mask
        mask = [1 if c == kernel.layout.offset[0] else 0 for c in ct_index]
        if any(mask):
            rot_align = mask.index(1)
            cts[cts_index] = (input_cts[i] * HETerm.mask([mask])) << rot_align
            cts_index += 1
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
