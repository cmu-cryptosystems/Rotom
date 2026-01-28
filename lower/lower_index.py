from ir.he import HETerm
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import get_dim_indices_by_dim
from util.util import split_lists


def lower_index(env, kernel):
    input_cts = env[kernel.cs[0]]
    # map kernel dims to layout indices
    dim_map = get_dim_indices_by_dim(kernel.cs[0].layout.get_dims())
    ct_indices = split_lists(dim_map[0], kernel.cs[0].layout.num_ct())

    # Get the index value from the term
    # For INDEX operations, kernel.layout.term.cs[1] contains the index value
    index_value = kernel.layout.term.cs[1]
    if not isinstance(index_value, int):
        # Handle slice or tuple indexing - for now, assume first element for slices
        if isinstance(index_value, slice):
            index_value = index_value.start if index_value.start is not None else 0
        elif isinstance(index_value, tuple):
            index_value = (
                index_value[0]
                if len(index_value) > 0 and isinstance(index_value[0], int)
                else 0
            )
        else:
            index_value = 0

    cts = {}
    cts_index = 0
    for i, ct_index in enumerate(ct_indices):
        # create mask - match the index value from the term
        # TODO: this is the issue
        mask = [1 if c == index_value else 0 for c in ct_index]
        if any(mask):
            rot_align = mask.index(1)
            cts[cts_index] = (input_cts[i] * HETerm.mask([mask])) << rot_align
            cts_index += 1
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
