from lower.layout_cts import LayoutCiphertexts
from util.layout_util import layout_to_ct_indices


def lower_reorder(env, kernel):
    input_cts = env[kernel.cs[0]]
    cs_ct_dim_indices = layout_to_ct_indices(kernel.cs[0].layout)
    ct_dim_indices = layout_to_ct_indices(kernel.layout)

    reorder = {}
    for i, ct_dim_index in enumerate(ct_dim_indices):
        index = cs_ct_dim_indices.index(ct_dim_index)
        reorder[i] = input_cts[index]
    return LayoutCiphertexts(layout=kernel.layout, cts=reorder)
