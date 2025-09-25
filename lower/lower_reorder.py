from util.layout_util import layout_to_ct_indices


def lower_reorder(env, kernel):
    print(kernel.cs[0])
    print(kernel)
    cs_ct_dim_indices = layout_to_ct_indices(kernel.cs[0].layout)
    ct_dim_indices = layout_to_ct_indices(kernel.layout)
    print(cs_ct_dim_indices)
    print(ct_dim_indices)

    reorder = {}
    for i, ct_dim_index in enumerate(ct_dim_indices):
        index = cs_ct_dim_indices.index(ct_dim_index)
        print("new index order:", index)
        reorder[i] = env[kernel.cs[0]][index]
    return reorder
