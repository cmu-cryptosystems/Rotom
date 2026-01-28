from ir.dim import DimType
from lower.layout_cts import LayoutCiphertexts
from lower.lower_util import rotate_and_sum
from util.layout_util import align_dimension_extents, get_segments


def lower_replicate(env, kernel):
    layout = kernel.layout
    cs = kernel.cs[0]
    input_cts = env[cs]

    # 1. compare slot dimensions
    rotate_and_sums = []
    mul_offset = 1

    # align slot dimensions to see if replication can be performed internally
    slot_dims, cs_slot_dims = align_dimension_extents(
        layout.slot_dims, cs.layout.slot_dims
    )

    for layout_dim, cs_dim in zip(slot_dims[::-1], cs_slot_dims[::-1]):
        assert layout_dim.dim == cs_dim.dim
        assert layout_dim.extent == cs_dim.extent
        if layout_dim.dim_type == DimType.FILL and cs_dim.dim_type == DimType.EMPTY:
            rotate_and_sums.append((layout_dim.extent, mul_offset))
        mul_offset *= layout_dim.extent

    # 2. apply rotate_and_sum
    replicated_cts = {}
    for index, term in input_cts.items():
        base_term = term
        for extent, mul_offset in rotate_and_sums:
            base_term = rotate_and_sum(base_term, extent, mul_offset, True)
        replicated_cts[index] = base_term

    # 3. compare ct dimensions
    num_ct = layout.num_ct()
    ct_segments = get_segments(layout.ct_dims)
    relevant_ct_indices = [0] * num_ct
    rolled_dims = []
    for roll in layout.rolls:
        rolled_dims.append(roll.dim_to_roll)
        rolled_dims.append(roll.dim_to_roll_by)

    for ct_index, ct_dim in enumerate(layout.ct_dims):
        if ct_dim.dim is None and ct_dim not in rolled_dims:
            continue
        i_len = ct_segments[ct_index][0]
        j_len = ct_segments[ct_index][1]
        k_len = ct_segments[ct_index][2]
        indices = []
        for i in range(i_len):
            for j in range(j_len):
                for k in range(k_len):
                    indices.append(j * k_len)
        relevant_ct_indices = [a + b for a, b in zip(relevant_ct_indices, indices)]

    cts = {}
    for i, ct_index in enumerate(relevant_ct_indices):
        if ct_index not in replicated_cts:
            raise KeyError(
                f"Computed ct_index {ct_index} not found in replicated_cts (keys: {list(replicated_cts.keys())})"
            )
        cts[i] = replicated_cts[ct_index]
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
