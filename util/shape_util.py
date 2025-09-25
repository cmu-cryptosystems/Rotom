from util.util import prod


def add_vec(a, b):
    return [x + y if x is not None and y is not None else None for x, y in zip(a, b)]


def mul_vec(a, b):
    return [x * y if x is not None and y is not None else None for x, y in zip(a, b)]


def add(vec, n):
    return [v + n if v is not None else None for v in vec]


def mul(vec, n):
    return [v * n if v is not None else None for v in vec]


def get_segments(dims):
    n = 1
    for dim in dims:
        n *= dim.extent

    segments = {}
    for i in range(len(dims)):
        segment_len = int(prod([dim.extent for dim in dims[i + 1 :]]))
        extent = dims[i].extent
        count = n // segment_len // extent
        segments[i] = [count, extent, segment_len]
    return segments


def get_dim_indices(dims):
    segments = get_segments(dims)
    all_indices = []
    for i in range(len(dims)):
        segment = segments[i]
        i_len = segment[0]
        j_len = segment[1]
        k_len = segment[2]
        indices = []
        for _ in range(i_len):
            for j in range(j_len):
                for _ in range(k_len):
                    indices.append(j)
        all_indices.append(indices)
    return all_indices


def layout_to_shape_indices(layout):
    dims = layout.ct_dims
    dim_indices = get_dim_indices(dims)
    unique_dims = len(set(dim.dim for dim in layout.get_dims()))

    # keep only the pertinent dims
    indices_map = {}
    for i in range(unique_dims):
        indices_map[i] = [0] * layout.num_ct()

    for i, dim in enumerate(dims):
        if dim.dim is not None:
            dim_indices[i] = mul(dim_indices[i], dim.stride)

            if dim.dim not in indices_map:
                indices_map[dim.dim] = dim_indices[i]
            else:
                indices_map[dim.dim] = add_vec(indices_map[dim.dim], dim_indices[i])

    if unique_dims == 1:
        return [(a,) for a in indices_map[0]]
    elif unique_dims == 2:
        return [(a, b) for a, b in zip(indices_map[0], indices_map[1])]
    elif unique_dims == 3:
        return [
            (a, b, c) for a, b, c in zip(indices_map[0], indices_map[1], indices_map[2])
        ]
    elif unique_dims == 4:
        return [
            (a, b, c, d)
            for a, b, c, d in zip(
                indices_map[0], indices_map[1], indices_map[2], indices_map[3]
            )
        ]
    else:
        raise NotImplementedError("more than 2 dimensions")
