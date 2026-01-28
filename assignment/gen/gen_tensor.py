"""
Tensor layout generation utilities.

This module provides functions for generating optimal tensor layouts
for input tensors in FHE computations. It handles tiling strategies,
dimension ordering, and layout optimization for various tensor shapes.

Key functions:
- gen_tiles: Generates tile configurations for tensor layouts
- tiling_heuristic: Applies tiling heuristics for layout optimization
- gen_tensor: Main function for generating tensor layouts
"""

import itertools

from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from util.util import prod, round_to_ceiling_power_of_2, round_up


def gen_tiles(extents, starting_tile):
    """Generates tile configurations for tensor layouts.

    This function recursively generates tile configurations by exploring
    different tiling strategies. It starts with a base tile configuration
    and generates variations by adjusting tile sizes and dimensions.

    The tiling process:
    1. Takes a starting tile configuration
    2. Generates new tiles by halving the first tile size
    3. Doubles the size of subsequent tiles to maintain total extent
    4. Recursively explores all valid tile combinations

    Args:
        extents: Dictionary mapping dimension indices to their extents
        starting_tile: Initial tile configuration to build from

    Returns:
        List of tile configurations, each represented as a list of
        (dim, tile_size, num_tiles) tuples
    """
    new_tiles = []
    if starting_tile[0][1] > 1:
        new_starting_tile = (
            starting_tile[0][0],
            starting_tile[0][1] // 2,
            starting_tile[0][2] * 2,
        )
        for tile in starting_tile[1:]:
            tile_extent = extents[tile[0]]
            if tile_extent >= tile[1] * 2 and tile[2] > 1:
                new_next_tile = (tile[0], tile[1] * 2, tile[2] // 2)
                new_tiles.append([new_starting_tile, new_next_tile])
        for new_tile in new_tiles:
            new_tiles += gen_tiles(extents, new_tile)
    return new_tiles


def tiling_heuristic(dims, n):
    """Applies tiling heuristics to generate optimal tensor layouts.

    This function explores different tiling strategies for tensor layouts
    by considering all possible dimension orderings and tile configurations.
    It generates layouts that are optimized for the given HE vector size.

    The heuristic process:
    1. Generates all possible dimension orderings
    2. For each ordering, creates starting tile configurations
    3. Recursively generates tile variations
    4. Creates layouts with different tiling strategies

    Args:
        dims: List of Dim objects representing tensor dimensions
        n: HE vector size for layout optimization

    Returns:
        List of Layout objects representing different tiling strategies

    Note:
        Currently restricted to 2-dimensional tensors for simplicity
    """
    extents = {}
    for dim in dims:
        extents[dim.dim] = dim.extent

    # HACK: restrict to 2 dimensional tensors for now
    assert len(dims) == 2

    all_dims = [
        [dim.dim for dim in reorder_dims]
        for reorder_dims in itertools.permutations(dims)
    ]

    starting_tiles = []
    for dim_order in all_dims:
        tiles = []
        tile_extent = extents[dim_order[0]]
        num_tiles = round_up(extents[dim_order[0]] / tile_extent)
        tiles.append((dim_order[0], tile_extent, num_tiles))
        total_tile_extent = tile_extent
        for dim in dim_order[1:]:
            next_tile_extent = round_to_ceiling_power_of_2(n / total_tile_extent)
            num_next_tiles = round_to_ceiling_power_of_2(
                extents[dim] / next_tile_extent
            )
            tiles.append((dim, next_tile_extent, num_next_tiles))
            total_tile_extent *= next_tile_extent
        starting_tiles.append(tiles)

    all_tiles = set()
    for starting_tile in starting_tiles:
        all_tiles.add(tuple(starting_tile))
        tiles = gen_tiles(extents, starting_tile)
        for tile in tiles:
            all_tiles.add(tuple(tile))
    return all_tiles


def gen_tensor(term, secret, shape, n):
    # if the number of elements in a tensor is greater than the
    # size of a ciphertext, then the tensor will need to be tiled.
    dims = [Dim(i, length) for i, length in enumerate(shape) if length > 1]
    num_elems = prod([dim.extent for dim in dims])
    kernels = []

    if len(dims) > 2:
        # HACK: condition for ttm
        all_dims = [list(dims) for dims in itertools.permutations(dims)]
        for dims in all_dims:
            layout = Layout(term, [], dims, n, secret)
            layout.ct_dims = sorted(
                [ct_dim for ct_dim in layout.ct_dims], key=lambda x: (x.dim, x.stride)
            )
            kernel = Kernel(KernelOp.TENSOR, [], layout)
            kernels.append(kernel)
        return kernels

    if num_elems > n:
        # create a new tiling heuristic that will split
        # the dimensions of the input tensor into
        # ct_dims and slot_dims
        tiles = tiling_heuristic(dims, n)
        for tile in tiles:
            ct_dims = []
            slot_dims = []
            for dim in tile:
                if dim[1] > 1:
                    slot_dims.append(Dim(dim[0], dim[1], 1))
                if dim[2] > 1:
                    ct_dims.append(Dim(dim[0], dim[2], dim[1]))
            dims = ct_dims + slot_dims
            layout = Layout(term, [], dims, n, secret)
            layout.ct_dims = sorted(
                [ct_dim for ct_dim in layout.ct_dims], key=lambda x: (x.dim, x.stride)
            )
            kernel = Kernel(KernelOp.TENSOR, [], layout)
            kernels.append(kernel)
    else:
        # seed the layout search with all possible permutations of the dim indices
        # e.g. if there's an input tensor of [0:4:1][1:4:1], seed the layout search
        # with [0:4:1][1:4:1] and [1:4:1][0:4:1].
        diff = n // prod(dim.extent for dim in dims)
        if diff > 1:
            empty_dim = Dim(None, diff, 1, DimType.EMPTY)
            dims.append(empty_dim)
        all_dims = [list(dims) for dims in itertools.permutations(dims)]
        for dims in all_dims:
            layout = Layout(term, [], dims, n, secret)
            layout.ct_dims = sorted(
                [ct_dim for ct_dim in layout.ct_dims], key=lambda x: (x.dim, x.stride)
            )
            kernel = Kernel(KernelOp.TENSOR, [], layout)
            kernels.append(kernel)

    return kernels
