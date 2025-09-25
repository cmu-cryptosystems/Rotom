"""
Strassen's algorithm layout generation utilities.

This module provides functions for generating optimal layouts for matrix
multiplication using Strassen's algorithm in FHE computations. Strassen's
algorithm reduces the number of multiplications required for matrix
multiplication from O(n³) to approximately O(n^2.81) by using a divide-and-conquer
approach with clever submatrix operations.

Key functions:
- get_shape: Extracts shape information from kernels
- traverse_ct_dims_helper: Helper for traversing ciphertext dimensions
- gen_strassens: Main function for generating Strassen's algorithm layouts
"""

from ir.dim import DimType
from ir.layout import Layout
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from assignment.gen.gen_binop import gen_binop
from opt.bsgs_matmul import run_bsgs_matmul
from opt.ct_roll_bsgs import run_ct_roll_bsgs
from opt.roll_propagation import run_roll_propogation
from opt.roll_reordering import run_roll_reordering
from opt.rot_roll import run_rot_roll

from copy import deepcopy as copy


def get_shape(kernel):
    """Extracts shape information from a kernel layout.
    
    This function analyzes a kernel's layout dimensions and returns
    the corresponding tensor shape. It aggregates dimensions with the
    same index to compute the total extent for each dimension.
    
    Args:
        kernel: Kernel to extract shape from
        
    Returns:
        List of integers representing the tensor shape
    """
    shapes = {}
    for dim in kernel.layout.get_dims():
        if dim.dim not in shapes:
            shapes[dim.dim] = dim.extent
        else:
            shapes[dim.dim] *= dim.extent
    shape = []
    for i in range(len(shapes)):
        shape.append(shapes[i])
    return shape


def traverse_ct_dims_helper(remaining_ct_dims, offsets):
    if not remaining_ct_dims:
        return offsets
    ct_dim = remaining_ct_dims[0]
    dim = ct_dim.dim
    extent = ct_dim.extent
    stride = ct_dim.stride

    dim_offsets = [i * stride for i in range(extent)]

    if dim not in offsets:
        offsets[dim] = dim_offsets
    else:
        new_offsets = []
        for a in offsets[dim]:
            for b in dim_offsets:
                new_offsets.append(a + b)
        offsets[dim] = new_offsets
    return traverse_ct_dims_helper(remaining_ct_dims[1:], offsets)


def traverse_ct_dims(ct_dims):
    offsets = {}
    offsets = traverse_ct_dims_helper(ct_dims, offsets)

    indices = []
    if len(offsets) == 1:
        for k, v in offsets.items():
            for i in v:
                indices.append({k: i})
    elif len(offsets) == 2:
        for k, v in offsets.items():
            if not indices:
                for i in v:
                    indices.append({k: i})
            else:
                new_indices = []
                for index in indices:
                    for i in v:
                        new_index = copy(index)
                        new_index[k] = i
                        new_indices.append(new_index)
                indices = new_indices
    return indices


def check_slot_dims_in_tiles(a_kernel, b_kernel):
    a_slot_dims = a_kernel.layout.slot_dims
    b_slot_dims = b_kernel.layout.slot_dims

    # check that extents match
    a_extents = [dim.extent for dim in a_slot_dims]
    b_extents = [dim.extent for dim in b_slot_dims]
    if a_extents != b_extents:
        return False

    # check that summation dimensions are aligned
    for a_dim, b_dim in zip(a_slot_dims, b_slot_dims):
        if a_dim.dim is not None and b_dim.dim is not None and a_dim.dim == b_dim.dim:
            return False

    return True


def check_ct_dims_in_tiles(a_kernel, b_kernel):
    a_ct_dims = a_kernel.layout.ct_dims
    b_ct_dims = b_kernel.layout.ct_dims

    # check that both are tiled into squares
    if (
        len(set([ct_dim.dim for ct_dim in a_ct_dims])) != 2
        or len(set([ct_dim.dim for ct_dim in b_ct_dims])) != 2
    ):
        return False
    # TODO: check that both can be tiled into squares
    return True


def check_no_rolls(kernel):
    return not kernel.layout.rolls


def check_strassens(a_kernel, b_kernel):
    return (
        check_slot_dims_in_tiles(a_kernel, b_kernel)
        and check_ct_dims_in_tiles(a_kernel, b_kernel)
        and check_no_rolls(a_kernel)
        and check_no_rolls(b_kernel)
    )


def add_tiles(a_kernel, b_kernel):
    a_kernel = copy(a_kernel)
    b_kernel = copy(b_kernel)
    a_kernel.layout.offset = {}
    b_kernel.layout.offset = {}
    return Kernel(KernelOp.ADD, [a_kernel, b_kernel], a_kernel.layout)


def sub_tiles(a_kernel, b_kernel):
    a_kernel = copy(a_kernel)
    b_kernel = copy(b_kernel)
    a_kernel.layout.offset = {}
    b_kernel.layout.offset = {}
    return Kernel(KernelOp.SUB, [a_kernel, b_kernel], a_kernel.layout)


def matmul_tiles(term, a_kernel, b_kernel, roll_flag, network):
    matmul_kernels = gen_binop(
        term,
        [[a_kernel], [b_kernel]],
        [get_shape(a_kernel), get_shape(b_kernel)],
        roll_flag,
    )

    optimized_kernels = set()
    for kernel in matmul_kernels:
        opt_kernel = copy(kernel)
        opt_kernel = run_roll_propogation(opt_kernel)
        opt_kernels = run_roll_reordering(opt_kernel)
        for opt_kernel in opt_kernels:
            opt_kernel = run_rot_roll(opt_kernel)
            opt_kernel = run_ct_roll_bsgs(opt_kernel)
            opt_kernel = run_bsgs_matmul(opt_kernel)
            optimized_kernels.add(opt_kernel)
    matmul_kernels = optimized_kernels

    tiled_matmul = {}
    for matmul_kernel in matmul_kernels:
        # remove offset
        matmul_kernel.layout.offset = {}
        if matmul_kernel.layout not in tiled_matmul:
            tiled_matmul[matmul_kernel.layout] = matmul_kernel
        elif (
            KernelCost(matmul_kernel, network).total_cost()
            < KernelCost(tiled_matmul[matmul_kernel.layout], network).total_cost()
        ):
            tiled_matmul[matmul_kernel.layout] = matmul_kernel
    return tiled_matmul


def strassens_opt(term, a_tiles, b_tiles, roll_flag, network):
    # perform addition and matmul
    A1 = a_tiles[0]
    A2 = a_tiles[1]
    A3 = a_tiles[2]
    A4 = a_tiles[3]
    B1 = b_tiles[0]
    B2 = b_tiles[1]
    B3 = b_tiles[2]
    B4 = b_tiles[3]

    # T1 = A1 + A4
    T1 = add_tiles(A1, A4)

    # T2 = B1 + B4
    T2 = add_tiles(B1, B4)

    # M1 = T1 @ T2
    M1 = matmul_tiles(term, T1, T2, roll_flag, network)

    # T3 = A3 + A4
    T3 = add_tiles(A3, A4)

    # M2 = T3 @ B1
    M2 = matmul_tiles(term, T3, B1, roll_flag, network)

    # T4 = B2 - B4
    T4 = sub_tiles(B2, B4)

    # M3 = A1 @ T4
    M3 = matmul_tiles(term, A1, T4, roll_flag, network)

    # T5 = B3 - B1
    T5 = sub_tiles(B3, B1)

    # M4 = A4 @ T5
    M4 = matmul_tiles(term, A4, T5, roll_flag, network)

    # T6 = A1 + A2
    T6 = add_tiles(A1, A2)

    # M5 = T6 @ B4
    M5 = matmul_tiles(term, T6, B4, roll_flag, network)

    # T7 = A3 - A1
    T7 = sub_tiles(A3, A1)
    # T8 = B1 + B2
    T8 = add_tiles(B1, B2)
    # M6 = T7 @ T8
    M6 = matmul_tiles(term, T7, T8, roll_flag, network)

    # T9 = A2 - A4
    T9 = sub_tiles(A2, A4)
    # T10 = B3 + B4
    T10 = add_tiles(B3, B4)
    # M7 = T9 @ T10
    M7 = matmul_tiles(term, T9, T10, roll_flag, network)

    # match on layouts
    output_kernels = []
    for layout in M1.keys():
        C1 = add_tiles(
            sub_tiles(add_tiles(M1[layout], M4[layout]),
                      M5[layout]), M7[layout]
        )
        C1.layout.offset = A1.layout.offset
        C2 = add_tiles(M3[layout], M5[layout])
        C2.layout.offset = A2.layout.offset
        C3 = add_tiles(M2[layout], M4[layout])
        C3.layout.offset = A3.layout.offset
        C4 = add_tiles(
            add_tiles(sub_tiles(M1[layout], M2[layout]),
                      M3[layout]), M6[layout]
        )
        C4.layout.offset = A4.layout.offset

        kernel_layout = copy(layout)
        kernel_layout.offset = {}
        # output_kernels.append(Kernel(KernelOp.COMBINE, [C1, C2, C3, C4], kernel_layout))
    return output_kernels


def gen_strassens(term, cs_kernels, roll_flag, network):
    print("generating strassen's:")
    outputs = set()
    for a_kernel in cs_kernels[0]:
        for b_kernel in cs_kernels[1]:
            if check_strassens(a_kernel, b_kernel):
                # create cs_placeholders
                a_cs_placeholder = Kernel(KernelOp.CS, [0], a_kernel.layout)
                b_cs_placeholder = Kernel(KernelOp.CS, [1], b_kernel.layout)

                if a_kernel.layout.num_ct() == 4:
                    # assumption, a_kernel and b_kernel have their ct-dims row-major packed

                    # create indexed tiles
                    a_ct_offsets = traverse_ct_dims(a_kernel.layout.ct_dims)
                    b_ct_offsets = traverse_ct_dims(b_kernel.layout.ct_dims)

                    # create offsets
                    indexed_a_kernels = []
                    for i, offset in enumerate(a_ct_offsets):
                        indexed_a_layout = Layout(
                            a_kernel.layout.term,
                            a_kernel.layout.rolls,
                            a_kernel.layout.slot_dims,
                            offset,
                            a_kernel.layout.n,
                            a_kernel.layout.secret,
                        )
                        indexed_a_kernel = Kernel(
                            KernelOp.INDEX,
                            [tuple([i]), a_cs_placeholder],
                            indexed_a_layout,
                        )
                        indexed_a_kernels.append(indexed_a_kernel)
                    indexed_b_kernels = []
                    for i, offset in enumerate(b_ct_offsets):
                        indexed_b_layout = Layout(
                            b_kernel.layout.term,
                            b_kernel.layout.rolls,
                            b_kernel.layout.slot_dims,
                            offset,
                            b_kernel.layout.n,
                            b_kernel.layout.secret,
                        )
                        indexed_b_kernel = Kernel(
                            KernelOp.INDEX,
                            [tuple([i]), b_cs_placeholder],
                            indexed_b_layout,
                        )
                        indexed_b_kernels.append(indexed_b_kernel)
                    print(a_kernel)
                    print(b_kernel)
                    print()
                    strassen_kernels = strassens_opt(
                        term, indexed_a_kernels, indexed_b_kernels, roll_flag, network
                    )

                    outputs.add(strassen_kernels)
                    # outputs += strassen_kernels
                elif a_kernel.layout.num_ct() > 4:
                    raise NotImplementedError("more than 4 tiles")
                else:
                    raise NotImplementedError("less than 4 tiles")
    return outputs
