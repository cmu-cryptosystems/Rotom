"""
Binary operation layout generation utilities.

This module provides functions for generating optimal layouts for binary
tensor operations (add, subtract, multiply, matrix multiplication) in FHE
computations. It handles dimension alignment, layout merging, and cost
optimization for various binary operations.

Key functions:
- check_extent_alignment: Validates dimension extent alignment
- match_kernel_dims: Matches dimensions between kernel operands
- gen_binop: Main function for generating binary operation layouts
"""

from assignment.alignment import get_dim_alignment
from assignment.gen.gen_align import (
    apply_sum_rolls,
    match_layout,
    output_layout,
    replicate_dimensions,
)
from assignment.gen.gen_compaction import find_compaction
from frontends.tensor import TensorOp
from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def match_layout_const(dims):
    matched_dims = []
    for dim in dims:
        if dim.dim is not None:
            matched_dims.append(Dim(0, dim.extent, dim.stride))
        else:
            matched_dims.append(Dim(None, dim.extent, dim.stride, dim.dim_type))
    return matched_dims


def gen_binop_const(term, cs_kernels):
    assert term.op in [TensorOp.ADD, TensorOp.SUB, TensorOp.MUL]
    match term.op:
        case TensorOp.ADD:
            kernel_op = KernelOp.ADD
        case TensorOp.SUB:
            kernel_op = KernelOp.SUB
        case TensorOp.MUL:
            kernel_op = KernelOp.MUL

    output_kernels = {}
    if term.cs[0].op == TensorOp.CONST:
        for cs_kernel in cs_kernels[1]:
            dims = cs_kernel.layout.get_dims()
            matched_dims = match_layout_const(dims)
            matched_layout = Layout(
                term.cs[0],
                [],
                matched_dims,
                cs_kernel.layout.n,
                cs_kernel.layout.secret,
            )
            matched_kernel = Kernel(KernelOp.CONST, [], matched_layout)
            if term.cs[0] not in output_kernels:
                output_kernels[term.cs[0]] = set()
            output_kernels[term.cs[0]].add(matched_kernel)

            a_cs_placeholder = Kernel(KernelOp.CS, [0], matched_layout)
            b_cs_placeholder = Kernel(KernelOp.CS, [1], cs_kernel.layout)

            output_layout = Layout(
                term, [], dims, cs_kernel.layout.n, cs_kernel.layout.secret
            )
            output_kernel = Kernel(
                kernel_op, [a_cs_placeholder, b_cs_placeholder], output_layout
            )
            if term not in output_kernels:
                output_kernels[term] = set()
            output_kernels[term].add(output_kernel)
    else:
        for cs_kernel in cs_kernels[0]:
            dims = cs_kernel.layout.get_dims()
            matched_dims = match_layout_const(dims)
            matched_layout = Layout(
                term.cs[1],
                [],
                matched_dims,
                cs_kernel.layout.n,
                cs_kernel.layout.secret,
            )
            matched_kernel = Kernel(KernelOp.CONST, [], matched_layout)
            if term.cs[1] not in output_kernels:
                output_kernels[term.cs[1]] = set()
            output_kernels[term.cs[1]].add(matched_kernel)

            a_cs_placeholder = Kernel(KernelOp.CS, [0], cs_kernel.layout)
            b_cs_placeholder = Kernel(KernelOp.CS, [1], matched_layout)

            output_layout = Layout(
                term, [], dims, cs_kernel.layout.n, cs_kernel.layout.secret
            )
            output_kernel = Kernel(
                kernel_op, [a_cs_placeholder, b_cs_placeholder], output_layout
            )
            if term not in output_kernels:
                output_kernels[term] = set()
            output_kernels[term].add(output_kernel)
    return output_kernels


def gen_binop(term, cs_kernels, shapes, roll_flag):
    if term.cs[0].op == TensorOp.CONST or term.cs[1].op == TensorOp.CONST:
        return gen_binop_const(term, cs_kernels)

    # get alignment
    alignment = get_dim_alignment(term, shapes)

    # replicate dimensions such that tensor dimensions match in extent
    replicated_kernels = []
    for a in cs_kernels[0]:
        for b in cs_kernels[1]:
            # create placeholders
            a_cs_placeholder = Kernel(KernelOp.CS, [0], a.layout)
            b_cs_placeholder = Kernel(KernelOp.CS, [1], b.layout)
            replicated_kernels.append(
                tuple(
                    replicate_dimensions(
                        a_cs_placeholder, b_cs_placeholder, shapes, alignment
                    )
                )
            )

    # if the roll_flag is set, apply rolls to move summation dimensions the vector dimensions
    if roll_flag and term.op == TensorOp.MATMUL:
        replicated_kernels += apply_sum_rolls(term, replicated_kernels)

    output_kernels = set()
    for kernels in replicated_kernels:
        # add conversions or rolls to align layouts
        matched_layouts = match_layout(term, kernels, alignment, roll_flag)

        for matched_a_kernel, matched_b_kernel in matched_layouts:
            # find output layout
            output_kernel = output_layout(
                term, alignment, matched_a_kernel, matched_b_kernel
            )
            # try compacting if no rolls are applied:
            if not output_kernel.layout.rolls:
                compacted_kernel = find_compaction(output_kernel)
                output_kernels.add(compacted_kernel)
            else:
                output_kernels.add(output_kernel)
    # print("len output:", len(output_kernels))
    return {term: output_kernels}
