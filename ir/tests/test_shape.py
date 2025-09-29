#!/usr/bin/env python3
"""
Comprehensive test suite for shape analysis functionality.

This module tests the shape analysis capabilities of the Rotom IR system,
including shape inference, padding to power-of-2 sizes, and shape propagation
through various tensor operations.
"""

import pytest

from frontends.tensor import TensorOp, TensorTerm
from ir.analysis.shape import Shape


class TestBasicOperations:
    """Test basic tensor operations (ADD, SUB, MUL)."""

    def test_element_wise_add_same_shapes(self):
        """Test element-wise addition with tensors of the same shape."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [4, 3], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [4, 3], False])
        add_op = TensorTerm(TensorOp.ADD, [a, b])

        shape_analyzer = Shape(add_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(add_op)
        padded_shape = shape_analyzer.get_padded_shape(add_op)

        assert result_shape == [4, 3]
        assert padded_shape == [4, 4]  # Padded to power of 2

    def test_element_wise_add_broadcasting(self):
        """Test element-wise addition with broadcasting."""
        # Test higher-dimensional tensor with lower-dimensional tensor
        a = TensorTerm(TensorOp.TENSOR, ["a", [2, 4, 3], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [4, 3], False])
        add_op = TensorTerm(TensorOp.ADD, [a, b])

        shape_analyzer = Shape(add_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(add_op)
        assert result_shape == [2, 4, 3]  # Higher dimension is preserved

    def test_element_wise_sub_same_shapes(self):
        """Test element-wise subtraction with tensors of the same shape."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [5, 6], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [5, 6], False])
        sub_op = TensorTerm(TensorOp.SUB, [a, b])

        shape_analyzer = Shape(sub_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(sub_op)
        assert result_shape == [5, 6]

    def test_element_wise_mul_broadcasting(self):
        """Test element-wise multiplication with broadcasting."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [3, 7], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [7], False])
        mul_op = TensorTerm(TensorOp.MUL, [a, b])

        shape_analyzer = Shape(mul_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(mul_op)
        assert result_shape == [3, 7]


class TestMatrixMultiplication:
    """Test matrix multiplication operations."""

    def test_1d_1d_matmul(self):
        """Test 1D × 1D matrix multiplication (dot product)."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [4], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [4], False])
        matmul_op = TensorTerm(TensorOp.MATMUL, [a, b])

        shape_analyzer = Shape(matmul_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(matmul_op)
        assert result_shape == [1]  # Dot product results in scalar (represented as 1D)

    def test_1d_2d_matmul(self):
        """Test 1D × 2D matrix multiplication."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [3], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [3, 5], False])
        matmul_op = TensorTerm(TensorOp.MATMUL, [a, b])

        shape_analyzer = Shape(matmul_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(matmul_op)
        assert result_shape == [1, 5]

    def test_2d_1d_matmul(self):
        """Test 2D × 1D matrix multiplication."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [4, 3], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [3], False])
        matmul_op = TensorTerm(TensorOp.MATMUL, [a, b])

        shape_analyzer = Shape(matmul_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(matmul_op)
        assert result_shape == [4, 1]

    def test_2d_2d_matmul(self):
        """Test 2D × 2D matrix multiplication."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [4, 3], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [3, 2], False])
        matmul_op = TensorTerm(TensorOp.MATMUL, [a, b])

        shape_analyzer = Shape(matmul_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(matmul_op)
        assert result_shape == [4, 2]

    def test_3d_2d_matmul(self):
        """Test 3D × 2D matrix multiplication (batched)."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [2, 4, 3], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [3, 5], False])
        matmul_op = TensorTerm(TensorOp.MATMUL, [a, b])

        shape_analyzer = Shape(matmul_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(matmul_op)
        assert result_shape == [2, 4, 5]

    def test_4d_2d_matmul(self):
        """Test 4D × 2D matrix multiplication."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [1, 2, 3, 4], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [4, 5], False])
        matmul_op = TensorTerm(TensorOp.MATMUL, [a, b])

        shape_analyzer = Shape(matmul_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(matmul_op)
        assert result_shape == [1, 2, 3, 5]

    def test_block_matmul(self):
        """Test block matrix multiplication."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [2, 3, 4], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [2, 4, 5], False])
        block_matmul_op = TensorTerm(TensorOp.BLOCK_MATMUL, [a, b])

        shape_analyzer = Shape(block_matmul_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(block_matmul_op)
        assert result_shape == [2, 3, 5]

    def test_matmul_invalid_shapes(self):
        """Test that invalid matrix multiplication shapes raise errors."""
        # Test mismatched inner dimensions
        a = TensorTerm(TensorOp.TENSOR, ["a", [4, 3], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [5, 2], False])
        matmul_op = TensorTerm(TensorOp.MATMUL, [a, b])

        shape_analyzer = Shape(matmul_op)

        with pytest.raises(AssertionError):
            shape_analyzer.run()

    def test_matmul_scalar_tensors(self):
        """Test that matrix multiplication with scalar tensors raises errors."""
        # Test scalar × tensor
        a = TensorTerm(TensorOp.TENSOR, ["a", [], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [4], False])
        matmul_op = TensorTerm(TensorOp.MATMUL, [a, b])

        shape_analyzer = Shape(matmul_op)

        with pytest.raises(ValueError, match="Cannot perform matmul on scalar tensors"):
            shape_analyzer.run()


class TestShapeManipulation:
    """Test shape manipulation operations."""

    def test_transpose_2d(self):
        """Test transpose operation on 2D tensor."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [3, 4], True])
        transpose_op = TensorTerm(TensorOp.TRANSPOSE, [a])

        shape_analyzer = Shape(transpose_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(transpose_op)
        assert result_shape == [4, 3]

    def test_transpose_invalid_dimensions(self):
        """Test that transpose on non-2D tensors raises errors."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [3, 4, 5], True])
        transpose_op = TensorTerm(TensorOp.TRANSPOSE, [a])

        shape_analyzer = Shape(transpose_op)

        with pytest.raises(AssertionError):
            shape_analyzer.run()

    def test_reshape(self):
        """Test reshape operation."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [2, 3, 4], True])
        # Reshape: remove dimension 1, add new dimensions
        # Original shape: [2, 3, 4] with indices [0, 1, 2]
        # Remove dim 1: shape_map becomes {0: 2, 2: 4}
        # Add new dims: {1: 6, 3: 4} -> shape_map becomes {0: 2, 1: 6, 2: 4, 3: 4}
        # Final shape: [2, 6, 4, 4]
        reshape_op = TensorTerm(TensorOp.RESHAPE, [a, 1, {1: 6, 3: 4}])

        shape_analyzer = Shape(reshape_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(reshape_op)
        assert result_shape == [2, 6, 4, 4]

    def test_permute(self):
        """Test dimension permutation."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [2, 3, 4], True])
        # Permute: original [2, 3, 4] -> [4, 2, 3]
        # This means: dim 0 (size 2) goes to position 1, dim 1 (size 3) goes to position 2, dim 2 (size 4) goes to position 0
        permute_op = TensorTerm(TensorOp.PERMUTE, [a, {0: 1, 1: 2, 2: 0}])

        shape_analyzer = Shape(permute_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(permute_op)
        assert result_shape == [4, 2, 3]


class TestReductionOperations:
    """Test reduction operations (SUM, PRODUCT)."""

    def test_sum_along_dimension(self):
        """Test sum operation along a specific dimension."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [2, 3, 4], True])
        sum_op = TensorTerm(TensorOp.SUM, [a, 1])  # Sum along dimension 1

        shape_analyzer = Shape(sum_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(sum_op)
        assert result_shape == [2, 4]  # Dimension 1 (size 3) is removed

    def test_product_along_dimension(self):
        """Test product operation along a specific dimension."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [3, 4, 5], True])
        product_op = TensorTerm(TensorOp.PRODUCT, [a, 0])  # Product along dimension 0

        shape_analyzer = Shape(product_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(product_op)
        assert result_shape == [4, 5]  # Dimension 0 (size 3) is removed

    def test_sum_first_dimension(self):
        """Test sum operation along the first dimension."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [5, 3, 2], True])
        sum_op = TensorTerm(TensorOp.SUM, [a, 0])

        shape_analyzer = Shape(sum_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(sum_op)
        assert result_shape == [3, 2]

    def test_product_last_dimension(self):
        """Test product operation along the last dimension."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [4, 3, 6], True])
        product_op = TensorTerm(TensorOp.PRODUCT, [a, 2])

        shape_analyzer = Shape(product_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(product_op)
        assert result_shape == [4, 3]


class TestConvolutionOperations:
    """Test convolution operations."""

    def test_conv2d_valid_padding(self):
        """Test 2D convolution with valid padding."""
        # Input: [channels_in, height, width]
        a = TensorTerm(TensorOp.TENSOR, ["a", [3, 8, 8], True])
        # Filter: [channels_out, filter_height, filter_width]
        b = TensorTerm(TensorOp.TENSOR, ["b", [16, 3, 3], False])
        conv_op = TensorTerm(TensorOp.CONV2D, [a, b, 1, "valid"])

        shape_analyzer = Shape(conv_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(conv_op)
        # Output height = (8 - 3) // 1 + 1 = 6
        # Output width = (8 - 3) // 1 + 1 = 6
        assert result_shape == [16, 6, 6]

    def test_conv2d_same_padding(self):
        """Test 2D convolution with same padding."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [1, 5, 5], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [8, 2, 2], False])
        conv_op = TensorTerm(TensorOp.CONV2D, [a, b, 2, "same"])

        shape_analyzer = Shape(conv_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(conv_op)
        # With same padding, output size equals input size
        assert result_shape == [8, 5, 5]

    def test_conv2d_stride_2(self):
        """Test 2D convolution with stride 2."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [2, 10, 10], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [4, 3, 3], False])
        conv_op = TensorTerm(TensorOp.CONV2D, [a, b, 2, "valid"])

        shape_analyzer = Shape(conv_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(conv_op)
        # Output height = (10 - 3) // 2 + 1 = 4
        # Output width = (10 - 3) // 2 + 1 = 4
        assert result_shape == [4, 4, 4]


class TestIndexingOperations:
    """Test indexing operations."""

    def test_index_operation(self):
        """Test index operation."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [2, 3, 4], True])
        index_op = TensorTerm(TensorOp.INDEX, [a, 0])  # Index along dimension 0

        shape_analyzer = Shape(index_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(index_op)
        assert result_shape == [3, 4]  # First dimension is removed


class TestPowerOfTwoPadding:
    """Test power-of-2 padding functionality."""

    def test_padding_basic(self):
        """Test basic power-of-2 padding."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [3, 5], True])

        shape_analyzer = Shape(a)
        shape_analyzer.run()

        padded_shape = shape_analyzer.get_padded_shape(a)
        assert padded_shape == [4, 8]  # 3 -> 4, 5 -> 8

    def test_padding_already_power_of_2(self):
        """Test padding when dimensions are already powers of 2."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [4, 8, 16], True])

        shape_analyzer = Shape(a)
        shape_analyzer.run()

        padded_shape = shape_analyzer.get_padded_shape(a)
        assert padded_shape == [4, 8, 16]  # No change needed

    def test_padding_complex_operation(self):
        """Test padding with complex operations."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [3, 5], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [3, 5], False])
        add_op = TensorTerm(TensorOp.ADD, [a, b])

        shape_analyzer = Shape(add_op)
        shape_analyzer.run()

        padded_shape = shape_analyzer.get_padded_shape(add_op)
        assert padded_shape == [4, 8]

    def test_padding_matmul(self):
        """Test padding with matrix multiplication."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [3, 5], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [5, 7], False])
        matmul_op = TensorTerm(TensorOp.MATMUL, [a, b])

        shape_analyzer = Shape(matmul_op)
        shape_analyzer.run()

        padded_shape = shape_analyzer.get_padded_shape(matmul_op)
        assert padded_shape == [4, 8]  # 3 -> 4, 7 -> 8


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_tensor(self):
        """Test operations with empty tensors."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [], True])

        shape_analyzer = Shape(a)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(a)
        assert result_shape == []

    def test_scalar_tensor(self):
        """Test operations with scalar tensors."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [1], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [1], False])
        add_op = TensorTerm(TensorOp.ADD, [a, b])

        shape_analyzer = Shape(add_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(add_op)
        assert result_shape == [1]

    def test_large_dimensions(self):
        """Test operations with large dimensions."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [100, 200], True])

        shape_analyzer = Shape(a)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(a)
        assert result_shape == [100, 200]

        padded_shape = shape_analyzer.get_padded_shape(a)
        assert padded_shape == [128, 256]  # 100 -> 128, 200 -> 256

    def test_unsupported_operation(self):
        """Test that unsupported operations raise NotImplementedError."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [3, 4], True])
        # Create an operation with an unsupported TensorOp
        unsupported_op = TensorTerm(
            TensorOp.POLY, [a]
        )  # POLY is not implemented in shape analysis

        shape_analyzer = Shape(unsupported_op)

        with pytest.raises(NotImplementedError):
            shape_analyzer.run()


class TestComplexCompositions:
    """Test complex compositions of operations."""

    def test_matmul_transpose_composition(self):
        """Test composition of matrix multiplication and transpose."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [4, 3], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [2, 3], False])
        matmul_op = TensorTerm(TensorOp.MATMUL, [a, b.T])
        transpose_op = TensorTerm(TensorOp.TRANSPOSE, [matmul_op])

        shape_analyzer = Shape(transpose_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(transpose_op)
        assert result_shape == [2, 4]

    def test_add_reshape_composition(self):
        """Test composition of addition and reshape."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [2, 3, 4], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [2, 3, 4], False])
        add_op = TensorTerm(TensorOp.ADD, [a, b])
        # Reshape: remove dimension 1, add new dimensions
        # Original [2, 3, 4] -> remove dim 1 -> {0: 2, 2: 4} -> add dims -> {0: 2, 1: 6, 2: 4, 3: 4}
        # Final shape: [2, 6, 4, 4]
        reshape_op = TensorTerm(TensorOp.RESHAPE, [add_op, 1, {1: 6, 3: 4}])

        shape_analyzer = Shape(reshape_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(reshape_op)
        assert result_shape == [2, 6, 4, 4]

    def test_conv_sum_composition(self):
        """Test composition of convolution and sum."""
        a = TensorTerm(TensorOp.TENSOR, ["a", [3, 8, 8], True])
        b = TensorTerm(TensorOp.TENSOR, ["b", [16, 3, 3], False])
        conv_op = TensorTerm(TensorOp.CONV2D, [a, b, 1, "valid"])
        sum_op = TensorTerm(TensorOp.SUM, [conv_op, 0])  # Sum over channels

        shape_analyzer = Shape(sum_op)
        shape_analyzer.run()

        result_shape = shape_analyzer.get_shape(sum_op)
        assert result_shape == [6, 6]  # Channels dimension removed
