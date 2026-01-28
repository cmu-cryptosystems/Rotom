"""
Test suite for tensor evaluation functionality.

This module tests the tensor.eval() method and related evaluation
functionality in the tensor frontend.
"""

import numpy as np
import pytest

from frontends.tensor import TensorOp, TensorTerm


class TestTensorEvaluation:
    """Test basic tensor evaluation functionality."""

    def test_single_tensor_evaluation(self):
        """Test evaluation of a single tensor term."""
        a = TensorTerm.Tensor("a", [3, 3], True)
        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])}

        result = a.eval(inputs)
        # Tensor gets padded to power of 2: (3,3) -> (4,4)
        expected = np.array([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 0]])

        np.testing.assert_array_equal(result, expected)

    def test_constant_evaluation(self):
        """Test evaluation of a constant tensor term."""
        c = TensorTerm.const(42)
        inputs = {}

        result = c.eval(inputs)
        expected = 42

        assert result == expected

    def test_constant_with_layout_evaluation(self):
        """Test evaluation of a constant with layout parameter."""
        c = TensorTerm.const(42, layout="[0:1:1]")
        inputs = {}

        result = c.eval(inputs)
        expected = 42

        assert result == expected


class TestArithmeticEvaluation:
    """Test arithmetic operations evaluation."""

    def test_addition_evaluation(self):
        """Test addition operation evaluation."""
        a = TensorTerm.Tensor("a", [2, 2], True)
        b = TensorTerm.Tensor("b", [2, 2], True)
        c = a + b

        inputs = {"a": np.array([[1, 2], [3, 4]]), "b": np.array([[5, 6], [7, 8]])}

        result = c.eval(inputs)
        expected = np.array([[6, 8], [10, 12]])

        np.testing.assert_array_equal(result, expected)

    def test_addition_with_layout_evaluation(self):
        """Test addition operation evaluation with layout."""
        a = TensorTerm.Tensor("a", [2, 2], True)
        b = TensorTerm.Tensor("b", [2, 2], True)
        c = a.__add__(b, layout="[0:2:1][1:2:1]")

        inputs = {"a": np.array([[1, 2], [3, 4]]), "b": np.array([[5, 6], [7, 8]])}

        result = c.eval(inputs)
        expected = np.array([[6, 8], [10, 12]])

        np.testing.assert_array_equal(result, expected)

    def test_subtraction_evaluation(self):
        """Test subtraction operation evaluation."""
        a = TensorTerm.Tensor("a", [2, 2], True)
        b = TensorTerm.Tensor("b", [2, 2], True)
        c = a - b

        inputs = {"a": np.array([[10, 8], [6, 4]]), "b": np.array([[1, 2], [3, 4]])}

        result = c.eval(inputs)
        expected = np.array([[9, 6], [3, 0]])

        np.testing.assert_array_equal(result, expected)

    def test_multiplication_evaluation(self):
        """Test element-wise multiplication evaluation."""
        a = TensorTerm.Tensor("a", [2, 2], True)
        b = TensorTerm.Tensor("b", [2, 2], True)
        c = a * b

        inputs = {"a": np.array([[2, 3], [4, 5]]), "b": np.array([[1, 2], [3, 4]])}

        result = c.eval(inputs)
        expected = np.array([[2, 6], [12, 20]])

        np.testing.assert_array_equal(result, expected)


class TestMatrixOperationsEvaluation:
    """Test matrix operations evaluation."""

    def test_matrix_multiplication_evaluation(self):
        """Test matrix multiplication evaluation."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = TensorTerm.Tensor("b", [3, 2], True)
        c = a @ b

        inputs = {
            "a": np.array([[1, 2, 3], [4, 5, 6]]),
            "b": np.array([[7, 8], [9, 10], [11, 12]]),
        }

        result = c.eval(inputs)
        expected = np.array([[58, 64], [139, 154]])

        np.testing.assert_array_equal(result, expected)

    def test_matrix_multiplication_with_layout_evaluation(self):
        """Test matrix multiplication evaluation with layout."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = TensorTerm.Tensor("b", [3, 2], True)
        c = a.matmul(b, layout="[0:2:1][1:2:1]")

        inputs = {
            "a": np.array([[1, 2, 3], [4, 5, 6]]),
            "b": np.array([[7, 8], [9, 10], [11, 12]]),
        }

        result = c.eval(inputs)
        expected = np.array([[58, 64], [139, 154]])

        np.testing.assert_array_equal(result, expected)

    def test_transpose_evaluation(self):
        """Test transpose operation evaluation."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a.T

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then transposed to (4,2)
        expected = np.array([[1, 4], [2, 5], [3, 6], [0, 0]])

        np.testing.assert_array_equal(result, expected)

    def test_transpose_with_layout_evaluation(self):
        """Test transpose operation evaluation with layout."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a.transpose(layout="[1:3:1][0:2:1]")

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then transposed to (4,2)
        expected = np.array([[1, 4], [2, 5], [3, 6], [0, 0]])

        np.testing.assert_array_equal(result, expected)


class TestReductionOperationsEvaluation:
    """Test reduction operations evaluation."""

    def test_sum_evaluation(self):
        """Test sum operation evaluation."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a.sum(0)  # Sum along dimension 0

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then sum along dim 0
        expected = np.array([[5, 7, 9, 0]])  # keepdims=True

        np.testing.assert_array_equal(result, expected)

    def test_sum_with_layout_evaluation(self):
        """Test sum operation evaluation with layout."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a.sum(0, layout="[0:1:1]")

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then sum along dim 0
        expected = np.array([[5, 7, 9, 0]])  # keepdims=True

        np.testing.assert_array_equal(result, expected)

    def test_sum_along_different_dimensions(self):
        """Test sum operation along different dimensions."""
        a = TensorTerm.Tensor("a", [2, 3], True)

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        # Sum along dimension 0
        sum_0 = a.sum(0).eval(inputs)
        expected_0 = np.array([[5, 7, 9, 0]])  # padded to (2,4) first
        np.testing.assert_array_equal(sum_0, expected_0)

        # Sum along dimension 1
        sum_1 = a.sum(1).eval(inputs)
        expected_1 = np.array([[6], [15]])  # keepdims=True, padded input
        np.testing.assert_array_equal(sum_1, expected_1)


class TestShapeOperationsEvaluation:
    """Test shape manipulation operations evaluation."""

    def test_reshape_evaluation(self):
        """Test reshape operation evaluation."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a.reshape(0, {0: 3, 1: 2})

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then reshaped to (4,2) due to power-of-2 rounding
        expected = np.array([[1, 2], [3, 0], [4, 5], [6, 0]])

        np.testing.assert_array_equal(result, expected)

    def test_reshape_with_layout_evaluation(self):
        """Test reshape operation evaluation with layout."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a.reshape(0, {0: 3, 1: 2}, layout="[0:3:1][1:2:1]")

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then reshaped to (4,2) due to power-of-2 rounding
        expected = np.array([[1, 2], [3, 0], [4, 5], [6, 0]])

        np.testing.assert_array_equal(result, expected)

    def test_permute_evaluation(self):
        """Test permute operation evaluation."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a.permute({0: 1, 1: 0})

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then permuted to (4,2)
        expected = np.array([[1, 4], [2, 5], [3, 6], [0, 0]])

        np.testing.assert_array_equal(result, expected)

    def test_permute_with_layout_evaluation(self):
        """Test permute operation evaluation with layout."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a.permute({0: 1, 1: 0}, layout="[1:3:1][0:2:1]")

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then permuted to (4,2)
        expected = np.array([[1, 4], [2, 5], [3, 6], [0, 0]])

        np.testing.assert_array_equal(result, expected)


class TestIndexingEvaluation:
    """Test tensor indexing evaluation."""

    def test_indexing_evaluation(self):
        """Test tensor indexing evaluation."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a[0]  # Get first row

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then indexed
        expected = np.array([1, 2, 3, 0])

        np.testing.assert_array_equal(result, expected)

    def test_indexing_with_layout_evaluation(self):
        """Test tensor indexing evaluation with layout."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a.__getitem__(0, layout="[0:3:1]")

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then indexed
        expected = np.array([1, 2, 3, 0])

        np.testing.assert_array_equal(result, expected)

    def test_slice_indexing_rows(self):
        """Test slice indexing along the first dimension (rows)."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a[0:1]  # slice first row (keep dimension)

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then sliced
        expected = np.array([[1, 2, 3, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_slice_indexing_cols(self):
        """Test slice indexing along the second dimension (columns)."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a[:, 1:3]  # take columns 1 and 2

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then sliced
        expected = np.array([[2, 3], [5, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_mixed_index_and_slice(self):
        """Test mixed integer indexing + slice (e.g., a[1, :])."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = a[1, :]  # second row

        inputs = {"a": np.array([[1, 2, 3], [4, 5, 6]])}

        result = b.eval(inputs)
        # Input gets padded to (2,4), then indexed
        expected = np.array([4, 5, 6, 0])
        np.testing.assert_array_equal(result, expected)


class TestConvolutionEvaluation:
    """Test convolution operation evaluation."""

    def test_conv2d_evaluation(self):
        """Test 2D convolution evaluation."""
        input_tensor = TensorTerm.Tensor("input", [1, 3, 3], True)
        filter_tensor = TensorTerm.Tensor("filter", [1, 1, 2, 2], False)

        conv = TensorTerm.conv2d(input_tensor, filter_tensor, 1, "valid")

        inputs = {
            "input": np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
            "filter": np.array([[[[1, 0], [0, 1]]]]),
        }

        result = conv.eval(inputs)

        # Expected result for valid convolution with 2x2 filter on 3x3 input
        # Input: [[1,2,3], [4,5,6], [7,8,9]]
        # Filter: [[1,0], [0,1]]
        # Result: [[1*1+2*0+4*0+5*1, 2*1+3*0+5*0+6*1],
        #          [4*1+5*0+7*0+8*1, 5*1+6*0+8*0+9*1]]
        #        = [[6, 8], [12, 14]]
        expected = np.array([[[6, 8], [12, 14]]])

        np.testing.assert_array_equal(result, expected)

    def test_conv2d_with_layout_evaluation(self):
        """Test 2D convolution evaluation with layout."""
        input_tensor = TensorTerm.Tensor("input", [1, 3, 3], True)
        filter_tensor = TensorTerm.Tensor("filter", [1, 1, 2, 2], False)

        conv = TensorTerm.conv2d(
            input_tensor, filter_tensor, 1, "valid", layout="[0:1:1][1:2:1][2:2:1]"
        )

        inputs = {
            "input": np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
            "filter": np.array([[[[1, 0], [0, 1]]]]),
        }

        result = conv.eval(inputs)
        expected = np.array([[[6, 8], [12, 14]]])

        np.testing.assert_array_equal(result, expected)


class TestComplexComputationEvaluation:
    """Test evaluation of complex tensor computations."""

    def test_chained_operations_evaluation(self):
        """Test evaluation of chained tensor operations."""
        a = TensorTerm.Tensor("a", [2, 2], True)
        b = TensorTerm.Tensor("b", [2, 2], True)
        c = TensorTerm.Tensor("c", [2, 2], True)

        # Complex computation: (a + b) * c
        result = (a + b) * c

        inputs = {
            "a": np.array([[1, 2], [3, 4]]),
            "b": np.array([[5, 6], [7, 8]]),
            "c": np.array([[2, 1], [3, 2]]),
        }

        eval_result = result.eval(inputs)
        expected = np.array([[12, 8], [30, 24]])

        np.testing.assert_array_equal(eval_result, expected)

    def test_matrix_chain_evaluation(self):
        """Test evaluation of matrix multiplication chain."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        b = TensorTerm.Tensor("b", [3, 4], True)
        c = TensorTerm.Tensor("c", [4, 2], True)

        # Matrix chain: a @ b @ c
        result = a @ b @ c

        inputs = {
            "a": np.array([[1, 2, 3], [4, 5, 6]]),
            "b": np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
            "c": np.array([[1, 1], [1, 1], [1, 1], [1, 1]]),
        }

        eval_result = result.eval(inputs)
        # Expected: (2,3) @ (3,4) @ (4,2) = (2,2)
        # All tensors get padded to power of 2, but operations preserve their natural output shapes
        expected = np.array([[24, 24], [60, 60]])

        np.testing.assert_array_equal(eval_result, expected)

    def test_mixed_operations_with_layouts_evaluation(self):
        """Test evaluation of mixed operations with layout specifications."""
        a = TensorTerm.Tensor("a", [2, 2], True)
        b = TensorTerm.Tensor("b", [2, 2], True)

        # Mixed operations with layouts
        step1 = a.matmul(b, layout="[0:2:1][1:2:1]")
        step2 = step1.transpose(layout="[1:2:1][0:2:1]")
        step3 = step2 + a

        inputs = {"a": np.array([[1, 2], [3, 4]]), "b": np.array([[1, 0], [0, 1]])}

        eval_result = step3.eval(inputs)
        # All operations work on padded inputs but preserve natural output shapes
        expected = np.array([[2, 5], [5, 8]])

        np.testing.assert_array_equal(eval_result, expected)


class TestPowerOfTwoPadding:
    """Test the power-of-two padding functionality."""

    def test_padding_1d_tensor(self):
        """Test padding of 1D tensor to power of 2."""
        a = TensorTerm.Tensor("a", [3], True)

        inputs = {"a": np.array([1, 2, 3])}

        result = a.eval(inputs)
        # Should pad from 3 to 4 (next power of 2)
        expected = np.array([1, 2, 3, 0])

        np.testing.assert_array_equal(result, expected)

    def test_padding_2d_tensor(self):
        """Test padding of 2D tensor to power of 2."""
        a = TensorTerm.Tensor("a", [3, 5], True)

        inputs = {
            "a": np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        }

        result = a.eval(inputs)
        # Should pad from (3,5) to (4,8)
        expected = np.array(
            [
                [1, 2, 3, 4, 5, 0, 0, 0],
                [6, 7, 8, 9, 10, 0, 0, 0],
                [11, 12, 13, 14, 15, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        np.testing.assert_array_equal(result, expected)

    def test_no_padding_already_power_of_two(self):
        """Test that no padding is applied when already power of 2."""
        a = TensorTerm.Tensor("a", [4, 4], True)

        inputs = {
            "a": np.array(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            )
        }

        result = a.eval(inputs)
        expected = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )

        np.testing.assert_array_equal(result, expected)


class TestErrorHandling:
    """Test error handling in evaluation."""

    def test_missing_input_error(self):
        """Test error when required input is missing."""
        a = TensorTerm.Tensor("a", [2, 2], True)

        inputs = {}  # Missing input "a"

        with pytest.raises(KeyError):
            a.eval(inputs)

    def test_not_implemented_operation_error(self):
        """Test error for not implemented operations."""
        # Create a mock operation that's not implemented
        a = TensorTerm(TensorOp.PRODUCT, [TensorTerm.Tensor("a", [2, 2], True), 0])

        inputs = {"a": np.array([[1, 2], [3, 4]])}

        with pytest.raises(NotImplementedError):
            a.eval(inputs)


class TestEdgeCases:
    """Test edge cases in evaluation."""

    def test_empty_tensor_evaluation(self):
        """Test evaluation with empty tensor should raise an error."""
        a = TensorTerm.Tensor("a", [0], True)

        inputs = {"a": np.array([])}

        # Empty tensor should raise an error
        with pytest.raises(ValueError, match="Input must be a positive number"):
            a.eval(inputs)

    def test_scalar_tensor_evaluation(self):
        """Test evaluation with scalar tensor."""
        a = TensorTerm.Tensor("a", [1], True)

        inputs = {"a": np.array([42])}

        result = a.eval(inputs)
        expected = np.array([42])

        np.testing.assert_array_equal(result, expected)

    def test_high_dimensional_tensor_evaluation(self):
        """Test evaluation with high-dimensional tensor (no padding)."""
        a = TensorTerm.Tensor("a", [2, 2, 2], True)

        inputs = {"a": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])}

        result = a.eval(inputs)
        # Should return original tensor (no padding for 3D+)
        expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
