#!/usr/bin/env python3
"""
Test script to verify the generalized matmul shape checking works correctly.
This tests various tensor dimension combinations that should now be supported.
"""

import numpy as np

from frontends.tensor import TensorOp, TensorTerm
from ir.analysis.shape import Shape


def test_generalized_matmul_shapes():
    """Test the generalized matmul shape checking with various tensor dimensions."""

    # Test cases: (a_shape, b_shape, expected_result_shape)
    test_cases = [
        # Original cases that should still work
        ([4], [4], [1]),  # 1D × 1D (dot product)
        ([4], [4, 3], [1, 3]),  # 1D × 2D
        ([4, 3], [3], [4, 1]),  # 2D × 1D
        ([4, 3], [3, 2], [4, 2]),  # 2D × 2D
        ([2, 4, 3], [3, 2], [2, 4, 2]),  # 3D × 2D
        # New generalized cases
        ([2, 3, 4], [4, 5], [2, 3, 5]),  # 3D × 2D
        ([1, 2, 3, 4], [4, 5], [1, 2, 3, 5]),  # 4D × 2D
        ([2, 3, 4], [2, 4, 5], [2, 3, 2, 5]),  # 3D × 3D
        ([1, 2, 3, 4], [1, 4, 5], [1, 2, 3, 1, 5]),  # 4D × 3D
        ([5, 6, 7, 8], [8, 9], [5, 6, 7, 9]),  # 4D × 2D
    ]

    print("Testing generalized matmul shape checking...")

    for i, (a_shape, b_shape, expected) in enumerate(test_cases):
        print(f"\nTest case {i+1}: {a_shape} × {b_shape} = {expected}")

        try:
            # Create tensor terms
            a = TensorTerm.Tensor("a", a_shape, True)
            b = TensorTerm.Tensor("b", b_shape, True)

            # Create matmul operation
            matmul_op = a @ b

            # Run shape analysis
            shape_analyzer = Shape(matmul_op)
            shape_analyzer.run()

            # Get the result shape
            result_shape = shape_analyzer.get_shape(matmul_op)
            padded_shape = shape_analyzer.get_padded_shape(matmul_op)

            print(f"  Result shape: {result_shape}")
            print(f"  Padded shape: {padded_shape}")

            # Verify the shape matches expected
            assert result_shape == expected, f"Expected {expected}, got {result_shape}"
            print(f"  ✓ PASS")

        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            raise

    print(f"\n✓ All {len(test_cases)} test cases passed!")


def test_invalid_matmul_shapes():
    """Test that invalid matmul shapes are properly rejected."""

    invalid_cases = [
        ([3], [5], "Mismatched dimensions"),  # 1D × 1D with different sizes
        (
            [4, 3],
            [5, 2],
            "Mismatched inner dimensions",
        ),  # 2D × 2D with mismatched inner dims
        (
            [2, 3, 4],
            [5, 6],
            "Mismatched inner dimensions",
        ),  # 3D × 2D with mismatched inner dims
        ([], [4], "Scalar tensor"),  # Scalar × tensor
        ([4], [], "Scalar tensor"),  # Tensor × scalar
    ]

    print("\nTesting invalid matmul shapes (should raise errors)...")

    for i, (a_shape, b_shape, description) in enumerate(invalid_cases):
        print(f"\nInvalid test case {i+1}: {a_shape} × {b_shape} ({description})")

        try:
            # Create tensor terms
            a = TensorTerm.Tensor("a", a_shape, True)
            b = TensorTerm.Tensor("b", b_shape, True)

            # Create matmul operation
            matmul_op = a @ b

            # Run shape analysis - this should raise an error
            shape_analyzer = Shape(matmul_op)
            shape_analyzer.run()

            print(f"  ✗ FAIL: Should have raised an error but didn't")

        except Exception as e:
            print(f"  ✓ PASS: Correctly raised error: {e}")
